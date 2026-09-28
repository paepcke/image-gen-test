# therapist-img-gen

LivePortrait-based client photo/expression generation for the
TherapistTrainer platform. Produces a bucketed library of synthetic
client photos (race x sex x age range), each with 3 posture variants
(neutral, shoulders_raised, fists_clenched) generated via GPT Image
2.5 edits, then animated per emotion via LivePortrait -- so a trainee
session can show one consistent client with a torso posture and a
facial expression that vary independently across the conversation.

This README covers bringing up the pipeline on a *new* GPU machine
(sextus, quintus, or any future one) from a clean clone. It assumes
you've already done this once elsewhere (quatro), so it's written as
a checklist, not a tutorial.

## 0. Before you start

- **NVIDIA GPU + driver.** Check with `nvidia-smi`. Anything Turing
  generation or newer (RTX 2080 Ti, Titan RTX, RTX 5000 Ada, etc.)
  works -- the only thing older/other architectures might lack is
  FlashAttention-2, which LivePortrait doesn't need.
- **conda**, already installed.
- **An OpenAI API key file at `$HOME/.ssh/openai_api_key.txt`** on
  *this* machine specifically -- line 1 is the API key, line 2
  (optional) is the organization id. This is per-machine, not
  synced via git; copy it over from another machine if you don't
  have one here yet. Nothing that calls the OpenAI API (posture
  generation/edits) works without it.
- **Outbound network access to `api.openai.com`.** Sanity-check with
  `curl -sS https://api.openai.com` if this machine's network is
  unusual (e.g. behind a restrictive institutional firewall).

## 1. Clone the repo

```bash
git clone git@github.com:paepcke/therapist-img-gen.git
cd therapist-img-gen
```

## 2. Run the setup script

```bash
bash src/bash/setup_env.sh
```

This one script creates the conda env (`python=3.10`), installs
torch for CUDA 12.x, clones `third_party/LivePortrait`, installs its
`requirements.txt`, installs `ffmpeg` via conda-forge (no sudo
needed), downloads the pretrained weights (scoping `HF_HUB_OFFLINE=0`
around just that command, in case this machine's shell sets
`HF_HUB_OFFLINE=1` globally for other local-model work), and does an
editable `pip install -e .` of this repo's own packages (`common`,
`cloud_img_procurement`, `image_gen`).

Confirm the weights actually landed -- this is the one step in the
script most likely to fail silently (network hiccup, disk quota,
etc. without failing the script itself):

```bash
ls -la third_party/LivePortrait/pretrained_weights/liveportrait/base_models/
```

You should see non-empty `.pth` files: `appearance_feature_extractor.pth`,
`motion_extractor.pth`, `warping_module.pth`, `spade_generator.pth`
(plus `pretrained_weights/liveportrait/retargeting_models/`).

**If it's empty or missing** (just a `.gitkeep`), re-run the download
step by hand as a remedy. The most likely cause is `HF_HUB_OFFLINE=1`
being set in this shell (`setup_env.sh` already scopes around it, but
double-check if this fails again):

```bash
env | grep HF_HUB   # check whether HF_HUB_OFFLINE=1 is set here

cd third_party/LivePortrait
HF_HUB_OFFLINE=0 conda run -n therapist-img-gen python -m huggingface_hub.commands.huggingface_cli \
    download KwaiVGI/LivePortrait --local-dir . --exclude "*.git*"
cd ../..
```

This pulls several GB, so check disk space and network reachability
to `huggingface.co` first if this machine's setup is unusual.

## 3. Fix the onnxruntime-gpu / CUDA mismatch

LivePortrait's own `requirements.txt` pins `onnxruntime-gpu==1.18.0`,
whose default PyPI wheel is linked against **CUDA 11**, not 12 --
but `setup_env.sh` just installed torch for CUDA 12.x. Left as-is,
face detection/landmark steps (which use onnxruntime, separately
from the torch-based animation networks) silently fall back to CPU,
with errors like:

```
[E:onnxruntime...] Failed to load library libonnxruntime_providers_cuda.so
with error: libcublasLt.so.11: cannot open shared object file
```

Fix (no sudo, no symlinks -- just upgrade the one package):

```bash
conda run -n therapist-img-gen python -m pip install -U "onnxruntime-gpu>=1.19.0"
```

(Microsoft switched the default PyPI `onnxruntime-gpu` wheel to a
CUDA-12-linked build starting at 1.19.0.) You'll still see one
harmless warning line about `/sys/class/drm/card0/device/vendor` --
that's an unrelated device-enumeration probe and doesn't affect the
CUDA execution provider.

## 4. Generate the client photo library (one-time, offline)

Never run this with the full defaults first -- smoke-test it, since
each image is a billed OpenAI API call (~20 seconds):

```bash
conda run -n therapist-img-gen python src/cloud_img_procurement/client_library_generator.py \
    --max-buckets 2 --images-per-bucket 1
```

Check the 2 resulting images (under `assets/client_library/`) look
right, then run the full 30-bucket library ~(17 minutes):

```bash
time conda run -n therapist-img-gen python src/cloud_img_procurement/client_library_generator.py \
    --images-per-bucket 4
```

Resumable: re-running with the same arguments skips any image file
that already exists, unless you pass `--force`.

## 5. Validate the library

Confirms LivePortrait's face cropper detects a face in every
generated photo, offline, rather than discovering a bad one mid
trainee-session (~8 secs):

```bash
time conda run -n therapist-img-gen python src/cloud_img_procurement/validate_library.py --gpu 0
```

Check the tail of the output for `Passed: N  Failed: 0`. Note the
`--gpu` argparse choices are hardcoded `[0, 1, 2]` from the
three-GPU quatro era -- pass whichever index(es) this machine
actually has (sextus has one GPU: use `--gpu 0`; quintus has two:
`--gpu 0` or `--gpu 1`).

## 6. Generate the posture x emotion matrix (one-time, offline)

This combines each bucket photo's 3 posture variants (generated via
OpenAI edits, cached next to the base photo) with every emotion
driving image under `assets/`, via LivePortrait. Smoke-test first  (~2min5secs):

```bash
time conda run -n therapist-img-gen python src/cloud_img_procurement/posture_expression_combiner.py \
    --gpu 0 --max-buckets 1
```

Check `assets/client_library_animated/<bucket>/<image>/<posture>/`
for that one bucket, and confirm the startup log line reads
`Found 8 driving emotion images: [...]` (not 9 -- if you see a 9th
entry, you're on a version of this script from before that bug was
fixed; pull latest). Then run the full library (31 minutes):

```bash
time conda run -n therapist-img-gen python src/cloud_img_procurement/posture_expression_combiner.py --gpu 0
```

This is the slow, expensive step (30 buckets x 4 images x 3 postures
x 8 emotions), so let it run unattended once the smoke test looks
right. It's resumable for posture variants (skips existing files
unless `--force-posture`), but re-runs every LivePortrait animation
each time it's invoked for a given bucket -- there's no
skip-if-exists on that side yet.

A single bad OpenAI edit (moderation rejection, etc.) or LivePortrait
animation no longer aborts the whole run -- it's recorded and
skipped, and the rest of the matrix keeps going. Check
`assets/client_library_animated/combine_failures.json` afterward for
anything that was skipped, and `combine_manifest.json` for the full
posture/emotion -> output-file map (a skipped entry is `{"_skipped":
...}` at the posture level, or `null` at the individual emotion
level).

## 7. Validate the posture x emotion matrix

Same idea as step 5, but for the combiner's output: confirms
LivePortrait's face cropper detects a face in every animated frame
the manifest says exists, catching a corrupt or degenerate animation
offline. Driven from `combine_manifest.json`, so it already knows
which combos the combiner itself skipped and doesn't re-flag those as
failures (46 secs):

```bash
time conda run -n therapist-img-gen python src/cloud_img_procurement/validate_combinations.py --gpu 0
```

Check the tail of the output for `Passed: N  Failed: 0`. `Failed`
entries and their paths land in
`assets/client_library_animated/combine_validation_report.json`, so
you can pull them up in the gallery (`image_gallery_generator.py`) by
filename and decide whether to regenerate them.

## 8. Browse the results in the gallery

`validate_combinations.py` only catches outright failures (no face
detected); it says nothing about whether an animation actually looks
good. For that, eyeball everything in a plain static HTML gallery --
no server, no dependencies beyond the stdlib, works the same whether
you open it right where the images were generated or after copying
`assets/` elsewhere:

```bash
conda run -n therapist-img-gen python src/cloud_img_procurement/image_gallery_generator.py \
    --root assets/client_library_animated
```

Then open `assets/client_library_animated/gallery.html` in any
browser. Images are grouped in collapsible sections by folder
(bucket/image/posture), shown large with the bare filename
underneath (no path -- handy for `find assets/ -name <filename>`), a
text box live-filters by filename substring, and a "redo" checkbox
under each image lets you mark ones to revisit -- "Copy marked
filenames" puts the checked bare filenames on the clipboard.

If a specific posture x emotion combo comes out too intense (or too
flat) rather than outright broken, there's no need to redo the whole
bucket or re-run the full matrix -- `reanimate_combo.py` regenerates
just that combo in place, with an overridden `driving_multiplier`,
at whatever output path the combiner already wrote:

```bash
conda run -n therapist-img-gen python src/cloud_img_procurement/reanimate_combo.py \
    --bucket asian_female_20s-30s --image 00 --posture all \
    --emotion Contempt --multiplier 1.3 --gpu 0
```

`--posture all` loops over neutral/shoulders_raised/fists_clenched in
one call; pass a single posture to redo just one. This overwrites the
existing file at the same path, so no manifest update is needed
afterward. Note `--image` selects one specific bucket image (e.g.
`00`) -- `--posture all` only fans out across postures *for that one
image*, not across every image in the bucket.

To fix a whole batch at once: mark the offending images with the
gallery's "redo" checkboxes, "Copy marked filenames", paste that list
into a file (or pipe it straight in), and redo all of them in one
call, at the same `--multiplier`:

```bash
pbpaste | conda run -n therapist-img-gen python src/cloud_img_procurement/reanimate_combo.py \
    --worklist - --multiplier 1.3 --gpu 0
```

Each line is parsed back into its bucket/image/posture/emotion from
the filename itself, so a worklist can freely mix combos from
different buckets, images, and even emotions -- only the multiplier
is shared. Malformed lines are skipped with a warning rather than
aborting the batch.

## Where things end up

```
assets/
  client_library/                    <bucket>/<bucket>_NN.png            (neutral base photos)
                                      <bucket>/<bucket>_NN_<posture>.png  (posture edits)
                                      manifest.json
                                      validation_report.json
  client_library_animated/
                                      <bucket>/<bucket>_NN/<posture>/<bucket>_NN--<emotion>.jpg
                                      <bucket>/<bucket>_NN/<posture>/<bucket>_NN--<emotion>_concat.jpg
                                      combine_manifest.json
                                      combine_failures.json           (per-image/combo failures, if any)
                                      combine_validation_report.json  (from validate_combinations.py)
                                      gallery.html                    (from image_gallery_generator.py --root
                                                                        assets/client_library_animated)
```

## Where this fits with therapist_trainer

This repo is intentionally standalone -- not a subpackage of
therapist_trainer -- while the pipeline is still being shaped. The
`assets/client_library*/` trees are what eventually gets pulled into
therapist_trainer's simulated-client feature; nothing here is a
runtime dependency of that app yet.
