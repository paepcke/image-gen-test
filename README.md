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
needed), downloads the pretrained weights, and does an editable
`pip install -e .` of this repo's own packages (`common`,
`cloud_img_procurement`, `image_gen`).

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
each image is a billed OpenAI API call:

```bash
conda run -n therapist-img-gen python src/cloud_img_procurement/client_library_generator.py \
    --max-buckets 2 --images-per-bucket 1
```

Check the 2 resulting images (under `assets/client_library/`) look
right, then run the full 30-bucket library:

```bash
conda run -n therapist-img-gen python src/cloud_img_procurement/client_library_generator.py \
    --images-per-bucket 4
```

Resumable: re-running with the same arguments skips any image file
that already exists, unless you pass `--force`.

## 5. Validate the library

Confirms LivePortrait's face cropper detects a face in every
generated photo, offline, rather than discovering a bad one mid
trainee-session:

```bash
conda run -n therapist-img-gen python src/cloud_img_procurement/validate_library.py --gpu 0
```

Check the tail of the output for `Passed: N  Failed: 0`. Note the
`--gpu` argparse choices are hardcoded `[0, 1, 2]` from the
three-GPU quatro era -- pass whichever index(es) this machine
actually has (sextus has one GPU: use `--gpu 0`; quintus has two:
`--gpu 0` or `--gpu 1`).

## 6. Generate the posture x emotion matrix (one-time, offline)

This combines each bucket photo's 3 posture variants (generated via
OpenAI edits, cached next to the base photo) with every emotion
driving image under `assets/`, via LivePortrait. Smoke-test first:

```bash
conda run -n therapist-img-gen python src/cloud_img_procurement/posture_expression_combiner.py \
    --gpu 0 --max-buckets 1
```

Check `assets/client_library_animated/<bucket>/<image>/<posture>/`
for that one bucket, and confirm the startup log line reads
`Found 8 driving emotion images: [...]` (not 9 -- if you see a 9th
entry, you're on a version of this script from before that bug was
fixed; pull latest). Then run the full library:

```bash
conda run -n therapist-img-gen python src/cloud_img_procurement/posture_expression_combiner.py --gpu 0
```

This is the slow, expensive step (30 buckets x 4 images x 3 postures
x 8 emotions), so let it run unattended once the smoke test looks
right. It's resumable for posture variants (skips existing files
unless `--force-posture`), but re-runs every LivePortrait animation
each time it's invoked for a given bucket -- there's no
skip-if-exists on that side yet.

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
```

## Where this fits with therapist_trainer

This repo is intentionally standalone -- not a subpackage of
therapist_trainer -- while the pipeline is still being shaped. The
`assets/client_library*/` trees are what eventually gets pulled into
therapist_trainer's simulated-client feature; nothing here is a
runtime dependency of that app yet.
