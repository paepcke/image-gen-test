#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-09-26 17:13:17
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/cloud_img_procurement/posture_expression_combiner.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-09-28 17:51:56
# **********************************************************

"""
Combines the two orthogonal axes of the simulated-client pipeline into
one full matrix per bucket photo:

  * posture (torso/hands) -- produced by GPT Image 2.5 edits, via
    cloud_img_procurement.posture_variant_generator.PostureVariantGenerator
  * facial expression -- produced by LivePortrait, via
    image_gen.live_portrait_service.LivePortraitService

These two stages don't interfere with each other: LivePortraitService
runs in single-image-driving mode here, and with a single-frame driving
image, LivePortrait's frame-0 pose is used as its own baseline, so the
source's original head pose is preserved exactly (see
live_portrait_service.py's own generate() docstring) -- meaning it never
touches the torso or hands a posture edit put there. So each posture
variant of a bucket photo (neutral, shoulders_raised, fists_clenched)
can be used as a LivePortrait *source* for every emotion driving image,
independently, without one stage undoing the other's work (see chat).

Per bucket photo this produces 3 postures x N driving emotions distinct
animated clips, all sharing one identity.

Per-emotion driving_multiplier values below are the ones calibrated in
an earlier session against a reference client photo (see chat / project
memory) -- not the LivePortrait or ArgumentConfig defaults. 'fear' has
no calibrated value (that session found 'anxiety' more clinically
relevant and calibrated that instead), so it falls back to 1.0; adjust
FEAR_FALLBACK_MULTIPLIER below if you calibrate it.

Every posture also gets a "Neutral" emotion entry -- the un-animated
posture source image itself (a file copy, no LivePortrait/GPU call),
so "posture tense, face neutral" (e.g. shoulders_raised + Neutral) is
its own addressable combo, not just available for posture=neutral.
This completes the (bucket, posture, emotion) lookup grid the
therapist_trainer integration relies on -- see chat / project memory
(the ThTrainer image-integration design doc) -- rather than leaving
posture=neutral as a special case with a differently-shaped directory.

CAVEAT -- where this needs to run: LivePortraitService needs one of
quatro's GPUs and the third_party/LivePortrait checkout, so this script
is a quatro-only script, same as validate_library.py. Unless
--skip-posture is passed, it *also* calls the OpenAI API for each
posture edit, so quatro needs outbound network access to OpenAI for a
full run. If quatro can't reach OpenAI, pre-generate the posture
variants elsewhere with posture_variant_generator.py, copy the
resulting *_shoulders_raised.png / *_fists_clenched.png files into
assets/client_library/<bucket>/ alongside the bases, then run this
script with --skip-posture so it only drives LivePortrait.

Lives at <proj-root>/src/cloud_img_procurement/posture_expression_combiner.py --
alongside the other batch-asset-production scripts (client_library_generator.py,
posture_variant_generator.py, validate_library.py), even though it needs a GPU
and imports image_gen.live_portrait_service to do its job -- validate_library.py
already crosses that same boundary, so this follows the established pattern
rather than being grouped by which library it happens to import.

Requires the editable install from setup_env.sh (`pip install -e .`).

Usage:
    conda run -n therapist-img-gen python src/cloud_img_procurement/posture_expression_combiner.py \\
        --gpu 0 --max-buckets 1

    # Only process one race (repeatable), e.g. after adding it to the
    # Race enum. Existing posture variants are reused, and
    # combine_manifest.json / combine_failures.json are merged rather
    # than replaced, so other races' entries are preserved:
    conda run -n therapist-img-gen python src/cloud_img_procurement/posture_expression_combiner.py \\
        --gpu 0 --race middle_eastern

(A tree generated before NEUTRAL_EMOTION_LABEL support existed can be
migrated with the one-time, throw-away backfill_neutral_once.py at the
project root -- see that file; it's not part of this script's CLI.)
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

from openai import OpenAIError

from cloud_img_procurement.bucket_enums import Race
from cloud_img_procurement.posture_variant_generator import (
    POSTURE_EDIT_PROMPTS, PostureVariantGenerator,
)
from image_gen.live_portrait_service import LivePortraitService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("image_gen")

PROJ_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ROOT = PROJ_ROOT / "assets" / "client_library"
DRIVING_IMAGES_DIR = PROJ_ROOT / "assets"
ANIMATED_ROOT = PROJ_ROOT / "assets" / "client_library_animated"

# Calibrated against a reference client photo (see chat / project
# memory: /areas/liveportrait-multipliers.md). Keys are driving-image
# stems (case-insensitive match against DRIVING_IMAGES_DIR/*.png).
DRIVING_MULTIPLIERS = {
    "sadness": 2.0,
    "contempt": 2.0,
    "surprise": 1.5,
    "anger": 2.0,
    "happiness": 0.5,
    "anxiety": 1.5,
    "disgust": 1.5,
}
FEAR_FALLBACK_MULTIPLIER = 1.0  # not separately calibrated -- see docstring

# The emotion label used for a posture's un-animated source image, kept
# alongside the actual driving-image emotions in both the output
# directory and combine_manifest.json (see module docstring). Matches
# the capitalization convention of the driving images themselves
# (Anger.png, Happiness.png, ...) even though matching elsewhere in
# this project is case-insensitive.
NEUTRAL_EMOTION_LABEL = "Neutral"

# Only files with one of these (case-insensitive) stems are treated as
# emotion driving images by _discover_driving_images(). assets/ also
# holds non-emotion photos -- e.g. clientInTherapyOfficeIsolated.png,
# the earlier single-photo demo's source image -- and a blind glob of
# every *.png under assets/ would sweep those in as a bogus "emotion"
# too (see chat: this bit us on the first real smoke-test run).
KNOWN_EMOTIONS = set(DRIVING_MULTIPLIERS) | {"fear"}

# A "base" bucket image is named '..._NN.png' (client_bucket.py's
# image_filename()); a posture variant of it is named
# '..._NN_<posture>.png'. This tells the two apart when re-globbing a
# bucket directory that already contains generated variants, so a
# variant is never mistaken for a new base image to process.
BASE_IMAGE_STEM_PATTERN = re.compile(r"_\d{2}$")


def _to_manifest_path(path: Path) -> str:
    """Renders a path for storage in combine_manifest.json, relative to PROJ_ROOT.

    combine_manifest.json used to hold absolute, machine-specific
    paths (PROJ_ROOT is wherever this checkout happens to sit), which
    breaks the moment the manifest is read on a different machine or
    from a copy of the repo elsewhere -- exactly the case for
    distributing assets/client_library_animated/ into
    therapist_trainer. Every project root here is the same shape
    (<proj-root>/src/<package>/...), so any consumer can resolve this
    back to an absolute path with its own PROJ_ROOT / Path(rel) join
    -- validate_combinations.py already does exactly that.

    :param path: Absolute path under PROJ_ROOT.
    :return: POSIX-style path string relative to PROJ_ROOT.
    """
    return path.resolve().relative_to(PROJ_ROOT).as_posix()


class PostureExpressionCombiner:
    """Produces the full posture x emotion matrix for the client library.

    :param gpu_index: Physical GPU index (0, 1, or 2 on quatro) to pin
        LivePortrait to.
    :param quality: Image quality tier passed to PostureVariantGenerator
        for posture edits ('low', 'medium', 'high', 'xhigh', 'max', or
        'auto').
    :param skip_posture: If True, never call the OpenAI API -- assumes
        posture variant files already exist next to each base image
        (e.g. generated earlier, or copied in from elsewhere) and
        raises FileNotFoundError if one is missing.
    :param force_posture: Regenerate posture variants even if the
        target file already exists. Ignored when skip_posture is True.
    """

    def __init__(self, gpu_index: int, quality: str = "medium",
                 skip_posture: bool = False, force_posture: bool = False):
        self.skip_posture = skip_posture
        self.force_posture = force_posture
        self.posture_gen = None if skip_posture else PostureVariantGenerator(quality=quality)
        # Constructed after posture_gen: LivePortraitService sets
        # CUDA_VISIBLE_DEVICES and must be the first thing in this
        # process to import torch/onnxruntime (see its own docstring).
        # PostureVariantGenerator only imports the openai package, so
        # constructing it first is safe.
        self.lp_service = LivePortraitService(gpu_index=gpu_index)
        self.driving_images = self._discover_driving_images()
        self.manifest: dict = {}
        # Keyed by "<bucket>/<base-image-stem>" -> list of
        # {stage, error_type, error} dicts. A single OpenAI moderation
        # rejection or LivePortrait hiccup on one image used to crash
        # the entire run (see chat: 32 minutes of work lost to one
        # moderation_blocked edit) -- these are now caught and recorded
        # here instead, so the run keeps going and you can review what
        # got skipped afterward via combine_failures.json (and re-find
        # the actual files with the gallery's filename search).
        self.failures: dict = {}

    #------------------------------------
    # _discover_driving_images
    #-------------------

    def _discover_driving_images(self) -> list:
        """Finds the emotion driving photos directly under assets/.

        Filters to recognized emotion names (KNOWN_EMOTIONS) rather
        than globbing every *.png under assets/ -- that directory also
        holds non-emotion photos, which would otherwise be swept in as
        a bogus extra "emotion".

        :return: Sorted list of Paths (e.g. assets/Anger.png, ...).
        """
        images = sorted(
            p for p in DRIVING_IMAGES_DIR.glob("*.png")
            if p.stem.lower() in KNOWN_EMOTIONS
        )
        if not images:
            raise FileNotFoundError(
                f"No recognized emotion driving images (one of {sorted(KNOWN_EMOTIONS)}) "
                f"found directly under {DRIVING_IMAGES_DIR}")
        log.info("Found %d driving emotion images: %s",
                  len(images), [p.stem for p in images])
        return images

    #------------------------------------
    # is_base_image
    #-------------------

    @staticmethod
    def is_base_image(path: Path) -> bool:
        """True if path is a bucket base photo, not a posture variant of one.

        :param path: Candidate image path.
        :return: True for '..._00.png'-style base images; False for
            '..._00_fists_clenched.png'-style variants.
        """
        return bool(BASE_IMAGE_STEM_PATTERN.search(path.stem))

    #------------------------------------
    # _record_failure
    #-------------------

    def _record_failure(self, base_image_path: Path, stage: str, error: Exception) -> None:
        """Records a per-image failure without aborting the run.

        :param base_image_path: The bucket base photo being processed
            when the failure happened.
        :param stage: Short label for what failed, e.g.
            'posture_edit:shoulders_raised' or 'animate:neutral:anger'.
        :param error: The exception that was caught.
        """
        key = f"{base_image_path.parent.name}/{base_image_path.stem}"
        self.failures.setdefault(key, []).append({
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        })

    #------------------------------------
    # _load_existing_state
    #-------------------

    def _load_existing_state(self) -> None:
        """Seeds self.manifest and self.failures from the files on disk.

        Makes every run *merge* into the existing combine_manifest.json
        and combine_failures.json instead of replacing them, so a
        --race or --max-buckets run can't erase the entries of bucket
        images it didn't process. A corrupt file raises on purpose:
        better to stop than to silently overwrite it.
        """
        manifest_path = ANIMATED_ROOT / "combine_manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
        failures_path = ANIMATED_ROOT / "combine_failures.json"
        if failures_path.exists():
            self.failures = json.loads(failures_path.read_text())

    #------------------------------------
    # _write_manifest
    #-------------------

    def _write_manifest(self) -> None:
        """Writes the manifest and failures list to disk.

        Called after every bucket image, not just once at the end of
        combine_all(), so a later failure -- another moderation
        rejection, a GPU hiccup, anything -- can't erase already
        -completed work the way it used to (see chat).
        """
        ANIMATED_ROOT.mkdir(parents=True, exist_ok=True)
        (ANIMATED_ROOT / "combine_manifest.json").write_text(
            json.dumps(self.manifest, indent=2))
        (ANIMATED_ROOT / "combine_failures.json").write_text(
            json.dumps(self.failures, indent=2))

    #------------------------------------
    # posture_sources_for
    #-------------------

    def posture_sources_for(self, base_image_path: Path) -> dict:
        """Resolves (or generates) the 3 posture-variant sources for one base photo.

        :param base_image_path: Path to a bucket base photo (e.g.
            assets/client_library/caucasian_female_20s-30s/caucasian_female_20s-30s_00.png).
        :return: Dict {posture_name: Path or None}, posture_name one of
            'neutral', 'shoulders_raised', 'fists_clenched'. 'neutral'
            maps to base_image_path itself -- no edit needed. A value
            of None means this posture's OpenAI edit failed (e.g. a
            safety-system rejection) -- skip it rather than treating
            it as a real source; see self.failures for why.
        :raises FileNotFoundError: if skip_posture is True and a
            variant file doesn't already exist.
        """
        sources = {"neutral": base_image_path}
        for posture in POSTURE_EDIT_PROMPTS:
            variant_path = base_image_path.with_name(
                f"{base_image_path.stem}_{posture}{base_image_path.suffix}")
            if self.skip_posture:
                if not variant_path.exists():
                    raise FileNotFoundError(
                        f"--skip-posture was passed but {variant_path} doesn't "
                        f"exist. Generate it first, or drop --skip-posture.")
            elif variant_path.exists() and not self.force_posture:
                log.info("Reusing existing posture variant %s", variant_path.name)
            else:
                try:
                    self.posture_gen.generate_variant(base_image_path, posture, variant_path)
                except OpenAIError as exc:
                    log.error(
                        "Posture edit failed for %s / %s -- skipping just this "
                        "posture for this image and continuing (%s: %s)",
                        base_image_path.name, posture, type(exc).__name__, exc)
                    self._record_failure(
                        base_image_path, stage=f"posture_edit:{posture}", error=exc)
                    sources[posture] = None
                    continue
            sources[posture] = variant_path
        return sources

    #------------------------------------
    # _add_neutral_entry
    #-------------------

    @staticmethod
    def _add_neutral_entry(source_path: Path, output_dir: Path) -> Path:
        """Copies a posture's own source image in as its "Neutral" emotion entry.

        Every posture source IS already its own neutral-face version
        before any LivePortrait animation touches it, so this is a
        file copy, not a GPU call. Idempotent (skips the copy if the
        target already exists), so it's safe to call on every run,
        including a resumed one.

        :param source_path: The posture source image (a bucket base
            photo for posture='neutral', or its shoulders_raised/
            fists_clenched edit).
        :param output_dir: That posture's output directory under
            ANIMATED_ROOT.
        :return: Path the copy lives at (existing or newly written).
        """
        target = output_dir / f"{source_path.stem}--{NEUTRAL_EMOTION_LABEL}{source_path.suffix}"
        if not target.exists():
            shutil.copyfile(source_path, target)
        return target

    #------------------------------------
    # combine_bucket_image
    #-------------------

    def combine_bucket_image(self, base_image_path: Path) -> dict:
        """Runs every posture x emotion combination for one base bucket photo.

        :param base_image_path: Path to a bucket base photo.
        :return: Dict {posture_name: {emotion: output_file_path_str}},
            where emotion includes NEUTRAL_EMOTION_LABEL alongside the
            driving-image emotions (see _add_neutral_entry()), and each
            path string is relative to PROJ_ROOT (see _to_manifest_path()).
        """
        bucket_key = base_image_path.parent.name
        sources = self.posture_sources_for(base_image_path)
        results: dict = {}

        for posture_name, source_path in sources.items():
            if source_path is None:
                log.warning(
                    "Skipping %s / %s entirely -- its posture edit failed "
                    "(see combine_failures.json)", bucket_key, posture_name)
                results[posture_name] = {"_skipped": "posture_edit_failed"}
                continue

            results[posture_name] = {}
            output_dir = ANIMATED_ROOT / bucket_key / base_image_path.stem / posture_name
            output_dir.mkdir(parents=True, exist_ok=True)

            neutral_path = self._add_neutral_entry(source_path, output_dir)
            results[posture_name][NEUTRAL_EMOTION_LABEL] = _to_manifest_path(neutral_path)

            for driving_path in self.driving_images:
                emotion = driving_path.stem
                multiplier = DRIVING_MULTIPLIERS.get(
                    emotion.lower(), FEAR_FALLBACK_MULTIPLIER)
                log.info("%s / %s / %s (multiplier=%.1f)",
                          bucket_key, posture_name, emotion, multiplier)
                try:
                    outcome = self.lp_service.generate(
                        source=source_path, driving=driving_path,
                        output_dir=output_dir, driving_multiplier=multiplier,
                    )
                    results[posture_name][emotion] = _to_manifest_path(outcome["wfp"])
                except Exception as exc:
                    log.error(
                        "Animation failed for %s / %s / %s -- skipping just "
                        "this one and continuing (%s: %s)",
                        bucket_key, posture_name, emotion, type(exc).__name__, exc)
                    self._record_failure(
                        base_image_path, stage=f"animate:{posture_name}:{emotion}",
                        error=exc)
                    results[posture_name][emotion] = None

        return results

    #------------------------------------
    # combine_all
    #-------------------

    def combine_all(self, max_buckets: int = None, races: list = None) -> None:
        """Runs combine_bucket_image() over every base photo in the library.

        Merges into the existing combine_manifest.json /
        combine_failures.json (see _load_existing_state()).

        :param max_buckets: If set, only process the first N distinct
            bucket subdirectories (a cheap smoke test), not N images.
            Applied after the races filter.
        :param races: If set, only process bucket images whose bucket
            directory belongs to one of these Race members.
        """
        self._load_existing_state()
        base_images = sorted(p for p in LIBRARY_ROOT.glob("*/*.png") if self.is_base_image(p))
        if races:
            # Bucket keys are '<race>_<sex>_<age>' and race values can
            # themselves contain underscores, so match on the prefix.
            prefixes = tuple(f"{r.value}_" for r in races)
            base_images = [p for p in base_images if p.parent.name.startswith(prefixes)]
            log.info("Restricting run to race(s): %s", ", ".join(r.value for r in races))
        if not base_images:
            raise FileNotFoundError(
                f"No base images found under {LIBRARY_ROOT}"
                f"{' for races ' + ', '.join(r.value for r in races) if races else ''}"
                f" -- run client_library_generator.py first.")

        if max_buckets is not None:
            bucket_keys_seen = []
            limited = []
            for p in base_images:
                key = p.parent.name
                if key not in bucket_keys_seen:
                    if len(bucket_keys_seen) >= max_buckets:
                        break
                    bucket_keys_seen.append(key)
                limited.append(p)
            base_images = limited
            log.info("Limiting run to first %d bucket(s) (smoke test)", max_buckets)

        for base_image_path in base_images:
            key = f"{base_image_path.parent.name}/{base_image_path.stem}"
            # Forget failures recorded for this image by an earlier run:
            # it's being redone now, and new failures are re-recorded.
            self.failures.pop(key, None)
            try:
                self.manifest[key] = self.combine_bucket_image(base_image_path)
            except Exception as exc:
                # Last-resort net: anything that slips past the
                # per-posture/per-animation handling above (a bug, an
                # unanticipated error class) skips this one bucket
                # image rather than taking down the whole run.
                log.error(
                    "Unexpected failure on %s -- skipping this bucket image "
                    "entirely and continuing (%s: %s)",
                    key, type(exc).__name__, exc)
                self._record_failure(base_image_path, stage="bucket_image", error=exc)
                self.manifest[key] = {"_skipped": "unexpected_error"}
            # Written after every image, not just at the end, so a
            # later failure can't erase progress already on disk.
            self._write_manifest()

        log.info("Wrote manifest: %s", ANIMATED_ROOT / "combine_manifest.json")
        if self.failures:
            log.warning(
                "%d bucket image(s) had at least one failure -- see %s",
                len(self.failures), ANIMATED_ROOT / "combine_failures.json")
        log.info("New images below: %s", ANIMATED_ROOT)


class PostureExpressionCombinerCLI:
    """Parses CLI arguments and runs PostureExpressionCombiner.

    :param argv: Argument list to parse (defaults to sys.argv).
    """

    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--gpu", type=int, required=True, choices=[0, 1, 2])
        parser.add_argument("--max-buckets", type=int, default=None,
                             help="Limit to the first N bucket subdirectories, "
                                  "for a cheap smoke test before the full run.")
        parser.add_argument("--race", action="append", dest="races",
                             choices=[r.value for r in Race], default=None,
                             help="Only process this race (repeatable). "
                                  "Default: all races.")
        parser.add_argument("--skip-posture", action="store_true",
                             help="Never call the OpenAI API; require posture "
                                  "variant files to already exist next to each "
                                  "base image.")
        parser.add_argument("--force-posture", action="store_true",
                             help="Regenerate posture variants even if already "
                                  "present. Ignored with --skip-posture.")
        parser.add_argument("--quality", choices=["low", "medium", "high", "xhigh", "max", "auto"],
                             default="medium",
                             help="Image quality tier for posture edits.")
        self.args = parser.parse_args(argv)

    def run(self) -> None:
        """Builds and runs the combiner with the parsed arguments."""
        combiner = PostureExpressionCombiner(
            gpu_index=self.args.gpu, quality=self.args.quality,
            skip_posture=self.args.skip_posture, force_posture=self.args.force_posture,
        )
        combiner.combine_all(
            max_buckets=self.args.max_buckets,
            races=[Race.from_value(v) for v in self.args.races]
                   if self.args.races else None,
        )


if __name__ == "__main__":
    PostureExpressionCombinerCLI().run()
