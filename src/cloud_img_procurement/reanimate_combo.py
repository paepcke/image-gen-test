#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-09-28 09:52:25
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/cloud_img_procurement/reanimate_combo.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-09-28 10:14:46
# **********************************************************

"""
Regenerates one or more specific posture x emotion combos from
posture_expression_combiner.py's output -- e.g. to retry a combo with
a gentler driving_multiplier after spotting it looks overdone in the
gallery -- without re-running the full (expensive) combine_all()
matrix.

Resolves the source/driving images the same way
posture_expression_combiner.py does (LIBRARY_ROOT for posture
sources, DRIVING_IMAGES_DIR for emotion driving images), so the
regenerated output lands at the exact same path LivePortrait wrote it
to the first time -- this overwrites in place; combine_manifest.json
doesn't need updating.

Lives at <proj-root>/src/cloud_img_procurement/reanimate_combo.py.
Requires the editable install from setup_env.sh (`pip install -e .`).

Usage:
    # Redo just Contempt for one bucket image, across all 3 postures,
    # with a gentler multiplier than the calibrated default (2.0):
    conda run -n therapist-img-gen python src/cloud_img_procurement/reanimate_combo.py \\
        --bucket asian_female_20s-30s --image 00 --posture all \\
        --emotion Contempt --multiplier 1.3 --gpu 0

    # Or: mark the offending images with the gallery's "redo" checkboxes,
    # "Copy marked filenames", paste the clipboard into a file (or pipe
    # it straight in), and redo all of them at once, same multiplier
    # for every entry:
    pbpaste | conda run -n therapist-img-gen python src/cloud_img_procurement/reanimate_combo.py \\
        --worklist - --multiplier 1.3 --gpu 0
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from image_gen.live_portrait_service import LivePortraitService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cloud_img_procurement")

PROJ_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ROOT = PROJ_ROOT / "assets" / "client_library"
DRIVING_IMAGES_DIR = PROJ_ROOT / "assets"
ANIMATED_ROOT = PROJ_ROOT / "assets" / "client_library_animated"

POSTURES = ["neutral", "shoulders_raised", "fists_clenched"]
EDITED_POSTURES = "|".join(p for p in POSTURES if p != "neutral")

# Matches the bare filenames posture_expression_combiner.py /
# reanimate_combo.py itself write, e.g.
#   asian_female_20s-30s_00--Contempt.jpg                      (neutral)
#   asian_female_20s-30s_00_shoulders_raised--Contempt.jpg
#   asian_female_20s-30s_01_fists_clenched--Contempt.jpg
# -- i.e. what the gallery's "Copy marked filenames" button hands you.
# The bucket name itself may contain underscores, so the 2-digit image
# index anchors the split: greedy '.+' on the left backs off only as
# far as the last '_NN' it can find.
FILENAME_PATTERN = re.compile(
    rf"^(?P<bucket>.+)_(?P<image>\d{{2}})(?:_(?P<posture>{EDITED_POSTURES}))?"
    rf"--(?P<emotion>[^.]+)\.\w+$"
)


def parse_combo_filename(filename: str) -> tuple:
    """Parses one bare output filename into its (bucket, image, posture, emotion).

    :param filename: A bare filename as the gallery shows it (no
        path), e.g. 'asian_female_20s-30s_01_fists_clenched--Contempt.jpg'.
    :return: (bucket, image_num, posture, emotion) tuple. posture is
        'neutral' when the filename has no posture suffix.
    :raises ValueError: if filename doesn't match the expected
        '<bucket>_<NN>[_<posture>]--<emotion>.<ext>' shape.
    """
    match = FILENAME_PATTERN.match(filename.strip())
    if not match:
        raise ValueError(f"Doesn't look like a combiner output filename: {filename!r}")
    posture = match.group("posture") or "neutral"
    return match.group("bucket"), match.group("image"), posture, match.group("emotion")


def _source_path_for(bucket: str, image_num: str, posture: str) -> Path:
    """Resolves the posture-source image path the combiner would have used.

    :param bucket: Bucket directory name (e.g. 'asian_female_20s-30s').
    :param image_num: Bucket image index as it appears in the
        filename (e.g. '00').
    :param posture: 'neutral', 'shoulders_raised', or 'fists_clenched'.
    :return: Path to the base photo (posture == 'neutral') or the
        posture-edited variant next to it.
    """
    base = LIBRARY_ROOT / bucket / f"{bucket}_{image_num}.png"
    if posture == "neutral":
        return base
    return base.with_name(f"{base.stem}_{posture}{base.suffix}")


class ComboReanimator:
    """Regenerates specific posture x emotion combos in place.

    :param gpu_index: Physical GPU index to pin LivePortrait to.
    """

    def __init__(self, gpu_index: int):
        self.lp_service = LivePortraitService(gpu_index=gpu_index)

    #------------------------------------
    # reanimate
    #-------------------

    def reanimate(self, bucket: str, image_num: str, posture: str,
                  emotion: str, multiplier: float) -> Path:
        """Re-runs one posture x emotion combo, overwriting the existing output.

        :param bucket: Bucket directory name.
        :param image_num: Bucket image index (e.g. '00').
        :param posture: 'neutral', 'shoulders_raised', or 'fists_clenched'.
        :param emotion: Driving-image stem (case-insensitive match
            against DRIVING_IMAGES_DIR/*.png), e.g. 'Contempt'.
        :param multiplier: driving_multiplier to use for this call --
            pass the calibrated default explicitly if you're
            regenerating for some other reason (e.g. a corrupted
            frame), not to change the intensity.
        :return: Path to the regenerated output file.
        :raises FileNotFoundError: if the source or driving image
            doesn't exist.
        """
        source_path = _source_path_for(bucket, image_num, posture)
        if not source_path.exists():
            raise FileNotFoundError(f"Posture source not found: {source_path}")

        driving_candidates = [
            p for p in DRIVING_IMAGES_DIR.glob("*.png")
            if p.stem.lower() == emotion.lower()
        ]
        if not driving_candidates:
            raise FileNotFoundError(
                f"No driving image found for emotion '{emotion}' under {DRIVING_IMAGES_DIR}")
        driving_path = driving_candidates[0]

        output_dir = ANIMATED_ROOT / bucket / f"{bucket}_{image_num}" / posture
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info("Reanimating %s / %s / %s (multiplier=%.2f)",
                  f"{bucket}_{image_num}", posture, driving_path.stem, multiplier)
        outcome = self.lp_service.generate(
            source=source_path, driving=driving_path,
            output_dir=output_dir, driving_multiplier=multiplier,
        )
        log.info("Wrote %s", outcome["wfp"])
        return outcome["wfp"]


class ComboReanimatorCLI:
    """Parses CLI arguments and runs ComboReanimator.

    Two mutually exclusive modes: name one combo (or --posture all for
    one image) via --bucket/--image/--posture/--emotion, or point
    --worklist at a list of bare filenames (one per line, e.g. the
    gallery's "Copy marked filenames" clipboard output) to redo many
    combos in one call, all at the same --multiplier.

    :param argv: Argument list to parse (defaults to sys.argv).
    """

    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--bucket",
                             help="Bucket directory name, e.g. asian_female_20s-30s. "
                                  "Required unless --worklist is given.")
        parser.add_argument("--image",
                             help="Bucket image index as it appears in the "
                                  "filename, e.g. 00. Required unless --worklist "
                                  "is given.")
        parser.add_argument("--posture", choices=POSTURES + ["all"],
                             help="Required unless --worklist is given.")
        parser.add_argument("--emotion",
                             help="Driving-image stem, e.g. Contempt. Required "
                                  "unless --worklist is given.")
        parser.add_argument("--worklist", type=Path,
                             help="Path to a file of bare output filenames, one "
                                  "per line (blank lines and '#' comments "
                                  "ignored) -- e.g. paste the gallery's 'Copy "
                                  "marked filenames' clipboard output into a "
                                  "file. Pass '-' to read from stdin. Redoes "
                                  "every parsed combo at the same --multiplier. "
                                  "Mutually exclusive with "
                                  "--bucket/--image/--posture/--emotion.")
        parser.add_argument("--multiplier", type=float, required=True,
                             help="driving_multiplier to use for this run.")
        parser.add_argument("--gpu", type=int, required=True, choices=[0, 1, 2])
        self.args = parser.parse_args(argv)

        explicit_given = any(
            v is not None for v in
            (self.args.bucket, self.args.image, self.args.posture, self.args.emotion))
        if self.args.worklist and explicit_given:
            parser.error("--worklist is mutually exclusive with "
                          "--bucket/--image/--posture/--emotion.")
        if not self.args.worklist and not all(
                (self.args.bucket, self.args.image, self.args.posture, self.args.emotion)):
            parser.error("Either --worklist, or all of "
                          "--bucket/--image/--posture/--emotion, is required.")

    #------------------------------------
    # _combos_from_worklist
    #-------------------

    def _combos_from_worklist(self) -> list:
        """Reads --worklist and parses each line into a (bucket, image, posture, emotion) tuple.

        A malformed line is logged as a warning and skipped rather
        than aborting the whole worklist -- same resilience philosophy
        as posture_expression_combiner.py's per-combo failure handling.
        Each line keeps its own emotion (parsed from that filename),
        so a worklist can mix combos from different emotions in one
        call -- only --multiplier is shared across all of them.

        :return: List of (bucket, image_num, posture, emotion)
            tuples, in first-seen order, de-duplicated.
        """
        if str(self.args.worklist) == "-":
            lines = sys.stdin.read().splitlines()
        else:
            # .expanduser(): argparse's type=Path does NOT expand a
            # leading '~' on its own -- without this, --worklist
            # ~/tmp/badImages.txt would look for a literal './~/tmp/...'
            # relative to cwd and fail with FileNotFoundError.
            lines = self.args.worklist.expanduser().read_text().splitlines()

        seen = set()
        combos = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                combo = parse_combo_filename(line)
            except ValueError as exc:
                log.warning("Skipping worklist line -- %s", exc)
                continue
            if combo not in seen:
                seen.add(combo)
                combos.append(combo)
        return combos

    def run(self) -> None:
        """Reanimates the requested combo(s)."""
        reanimator = ComboReanimator(gpu_index=self.args.gpu)

        if self.args.worklist:
            combos = self._combos_from_worklist()
            log.info("Worklist: %d distinct combo(s) to reanimate at multiplier=%.2f",
                      len(combos), self.args.multiplier)
            for bucket, image_num, posture, emotion in combos:
                reanimator.reanimate(bucket, image_num, posture,
                                      emotion, self.args.multiplier)
            return

        postures = POSTURES if self.args.posture == "all" else [self.args.posture]
        for posture in postures:
            reanimator.reanimate(
                self.args.bucket, self.args.image, posture,
                self.args.emotion, self.args.multiplier)
        log.info("New images below: %s",
                  ANIMATED_ROOT / self.args.bucket / f"{self.args.bucket}_{self.args.image}")


if __name__ == "__main__":
    ComboReanimatorCLI().run()
