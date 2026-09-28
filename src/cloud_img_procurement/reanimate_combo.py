#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-09-28 09:52:25
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/cloud_img_procurement/reanimate_combo.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-09-28 09:52:25
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
"""

import argparse
import logging
from pathlib import Path

from image_gen.live_portrait_service import LivePortraitService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cloud_img_procurement")

PROJ_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ROOT = PROJ_ROOT / "assets" / "client_library"
DRIVING_IMAGES_DIR = PROJ_ROOT / "assets"
ANIMATED_ROOT = PROJ_ROOT / "assets" / "client_library_animated"

POSTURES = ["neutral", "shoulders_raised", "fists_clenched"]


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

    :param argv: Argument list to parse (defaults to sys.argv).
    """

    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--bucket", required=True,
                             help="Bucket directory name, e.g. asian_female_20s-30s.")
        parser.add_argument("--image", required=True,
                             help="Bucket image index as it appears in the "
                                  "filename, e.g. 00.")
        parser.add_argument("--posture", required=True,
                             choices=POSTURES + ["all"])
        parser.add_argument("--emotion", required=True,
                             help="Driving-image stem, e.g. Contempt.")
        parser.add_argument("--multiplier", type=float, required=True,
                             help="driving_multiplier to use for this run.")
        parser.add_argument("--gpu", type=int, required=True, choices=[0, 1, 2])
        self.args = parser.parse_args(argv)

    def run(self) -> None:
        """Reanimates the requested combo(s)."""
        reanimator = ComboReanimator(gpu_index=self.args.gpu)
        postures = POSTURES if self.args.posture == "all" else [self.args.posture]
        for posture in postures:
            reanimator.reanimate(
                self.args.bucket, self.args.image, posture,
                self.args.emotion, self.args.multiplier)
        log.info("New images below: %s",
                  ANIMATED_ROOT / self.args.bucket / f"{self.args.bucket}_{self.args.image}")


if __name__ == "__main__":
    ComboReanimatorCLI().run()
