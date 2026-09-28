#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-09-28 09:22:15
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/cloud_img_procurement/validate_combinations.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-09-28 09:22:15
# **********************************************************

"""
Validates the posture x emotion matrix produced by
posture_expression_combiner.py, the same way validate_library.py
validates the base client photo library: by confirming LivePortrait's
face cropper actually detects a face in every animated output --
catching a corrupt or degenerate animation frame offline, before it
could reach a student mid-session.

Driven from combine_manifest.json rather than a directory glob, so it
knows exactly what the combiner intended to produce: entries the
combiner itself already marked as skipped (a moderation-blocked
posture edit, a failed animation -- see posture_expression_combiner.py's
per-image/per-combo failure handling) are reported separately, not
re-flagged as validation failures.

Lives at <proj-root>/src/cloud_img_procurement/validate_combinations.py,
alongside the other batch-asset-production scripts, for the same
reason validate_library.py does (needs a GPU and image_gen.LivePortraitService).
Requires the editable install from setup_env.sh (`pip install -e .`).

Usage:
    conda run -n therapist-img-gen python src/cloud_img_procurement/validate_combinations.py --gpu 0
"""

import argparse
import json
import logging
from pathlib import Path

from image_gen.live_portrait_service import LivePortraitService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cloud_img_procurement")

PROJ_ROOT = Path(__file__).resolve().parents[2]
ANIMATED_ROOT = PROJ_ROOT / "assets" / "client_library_animated"


class CombinationValidator:
    """Runs every combiner output through LivePortrait's face cropper.

    :param gpu_index: Which GPU to load LivePortrait's models onto for
        this validation pass.
    :param manifest_path: Path to posture_expression_combiner.py's
        combine_manifest.json.
    """

    def __init__(self, gpu_index: int, manifest_path: Path):
        self.service = LivePortraitService(gpu_index=gpu_index)
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"{self.manifest_path} not found -- run "
                f"posture_expression_combiner.py first.")
        self.manifest = json.loads(self.manifest_path.read_text())

    #------------------------------------
    # _iter_outputs
    #-------------------

    def _iter_outputs(self):
        """Walks the manifest, yielding (label, path_str_or_None) pairs.

        A posture entry that's a dict of {'_skipped': reason} (the
        combiner's marker for a posture whose edit failed -- see
        posture_expression_combiner.py) yields one skip with path=None
        for the whole posture, not one per emotion, since no per
        -emotion output was ever attempted.

        :return: Generator of (label, path_str_or_None) tuples, where
            label is '<bucket-key> / <posture> [/ <emotion>]'.
        """
        for bucket_key, postures in self.manifest.items():
            if postures.get("_skipped"):
                yield f"{bucket_key} (bucket image)", None
                continue
            for posture, emotions in postures.items():
                if isinstance(emotions, dict) and emotions.get("_skipped"):
                    yield f"{bucket_key} / {posture}", None
                    continue
                for emotion, path_str in emotions.items():
                    yield f"{bucket_key} / {posture} / {emotion}", path_str

    #------------------------------------
    # validate_all
    #-------------------

    def validate_all(self) -> dict:
        """Validates every non-skipped output the manifest references.

        :return: Dict with 'passed', 'failed', and 'skipped' lists.
            'passed'/'failed' entries are paths (relative to
            PROJ_ROOT); 'skipped' entries are the descriptive labels
            from _iter_outputs() for combos the combiner itself never
            produced.
        """
        results = {"passed": [], "failed": [], "skipped": []}

        for label, path_str in self._iter_outputs():
            if path_str is None:
                results["skipped"].append(label)
                continue

            path = Path(path_str)
            if not path.is_absolute():
                path = PROJ_ROOT / path
            rel_path = str(path.relative_to(PROJ_ROOT)) if path.exists() else str(path)

            if not path.exists():
                log.warning("MISSING FILE (manifest says it exists): %s", rel_path)
                results["failed"].append(rel_path)
            elif self.service.validate_source_photo(path):
                results["passed"].append(rel_path)
            else:
                log.warning("FAILED face detection: %s", rel_path)
                results["failed"].append(rel_path)

        return results


class CombinationValidatorCLI:
    """Parses CLI arguments and runs CombinationValidator.

    :param argv: Argument list to parse (defaults to sys.argv).
    """

    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--gpu", type=int, required=True, choices=[0, 1, 2])
        parser.add_argument("--manifest", type=Path,
                             default=ANIMATED_ROOT / "combine_manifest.json")
        parser.add_argument("--report", type=Path,
                             default=ANIMATED_ROOT / "combine_validation_report.json")
        self.args = parser.parse_args(argv)

    def run(self) -> None:
        """Runs validation and writes the JSON report."""
        validator = CombinationValidator(
            gpu_index=self.args.gpu, manifest_path=self.args.manifest)
        results = validator.validate_all()

        self.args.report.write_text(json.dumps(results, indent=2))
        log.info("Passed: %d  Failed: %d  Skipped (never generated): %d",
                  len(results["passed"]), len(results["failed"]), len(results["skipped"]))
        log.info("Report written to %s", self.args.report)


if __name__ == "__main__":
    CombinationValidatorCLI().run()
