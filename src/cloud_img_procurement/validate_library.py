"""
Validates every image in the generated client-photo library by
confirming LivePortrait's face cropper detects a face in it --
catching generation failures (no face, bad crop, extreme angle)
offline, before any of these images could reach a student mid-session.

Lives at <proj-root>/src/cloud_img_procurement/validate_library.py.

Usage:
    conda run -n image-gen-test python src/cloud_img_procurement/validate_library.py --gpu 0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT / "src"))

from image_gen.live_portrait_service import LivePortraitService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cloud_img_procurement")

LIBRARY_ROOT = PROJ_ROOT / "assets" / "client_library"


class ClientLibraryValidator:
    """Runs every library image through LivePortrait's face cropper.

    :param gpu_index: Which GPU to load LivePortrait's models onto for
        this validation pass.
    """

    def __init__(self, gpu_index: int):
        self.service = LivePortraitService(gpu_index=gpu_index)

    def validate_all(self) -> dict:
        """Validates every .png under each bucket subdirectory.

        :return: Dict with 'passed' and 'failed' lists of paths,
            relative to PROJ_ROOT.
        """
        results = {"passed": [], "failed": []}
        image_paths = sorted(LIBRARY_ROOT.glob("*/*.png"))
        log.info("Validating %d images...", len(image_paths))

        for path in image_paths:
            rel_path = str(path.relative_to(PROJ_ROOT))
            if self.service.validate_source_photo(path):
                results["passed"].append(rel_path)
            else:
                log.warning("FAILED face detection: %s", rel_path)
                results["failed"].append(rel_path)

        return results


class ClientLibraryValidatorCLI:
    """Parses CLI arguments and runs ClientLibraryValidator.

    :param argv: Argument list to parse (defaults to sys.argv).
    """

    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--gpu", type=int, required=True, choices=[0, 1, 2])
        parser.add_argument("--report", type=Path,
                             default=LIBRARY_ROOT / "validation_report.json")
        self.args = parser.parse_args(argv)

    def run(self) -> None:
        """Runs validation and writes the JSON report."""
        validator = ClientLibraryValidator(gpu_index=self.args.gpu)
        results = validator.validate_all()

        self.args.report.write_text(json.dumps(results, indent=2))
        log.info("Passed: %d  Failed: %d", len(results["passed"]), len(results["failed"]))
        log.info("Report written to %s", self.args.report)


if __name__ == "__main__":
    ClientLibraryValidatorCLI().run()
