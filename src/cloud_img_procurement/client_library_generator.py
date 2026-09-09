 # **********************************************************
 #
 # @Author: Andreas Paepcke
 # @Date:   2026-09-08 19:26:19
 # @File:   /Users/paepcke/VSCodeWorkspaces/image-gen-test/src/cloud_img_procurement/client_library_generator.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2026-09-08 19:31:47
 #
 # **********************************************************
"""
Batch generator for the pre-generated client-photo library, via GPT
Image 2. One-time offline asset creation -- not a runtime dependency
(see chat: this converts an unbounded per-session image-gen risk into
a bounded, reviewable, one-time task).

Lives at <proj-root>/src/cloud_img_procurement/client_library_generator.py.
5 races x 2 sexes x 3 age ranges = 30 buckets.

Reads the OpenAI API key (and org id) from $HOME/.ssh/openai_api_key.txt
via common.api_credentials.OpenAICredentials -- no key is read from an
environment variable or placed in code.

Usage:
    python src/cloud_img_procurement/client_library_generator.py \\
        --images-per-bucket 4
"""

import argparse
import base64
import itertools
import json
import logging
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT / "src"))

from openai import OpenAI

from common.api_credentials import OpenAICredentials
from cloud_img_procurement.bucket_enums import AgeRange, Race, Sex
from cloud_img_procurement.client_bucket import ClientBucket, OfficePromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cloud_img_procurement")

LIBRARY_ROOT = PROJ_ROOT / "assets" / "client_library"


class ClientLibraryGenerator:
    """Generates and saves the bucketed client-photo library via GPT Image 2.

    Resumable: an existing image file for a given bucket+index is
    skipped rather than regenerated unless force=True, so a failed or
    partial run can just be re-invoked with the same arguments.

    :param images_per_bucket: How many distinct photos per bucket.
    :param force: Regenerate images even if the target file exists.
    :param max_buckets: If set, only process the first N buckets --
        for a cheap smoke test of the whole pipeline (prompt, API
        call, file save, resumability) before committing to a full
        30-bucket run.
    """

    MODEL = "gpt-image-2"

    def __init__(self, images_per_bucket: int = 4, force: bool = False,
                 max_buckets: int = None):
        self.images_per_bucket = images_per_bucket
        self.force = force
        self.max_buckets = max_buckets
        creds = OpenAICredentials()
        self.client = OpenAI(api_key=creds.api_key, organization=creds.organization)
        self.prompt_template = OfficePromptTemplate()
        self.manifest: dict = {}

    def all_buckets(self) -> list:
        """Enumerates every (race, sex, age_range) combination.

        :return: List of ClientBucket instances.
        """
        return [
            ClientBucket(race=r, sex=s, age_range=a)
            for r, s, a in itertools.product(Race, Sex, AgeRange)
        ]

    def generate_bucket(self, bucket: ClientBucket) -> list:
        """Generates (or reuses) all images for one bucket.

        :param bucket: The demographic bucket to generate photos for.
        :return: List of saved file paths, relative to PROJ_ROOT.
        """
        bucket_dir = LIBRARY_ROOT / bucket.key
        bucket_dir.mkdir(parents=True, exist_ok=True)
        prompt = self.prompt_template.build(bucket)
        saved = []

        for i in range(self.images_per_bucket):
            out_path = bucket_dir / bucket.image_filename(i)
            if out_path.exists() and not self.force:
                log.info("Skipping existing %s", out_path.name)
                saved.append(str(out_path.relative_to(PROJ_ROOT)))
                continue

            log.info("Generating %s [%d/%d]", bucket.key, i + 1, self.images_per_bucket)
            response = self.client.images.generate(
                model=self.MODEL, prompt=prompt, size="1024x1024", n=1,
            )
            image_bytes = base64.b64decode(response.data[0].b64_json)
            out_path.write_bytes(image_bytes)
            saved.append(str(out_path.relative_to(PROJ_ROOT)))

        return saved

    def run(self) -> None:
        """Generates the (possibly bucket-limited) library and writes manifest.json."""
        buckets = self.all_buckets()
        if self.max_buckets is not None:
            buckets = buckets[:self.max_buckets]
            log.info("Limiting run to first %d of %d buckets (smoke test)",
                      self.max_buckets, len(self.all_buckets()))

        for bucket in buckets:
            self.manifest[bucket.key] = self.generate_bucket(bucket)

        manifest_path = LIBRARY_ROOT / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, indent=2))
        log.info("Wrote manifest: %s", manifest_path)


class ClientLibraryGeneratorCLI:
    """Parses CLI arguments and runs ClientLibraryGenerator.

    :param argv: Argument list to parse (defaults to sys.argv).
    """

    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--images-per-bucket", type=int, default=4)
        parser.add_argument("--force", action="store_true",
                             help="Regenerate images even if already present.")
        parser.add_argument("--max-buckets", type=int, default=None,
                             help="Limit to the first N buckets, for a cheap "
                                  "smoke test (e.g. --max-buckets 2 "
                                  "--images-per-bucket 1) before the full run.")
        self.args = parser.parse_args(argv)

    def run(self) -> None:
        """Builds and runs the generator with the parsed arguments."""
        generator = ClientLibraryGenerator(
            images_per_bucket=self.args.images_per_bucket, force=self.args.force,
            max_buckets=self.args.max_buckets,
        )
        generator.run()


if __name__ == "__main__":
    ClientLibraryGeneratorCLI().run()
