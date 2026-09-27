#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-09-24 18:12:40
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/cloud_img_procurement/posture_variant_generator.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-09-27 13:51:24
# **********************************************************

"""
Generates posture-variant images (e.g. shoulders raised, fists clenched)
from an already-generated "neutral" client-library base photo, via the
OpenAI Images *edit* endpoint (client.images.edit) rather than a fresh
text-to-image call.

Why edit instead of generate: images.edit takes an existing image plus
an instruction and changes only what's asked, and GPT Image 2.5
preserves identity, composition, and lighting across such edits far
more reliably than asking images.generate to reproduce the same face
twice from a text prompt alone (see chat). Every posture variant is
therefore branched from the *same* neutral base image -- never
edit-of-edit -- so small deviations don't compound across passes.

Lives at <proj-root>/src/cloud_img_procurement/posture_variant_generator.py.
Requires the editable install from setup_env.sh (`pip install -e .`), so
common/cloud_img_procurement are importable by package name.

Usage:
    python src/cloud_img_procurement/posture_variant_generator.py \\
        --base assets/client_library/caucasian_female_20s-30s/caucasian_female_20s-30s_00.png \\
        --posture fists_clenched
"""

import argparse
import base64
import logging
from pathlib import Path

from openai import OpenAI

from common.api_credentials import OpenAICredentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cloud_img_procurement")

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Model must support the images.edit endpoint. gpt-image-2.5-flare and
# gpt-image-2.5-sunburst both do; Sunburst trades speed for tighter
# identity/detail preservation if Flare's edits drift too much (see chat).
MODEL = "gpt-image-2.5-flare"

# Keep-list stated first, single change stated second -- the prompting
# order OpenAI recommends for GPT Image 2.5 edits, so the model treats
# everything else as fixed rather than reinterpreting the whole scene.
KEEP_LIST = (
    "Keep the person's face, identity, skin tone, hairstyle, clothing, "
    "the office background, lighting, camera angle, and framing exactly "
    "unchanged."
)

POSTURE_EDIT_PROMPTS = {
    "shoulders_raised": (
        f"{KEEP_LIST} Raise and tense the shoulders, pulling them up "
        f"toward the ears, as if the client is physically bracing under "
        f"stress."
    ),
    "fists_clenched": (
        f"{KEEP_LIST} Clench both hands into tight fists, tensing the "
        f"fingers and knuckles."
    ),
}


class PostureVariantGenerator:
    """Produces a posture-edit variant of a single base photo.

    :param model: OpenAI image model to call. Must support the
        images.edit endpoint (gpt-image-2.5-flare or
        gpt-image-2.5-sunburst).
    :param quality: Edit quality tier ('low', 'medium', 'high', 'xhigh',
        'max', or 'auto').
    """

    def __init__(self, model: str = MODEL, quality: str = "medium"):
        self.model = model
        self.quality = quality
        creds = OpenAICredentials()
        self.client = OpenAI(api_key=creds.api_key, organization=creds.organization)

    #------------------------------------
    # generate_variant
    #-------------------

    def generate_variant(self, base_image_path: Path, posture: str,
                          out_path: Path = None) -> Path:
        """Edits one base photo into a named posture variant.

        :param base_image_path: Path to the already-generated neutral
            photo to branch this variant from. Always pass the
            *original* neutral image here, never a previously
            generated variant -- editing an edit compounds drift
            instead of holding identity fixed.
        :type base_image_path: Path
        :param posture: Key into POSTURE_EDIT_PROMPTS (currently
            'shoulders_raised' or 'fists_clenched').
        :type posture: str
        :param out_path: Where to save the result. Defaults to
            '<base_image_path stem>_<posture>.png' next to the base
            image.
        :type out_path: Path
        :return: Path the variant was written to.
        :rtype: Path
        :raises KeyError: if posture is not a recognized key.
        :raises FileNotFoundError: if base_image_path does not exist.
        """
        if posture not in POSTURE_EDIT_PROMPTS:
            raise KeyError(
                f"Unknown posture '{posture}'; choose one of "
                f"{list(POSTURE_EDIT_PROMPTS.keys())}")

        base_image_path = Path(base_image_path)
        if not base_image_path.exists():
            raise FileNotFoundError(f"Base image not found: {base_image_path}")

        if out_path is None:
            out_path = base_image_path.with_name(
                f"{base_image_path.stem}_{posture}{base_image_path.suffix}")

        prompt = POSTURE_EDIT_PROMPTS[posture]
        log.info("Editing %s -> %s (%s)", base_image_path.name, out_path.name, posture)

        with open(base_image_path, "rb") as base_file:
            response = self.client.images.edit(
                model=self.model,
                image=base_file,
                prompt=prompt,
                quality=self.quality,
                n=1,
            )

        image_bytes = base64.b64decode(response.data[0].b64_json)
        out_path.write_bytes(image_bytes)
        log.info("Wrote %s", out_path)
        return out_path

    #------------------------------------
    # generate_all_variants
    #-------------------

    def generate_all_variants(self, base_image_path: Path) -> dict:
        """Convenience: generates every posture in POSTURE_EDIT_PROMPTS
        from one base photo.

        :param base_image_path: Path to the neutral base photo.
        :type base_image_path: Path
        :return: Dict mapping posture name -> output Path.
        :rtype: dict
        """
        return {
            posture: self.generate_variant(base_image_path, posture)
            for posture in POSTURE_EDIT_PROMPTS
        }


class PostureVariantGeneratorCLI:
    """Parses CLI arguments and runs PostureVariantGenerator.

    :param argv: Argument list to parse (defaults to sys.argv).
    """

    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--base", type=Path, required=True,
                             help="Path to the neutral base photo to edit.")
        parser.add_argument("--posture",
                             choices=list(POSTURE_EDIT_PROMPTS.keys()) + ["all"],
                             required=True,
                             help="Which posture to generate, or 'all'.")
        parser.add_argument("--out", type=Path, default=None,
                             help="Output path (default: alongside --base). "
                                  "Ignored when --posture all is used.")
        parser.add_argument("--quality",
                             choices=["low", "medium", "high", "xhigh", "max", "auto"],
                             default="medium")
        self.args = parser.parse_args(argv)

    def run(self) -> None:
        """Runs the requested edit(s) and reports the output path(s)."""
        generator = PostureVariantGenerator(quality=self.args.quality)
        if self.args.posture == "all":
            out_paths = generator.generate_all_variants(self.args.base)
            for posture, path in out_paths.items():
                log.info("%s -> %s", posture, path)
            log.info("New images below: %s", self.args.base.resolve().parent)
        else:
            out_path = generator.generate_variant(
                self.args.base, self.args.posture, self.args.out)
            log.info("Done: %s", out_path)
            log.info("New images below: %s", out_path.resolve().parent)


if __name__ == "__main__":
    PostureVariantGeneratorCLI().run()
