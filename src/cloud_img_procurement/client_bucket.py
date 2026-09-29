 # **********************************************************
 # @Author: Andreas Paepcke
 # @Date:   2026-09-08 19:13:19
 # @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/cloud_img_procurement/client_bucket.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2026-09-28 17:51:56
 # **********************************************************
"""
ClientBucket: one race x sex x age_range combination, plus the prompt
template used to generate its photos.

Lives at <proj-root>/src/cloud_img_procurement/client_bucket.py.
"""

from cloud_img_procurement.bucket_enums import AgeRange, Race, Sex


class OfficePromptTemplate:
    """Builds the image-generation prompt for a bucket.

    OFFICE_DESCRIPTION and FRAMING_INSTRUCTION are held fixed across
    every bucket -- the main lever for keeping the whole library
    visually consistent (same office, same composition), cheaper to
    enforce here than to notice inconsistency after generation and
    have to redo a subset.
    """

    OFFICE_DESCRIPTION = (
        "a warm, softly-lit therapist's office: a blue upholstered "
        "armchair, a bookshelf partially visible in the background, a "
        "framed picture on a cream-colored wall, a small potted plant"
    )
    FRAMING_INSTRUCTION = (
        "Medium-shot composition showing the client's face, torso, and "
        "both hands resting in their lap or on the armrests, seated in "
        "the armchair. Client is facing mostly toward the camera, "
        "well-lit, single subject, photorealistic."
    )

    def build(self, bucket: "ClientBucket") -> str:
        """Builds the full prompt string for one bucket.

        :param bucket: The demographic bucket to build a prompt for.
        :return: Prompt string to send to the image model.
        """
        return (
            f"Create a photorealistic image of a {bucket.age_range.value}"
            f"-year-old {bucket.sex.value} client of {bucket.race.prompt_label} "
            f"race/ethnicity, sitting in {self.OFFICE_DESCRIPTION}. "
            f"{self.FRAMING_INSTRUCTION}"
        )


class ClientBucket:
    """One (race, sex, age_range) combination a case vignette maps into.

    :param race: Race/ethnicity bucket.
    :param sex: Sex bucket.
    :param age_range: Age-range bucket.
    """

    def __init__(self, race: Race, sex: Sex, age_range: AgeRange):
        self.race = race
        self.sex = sex
        self.age_range = age_range

    @property
    def key(self) -> str:
        """Filesystem-safe bucket identifier.

        :return: e.g. 'caucasian_female_20s-30s' -- used as both the
            subdirectory name and the filename prefix, so a bucket's
            images are self-describing even if copied out of context.
        """
        return f"{self.race.value}_{self.sex.value}_{self.age_range.value}"

    def image_filename(self, index: int) -> str:
        """Filename for the index'th image of this bucket.

        :param index: Which image within the bucket (0-based).
        :return: e.g. 'caucasian_female_20s-30s_00.png'.
        """
        return f"{self.key}_{index:02d}.png"

    def __repr__(self) -> str:
        return f"<ClientBucket: {self.key}>"
