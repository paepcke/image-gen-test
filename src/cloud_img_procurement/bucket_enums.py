 # **********************************************************
 # @Author: Andreas Paepcke
 # @Date:   2026-09-08 19:12:44
 # @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/cloud_img_procurement/bucket_enums.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2026-09-28 17:51:56
 # **********************************************************
"""
Enums for the client-photo bucket dimensions: race, sex, age range.

Lives at <proj-root>/src/cloud_img_procurement/bucket_enums.py.
Subclasses SerializableEnum (common/serializable_enum.py) rather than
a locally-invented base class, so these round-trip through
common/json_plus.py the same way as every other enum in the
therapist_trainer codebase (e.g. Diagnosis, Role, AllDefenses in
classes.py/constants.py) -- and so any JS partner already decoding
that wire format can decode these too.

random_choice() is added per-class as a @staticmethod, mirroring
Diagnosis.random_choice() in classes.py, rather than via a shared
mixin -- that's the established pattern in this codebase, not
something SerializableEnum provides itself.

The case-generation LLM prompt in therapist_trainer must be
constrained to emit exactly these value strings for its race/sex/age
fields -- a runtime bucket lookup only works if both sides use
identical vocabulary (currently that prompt's race/sex fields are
free-form; see chat).
"""

import random

from common.serializable_enum import SerializableEnum

# Wording used for a Race in the image-generation prompt, where it
# differs from the machine-friendly enum value (which also becomes the
# bucket directory/file prefix and so must stay underscore/ASCII-safe).
# Races not listed here fall through to their raw value, which keeps
# the prompts -- and so any regeneration -- of the pre-existing
# buckets (caucasian, indian, asian, black_brown) byte-identical.
_RACE_PROMPT_LABELS = {
    'hispanic': 'Hispanic',
    'middle_eastern': 'Middle Eastern',
}


class Race(SerializableEnum):
    """Race/ethnicity bucket for a generated client photo.

    Vocabulary must match the therapist_trainer case-generation
    prompt's race field. HISPANIC replaced the earlier MEXICAN
    ('mexican') bucket to match ThTrainer's classification;
    MIDDLE_EASTERN was added to cover Iranian and other Middle Eastern
    clients. New members go at the END so itertools.product() bucket
    order (and thus --max-buckets smoke tests) stays stable.
    """
    CAUCASIAN = 'caucasian'
    INDIAN = 'indian'
    HISPANIC = 'hispanic'
    ASIAN = 'asian'
    POC = 'black_brown'
    MIDDLE_EASTERN = 'middle_eastern'

    @staticmethod
    def random_choice():
        """Returns a random member of the Race enum.

        :return: A randomly chosen Race member.
        """
        return random.choice(list(Race))

    @property
    def prompt_label(self) -> str:
        """Human-readable wording of this race for image-generation prompts.

        :return: e.g. 'Middle Eastern' for MIDDLE_EASTERN; the raw
            value for members with no special label.
        """
        return _RACE_PROMPT_LABELS.get(self.value, self.value)


class Sex(SerializableEnum):
    """Sex bucket for a generated client photo."""
    FEMALE = 'female'
    MALE = 'male'

    @staticmethod
    def random_choice():
        """Returns a random member of the Sex enum.

        :return: A randomly chosen Sex member.
        """
        return random.choice(list(Sex))


class AgeRange(SerializableEnum):
    """Age-range bucket for a generated client photo."""
    A20_30 = '20s-30s'
    A40_50 = '40s-50s'
    A60_PLUS = '60s-90s'

    @staticmethod
    def random_choice():
        """Returns a random member of the AgeRange enum.

        :return: A randomly chosen AgeRange member.
        """
        return random.choice(list(AgeRange))
