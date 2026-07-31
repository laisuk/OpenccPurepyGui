# dict_slot.py

from enum import Enum
from typing import Union


class DictSlot(str, Enum):
    STCharacters = "st_characters"
    STPhrases = "st_phrases"
    STPunctuations = "st_punctuations"

    TSCharacters = "ts_characters"
    TSPhrases = "ts_phrases"
    TSPunctuations = "ts_punctuations"

    TWPhrases = "tw_phrases"
    TWPhrasesRev = "tw_phrases_rev"

    TWVariantsPhrases = "tw_variants_phrases"
    TWVariants = "tw_variants"
    TWVariantsRev = "tw_variants_rev"
    TWVariantsRevPhrases = "tw_variants_rev_phrases"

    HKPhrases = "hk_phrases"
    HKPhrasesRev = "hk_phrases_rev"

    HKVariantsPhrases = "hk_variants_phrases"
    HKVariants = "hk_variants"
    HKVariantsRev = "hk_variants_rev"
    HKVariantsRevPhrases = "hk_variants_rev_phrases"

    JPSCharacters = "jps_characters"
    JPSCharactersRev = "jps_characters_rev"
    JPSPhrases = "jps_phrases"

    @classmethod
    def parse(cls, value: "DictSlotLike") -> "DictSlot":
        """Normalize a user-supplied dictionary slot.

        Accepts enum members, enum-style names, compact names, and canonical
        underscore-separated names.

        Args:
            value: Dictionary slot enum member or string representation.

        Returns:
            The normalized dictionary slot.

        Raises:
            TypeError: If value is neither a DictSlot nor a string.
            ValueError: If the slot name is unknown.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise TypeError(
                "Dictionary slot must be a DictSlot or str, got {}".format(
                    type(value).__name__
                )
            )

        key = value.strip().lower().replace("_", "")

        for member in cls:
            if member.name.lower() == key:
                return member

            if member.value.replace("_", "").lower() == key:
                return member

        raise ValueError("Unknown dictionary slot: {}".format(value))


DictSlotLike = Union[DictSlot, str]
