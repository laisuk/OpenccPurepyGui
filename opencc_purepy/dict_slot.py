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
    JPSPhrases = "jps_phrases"

    JPVariants = "jp_variants"
    JPVariantsRev = "jp_variants_rev"


DictSlotLike = Union[DictSlot, str]
