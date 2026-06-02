# dict_slot.py

from enum import Enum
from typing import Union


class DictSlot(str, Enum):
    ST_CHARACTERS = "st_characters"
    ST_PHRASES = "st_phrases"
    ST_PUNCTUATIONS = "st_punctuations"
    TS_CHARACTERS = "ts_characters"
    TS_PHRASES = "ts_phrases"
    TS_PUNCTUATIONS = "ts_punctuations"
    TW_PHRASES = "tw_phrases"
    TW_PHRASES_REV = "tw_phrases_rev"
    TW_VARIANTS_PHRASES = "tw_variants_phrases"
    TW_VARIANTS = "tw_variants"
    TW_VARIANTS_REV = "tw_variants_rev"
    TW_VARIANTS_REV_PHRASES = "tw_variants_rev_phrases"
    HK_VARIANTS_PHRASES = "hk_variants_phrases"
    HK_VARIANTS = "hk_variants"
    HK_VARIANTS_REV = "hk_variants_rev"
    HK_VARIANTS_REV_PHRASES = "hk_variants_rev_phrases"
    JPS_CHARACTERS = "jps_characters"
    JPS_PHRASES = "jps_phrases"
    JP_VARIANTS = "jp_variants"
    JP_VARIANTS_REV = "jp_variants_rev"


DictSlotLike = Union[DictSlot, str]
