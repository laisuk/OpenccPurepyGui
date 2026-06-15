from enum import Enum, auto
from typing import Dict, List

from .starter_union import DictSlot, StarterUnion


class UnionKey(Enum):
    S2T = auto()
    S2T_PUNCT = auto()
    T2S = auto()
    T2S_PUNCT = auto()
    TwPhrasesOnly = auto()
    TwVariantsPair = auto()
    S2TwpR2TwTriple = auto()
    TwPhrasesRevOnly = auto()
    TwRevPair = auto()
    Tw2SpR1TwRevTriple = auto()
    S2HkpR2HkTriple = auto()
    Hk2SpR1HkRevTriple = auto()
    HkVariantsPair = auto()
    HkRevPair = auto()
    JpsCharactersRev = auto()
    JpsPair = auto()


class UnionCache:
    """On-demand cache for starter-union dictionary combinations."""

    def __init__(self, dictionary: object) -> None:
        self._dict = dictionary
        self._union_by_key: Dict[UnionKey, StarterUnion] = {}
        self._merged_slot_by_key: Dict[UnionKey, DictSlot] = {}

    def get_union(self, key: UnionKey, indexed: bool = False) -> StarterUnion:
        union = self._union_by_key.get(key)
        if union is None:
            union = self._build_union(key)
            self._union_by_key[key] = union
        if indexed and not union.indexed:
            union.build_starter_index()
        return union

    def get_merged_slot(self, key: UnionKey) -> DictSlot:
        slot = self._merged_slot_by_key.get(key)
        if slot is None:
            union = self.get_union(key)
            slot = (union.merged_map, union.cap)
            self._merged_slot_by_key[key] = slot
        return slot

    def ensure_indexed(self, key: UnionKey) -> StarterUnion:
        return self.get_union(key, indexed=True)

    def _build_union(self, key: UnionKey) -> StarterUnion:
        return StarterUnion.merge_precedence(self._slots_for(key))

    def _slots_for(self, key: UnionKey) -> List[DictSlot]:
        get_slot = self._get

        if key is UnionKey.S2T:
            return [get_slot("st_phrases"), get_slot("st_characters")]
        if key is UnionKey.S2T_PUNCT:
            return [get_slot("st_phrases"), get_slot("st_characters"), get_slot("st_punctuations")]
        if key is UnionKey.T2S:
            return [get_slot("ts_phrases"), get_slot("ts_characters")]
        if key is UnionKey.T2S_PUNCT:
            return [get_slot("ts_phrases"), get_slot("ts_characters"), get_slot("ts_punctuations")]
        if key is UnionKey.TwPhrasesOnly:
            return [get_slot("tw_phrases")]
        if key is UnionKey.TwVariantsPair:
            return [get_slot("tw_variants_phrases"), get_slot("tw_variants")]
        if key is UnionKey.S2TwpR2TwTriple:
            return [get_slot("tw_phrases"), get_slot("tw_variants_phrases"), get_slot("tw_variants")]
        if key is UnionKey.TwPhrasesRevOnly:
            return [get_slot("tw_phrases_rev")]
        if key is UnionKey.TwRevPair:
            return [get_slot("tw_variants_rev_phrases"), get_slot("tw_variants_rev")]
        if key is UnionKey.Tw2SpR1TwRevTriple:
            return [get_slot("tw_phrases_rev"), get_slot("tw_variants_rev_phrases"), get_slot("tw_variants_rev")]
        if key is UnionKey.S2HkpR2HkTriple:
            return [get_slot("hk_phrases"), get_slot("hk_variants_phrases"), get_slot("hk_variants")]
        if key is UnionKey.Hk2SpR1HkRevTriple:
            return [get_slot("hk_phrases_rev"), get_slot("hk_variants_rev_phrases"), get_slot("hk_variants_rev")]
        if key is UnionKey.HkVariantsPair:
            return [get_slot("hk_variants_phrases"), get_slot("hk_variants")]
        if key is UnionKey.HkRevPair:
            return [get_slot("hk_variants_rev_phrases"), get_slot("hk_variants_rev")]
        if key is UnionKey.JpsCharactersRev:
            return [get_slot("jps_characters_rev")]
        if key is UnionKey.JpsPair:
            return [get_slot("jps_phrases"), get_slot("jps_characters")]

        raise KeyError("UnionKey not handled: {}".format(key))

    def _get(self, attr: str) -> DictSlot:
        slot = getattr(self._dict, attr, None)
        if not slot:
            return {}, 0
        dictionary, cap = slot
        return dictionary, int(cap) if cap is not None else 0
