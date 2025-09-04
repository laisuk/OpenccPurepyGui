from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List

# Reuse DictSlot and StarterUnion from starter_union.py
from .starter_union import DictSlot, StarterUnion


class UnionKey(Enum):
    # S2T / T2S (no punctuation – we convert punctuation separately)
    S2T = auto()
    T2S = auto()

    # TW helpers
    TwPhrasesOnly = auto()
    TwVariantsOnly = auto()
    TwPhrasesRevOnly = auto()
    TwRevPair = auto()
    Tw2SpR1TwRevTriple = auto()

    # HK helpers
    HkVariantsOnly = auto()
    HkRevPair = auto()

    # JP helpers
    JpVariantsOnly = auto()
    JpRevTriple = auto()


class UnionCache:
    """On-demand builder & cache for *starter unions* and merged-slot dicts.

    Parameters
    ----------
    dictionary : object
        An instance exposing OpenCC-style dict *slots* as `(dict, max_len)` tuples.

    Notes
    -----
    - Punctuation slots are intentionally excluded.
    - Cache keys are stable `UnionKey` values (slot fingerprint).
    - Legacy JSON only: we derive masks/caps from keys when requested.
    """

    def __init__(self, dictionary: object) -> None:
        self._dict = dictionary
        self._union_by_key: Dict[UnionKey, StarterUnion] = {}
        self._merged_slot_by_key: Dict[UnionKey, DictSlot] = {}

    # ---------------- Public API ----------------
    def get_union(self, key: UnionKey, *, indexed: bool = False) -> StarterUnion:
        u = self._union_by_key.get(key)
        if u is None:
            u = self._build_union(key)
            self._union_by_key[key] = u
        if indexed and not u.indexed:
            u.build_starter_index()
        return u

    def get_merged_slot(self, key: UnionKey) -> DictSlot:
        s = self._merged_slot_by_key.get(key)
        if s is None:
            u = self.get_union(key)
            s = (u.merged_map, u.cap)
            self._merged_slot_by_key[key] = s
        return s

    # Convenience: ensure union has mask/cap built
    def ensure_indexed(self, key: UnionKey) -> StarterUnion:
        return self.get_union(key, indexed=True)

    # ---------------- Builders ----------------
    def _build_union(self, key: UnionKey) -> StarterUnion:
        slots = self._slots_for(key)
        return StarterUnion.merge_precedence(slots)

    def _slots_for(self, key: UnionKey) -> List[DictSlot]:
        g = self._get
        if key is UnionKey.S2T:
            return [g("st_phrases"), g("st_characters")]
        if key is UnionKey.T2S:
            return [g("ts_phrases"), g("ts_characters")]

        # ---------- TW helpers ----------
        if key is UnionKey.TwPhrasesOnly:
            return [g("tw_phrases")]
        if key is UnionKey.TwVariantsOnly:
            return [g("tw_variants")]
        if key is UnionKey.TwPhrasesRevOnly:
            return [g("tw_phrases_rev")]
        if key is UnionKey.TwRevPair:
            return [g("tw_variants_rev_phrases"), g("tw_variants_rev")]
        if key is UnionKey.Tw2SpR1TwRevTriple:
            return [g("tw_phrases_rev"), g("tw_variants_rev_phrases"), g("tw_variants_rev")]

        # ---------- HK helpers ----------
        if key is UnionKey.HkVariantsOnly:
            return [g("hk_variants")]
        if key is UnionKey.HkRevPair:
            return [g("hk_variants_rev_phrases"), g("hk_variants_rev")]

        # ---------- JP helpers ----------
        if key is UnionKey.JpVariantsOnly:
            return [g("jp_variants")]
        if key is UnionKey.JpRevTriple:
            return [g("jps_phrases"), g("jps_characters"), g("jp_variants_rev")]

        raise KeyError(f"UnionKey not handled: {key}")

    # ---------------- Helpers ----------------
    def _get(self, attr: str) -> DictSlot:
        slot = getattr(self._dict, attr, None)
        if not slot:
            return {}, 0
        d, cap = slot
        return d, int(cap) if cap is not None else 0