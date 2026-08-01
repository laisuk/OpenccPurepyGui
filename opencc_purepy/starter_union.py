from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .dict_refs import DictSlot, StarterUnionLike


@dataclass
class StarterUnion(StarterUnionLike):
    """Precedence-aware merged dictionary slot with lazy per-starter indexes."""

    merged_map: Dict[str, str]
    cap: int
    bmp_mask: List[int] = field(default_factory=lambda: [0] * 0x10000)
    bmp_cap: List[int] = field(default_factory=lambda: [0] * 0x10000)
    astral_mask: Dict[str, int] = field(default_factory=dict)
    astral_cap: Dict[str, int] = field(default_factory=dict)
    _indexed: bool = False

    @staticmethod
    def merge_precedence(slots: Iterable[DictSlot]) -> "StarterUnion":
        slot_list = list(slots)

        if len(slot_list) == 1:
            dictionary, max_len = slot_list[0]
            return StarterUnion(merged_map=dictionary, cap=max_len)

        merged: Dict[str, str] = {}
        max_len = 0

        # Later update from earlier slots preserves existing conversion precedence:
        # the first slot in the requested order wins for duplicate keys.
        for dictionary, slot_max_len in reversed(slot_list):
            if dictionary:
                merged.update(dictionary)
            max_len = max(max_len, slot_max_len)

        return StarterUnion(merged_map=merged, cap=max_len)

    def build_starter_index(self) -> None:
        """Populate per-starter masks and caps from merged_map keys."""
        if self._indexed:
            return

        bmp_mask = self.bmp_mask
        bmp_cap = self.bmp_cap
        astral_mask = self.astral_mask
        astral_cap = self.astral_cap

        for key in self.merged_map:
            if not key:
                continue

            key_len = len(key)
            if key_len >= 64:
                bit = 1 << 63
            else:
                bit = 1 << (key_len - 1)

            starter = key[0]
            code = ord(starter)

            if code <= 0xFFFF:
                bmp_mask[code] |= bit
                if key_len > bmp_cap[code]:
                    bmp_cap[code] = key_len
            else:
                astral_mask[starter] = astral_mask.get(starter, 0) | bit
                if key_len > astral_cap.get(starter, 0):
                    astral_cap[starter] = key_len

        self._indexed = True

    @property
    def indexed(self) -> bool:
        return self._indexed
