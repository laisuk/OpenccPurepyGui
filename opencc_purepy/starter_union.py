from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

DictSlot = Tuple[Dict[str, str], int]


@dataclass
class StarterUnion:
    """Merged, precedence-aware union of 1..N dictionary *slots* **plus**
    optional per-starter length index (mask + cap), derived from keys.

    This version assumes the legacy JSON (no precomputed caps/masks). We build
    masks/caps lazily from `merged_map` via :meth:`build_starter_index`.

    Precedence: **earlier** slots win; later slots only fill missing keys.
    `cap` is the *global* maximum phrase length across slots.

    Mask/Cap semantics (same as opencc-fmmseg):
    - For each starter `ch`, `mask` uses bit layout:
      bit 0 -> len==1, bit 1 -> len==2, ..., bit 63 -> len>=64 (CAP bucket).
    - `cap` arrays store the maximum length (in characters) per starter.
    - BMP starters (<= 0xFFFF) are dense arrays; astral are sparse dicts.
    """

    merged_map: Dict[str, str]
    cap: int

    # Lazy-built starter index
    bmp_mask: List[int] = field(default_factory=lambda: [0] * 0x10000)
    bmp_cap: List[int] = field(default_factory=lambda: [0] * 0x10000)
    astral_mask: Dict[str, int] = field(default_factory=dict)
    astral_cap: Dict[str, int] = field(default_factory=dict)
    _indexed: bool = False

    @staticmethod
    def merge_precedence(slots: Iterable[DictSlot]) -> "StarterUnion":
        merged: Dict[str, str] = {}
        max_len = 0
        for d, m_len in slots:
            if d:
                for k, v in d.items():
                    if k not in merged:
                        merged[k] = v
            max_len = max(max_len, int(m_len))
        return StarterUnion(merged_map=merged, cap=max_len)

    # ----- Starter index builder (lazy) -----
    def build_starter_index(self) -> None:
        """Populate per-starter masks and caps by scanning `merged_map` keys.
        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._indexed:
            return

        # Local aliases for speed
        bmp_mask = self.bmp_mask
        bmp_cap = self.bmp_cap
        a_mask = self.astral_mask
        a_cap = self.astral_cap

        def bit_for(length: int) -> int:
            if length <= 0:
                return 0
            if length >= 64:
                return 1 << 63
            return 1 << (length - 1)

        # Build from keys in merged_map
        for key in self.merged_map.keys():
            if not key:
                continue
            starter = key[0]
            key_len = len(key)
            b = bit_for(key_len)
            code = ord(starter)
            if code <= 0xFFFF:
                idx = code
                bmp_mask[idx] |= b
                if key_len > bmp_cap[idx]:
                    bmp_cap[idx] = key_len
            else:
                prev = a_mask.get(starter, 0)
                a_mask[starter] = prev | b
                if key_len > a_cap.get(starter, 0):
                    a_cap[starter] = key_len

        self._indexed = True

    @property
    def indexed(self):
        return self._indexed
