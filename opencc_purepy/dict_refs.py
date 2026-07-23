from dataclasses import dataclass
import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

try:
    from typing import Protocol
except ImportError:
    try:
        from typing_extensions import Protocol
    except ImportError:
        class Protocol(object):
            pass

DictSlot = Tuple[Dict[str, str], int]


class StarterUnionLike(Protocol):
    merged_map: Dict[str, str]
    bmp_mask: List[int]
    bmp_cap: List[int]
    astral_mask: Dict[str, int]
    astral_cap: Dict[str, int]
    cap: int

    def build_starter_index(self) -> None:
        ...


RoundInput = Union[None, DictSlot, List[DictSlot], StarterUnionLike]


def _check_delegates(
        segment_replace: Optional[Callable[..., str]],
        union_replace: Optional[Callable[..., str]],
) -> None:
    if segment_replace is not None:
        try:
            params = inspect.signature(segment_replace).parameters
            positional = [
                p for p in params.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            positional = []
        if positional and len(positional) < 3:
            raise TypeError("segment_replace must accept text, slots, and cap")

    if union_replace is not None:
        try:
            params = inspect.signature(union_replace).parameters
            positional = [
                p for p in params.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            positional = []
        if positional and len(positional) < 2:
            raise TypeError("union_replace must accept text and union")

    if segment_replace is not None and union_replace is not None and segment_replace is union_replace:
        raise TypeError("segment_replace and union_replace must differ")


@dataclass
class DictRefs:
    """Wrap up to three rounds of dictionary or starter-union conversion."""

    round_1: RoundInput
    round_2: Optional[RoundInput] = None
    round_3: Optional[RoundInput] = None
    _norm: Optional[List[Tuple[List[DictSlot], int]]] = None

    def with_round_2(self, round_2: RoundInput) -> "DictRefs":
        self.round_2 = round_2
        self._norm = None
        return self

    def with_round_3(self, round_3: RoundInput) -> "DictRefs":
        self.round_3 = round_3
        self._norm = None
        return self

    @staticmethod
    def _is_starter_union_like(value: object) -> bool:
        return value is not None and hasattr(value, "merged_map") and hasattr(value, "cap")

    @staticmethod
    def _as_slots_and_cap(value: RoundInput) -> Tuple[List[DictSlot], int]:
        if value is None:
            return [], 0

        if DictRefs._is_starter_union_like(value):
            union = cast(StarterUnionLike, value)
            return [(union.merged_map, union.cap)], union.cap

        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict):
            dictionary, length = value
            return [(dictionary, length)], length

        if isinstance(value, list):
            slots: List[DictSlot] = []
            max_len = 0
            for item in value:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
                    dictionary, length = item
                    slot_len = length
                    slots.append((dictionary, slot_len))
                    max_len = max(max_len, slot_len)
                else:
                    raise TypeError("Round list contains non-slot entry: {}".format(type(item)))
            return slots, max_len

        raise TypeError("Unsupported round input type: {}".format(type(value)))

    def _normalize(self) -> List[Tuple[List[DictSlot], int]]:
        if self._norm is None:
            self._norm = [
                self._as_slots_and_cap(r)
                for r in (self.round_1, self.round_2, self.round_3)
            ]
        assert self._norm is not None
        return self._norm

    def _get_max_lengths(self) -> List[int]:
        return [cap for _slots, cap in self._normalize()]

    def apply_segment_replace(
            self,
            input_text: str,
            segment_replace: Optional[Callable[[str, List[DictSlot], int], str]] = None,
            union_replace: Optional[Callable[[str, StarterUnionLike], str]] = None,
            validate_delegates: bool = True,
    ) -> str:
        if validate_delegates:
            _check_delegates(segment_replace, union_replace)

        text = input_text

        for round_input in (self.round_1, self.round_2, self.round_3):
            if not round_input:
                continue

            if self._is_starter_union_like(round_input) and union_replace is not None:
                union = cast(StarterUnionLike, round_input)
                if not getattr(union, "_indexed", False):
                    union.build_starter_index()
                text = union_replace(text, union)
                continue

            slots, cap = self._as_slots_and_cap(round_input)
            if slots and cap > 0 and segment_replace is not None:
                text = segment_replace(text, slots, cap)

        return text
