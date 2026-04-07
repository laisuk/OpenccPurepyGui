from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, cast

DictSlot = Tuple[Dict[str, str], int]
"""Tuple representing a dictionary and its maximum word length.

The first element is a mapping of source strings to target strings,
and the second element is the maximum phrase length in that dictionary.
"""


class StarterUnionLike(Protocol):
    merged_map: Dict[str, str]
    bmp_mask: List[int]
    bmp_cap: List[int]
    astral_mask: Dict[str, int]
    astral_cap: Dict[str, int]
    cap: int

    def build_starter_index(self) -> None: ...


if TYPE_CHECKING:
    from .starter_union import StarterUnion as StarterUnionClass
else:
    StarterUnionClass = Any

try:
    from .starter_union import StarterUnion as starterUnionClass
except (ImportError, TypeError, KeyError, ValueError):
    starterUnionClass = None

StarterUnionT = StarterUnionLike
"""Type placeholder for StarterUnion.

This is typically a union data structure used internally for
starter-index optimizations.
"""

RoundInput = Union[
    None,
    DictSlot,
    List[DictSlot],
    StarterUnionT,
]
"""Union type describing valid inputs for a conversion round.

- ``None``: No dictionary or union provided.
- ``DictSlot``: A single dictionary with max length.
- ``List[DictSlot]``: Multiple dictionaries applied in order.
- ``StarterUnionT``: A pre-built union structure for fast lookup.
"""


def _check_delegates(
        segment_replace: Optional[Callable[..., str]],
        union_replace: Optional[Callable[..., str]],
) -> None:
    """
    Validate delegate function signatures for segment and union replacement.

    This function inspects the callables provided for segment-based and
    union-based replacement to ensure they have the expected number of
    positional parameters. It raises a ``TypeError`` if a delegate
    appears to have been mis-specified.

    Args:
        segment_replace:
            A function expected to accept three parameters:
            ``(text: str, slots: List[DictSlot], cap: int)``.
            Used for performing dictionary-based segment replacement.
        union_replace:
            A function expected to accept two parameters:
            ``(text: str, union: StarterUnion)``.
            Used for performing replacement with a pre-built union.

    Raises:
        TypeError:
            - If ``segment_replace`` does not accept at least 3 positional arguments.
            - If ``union_replace`` does not accept at least 2 positional arguments.
            - If both delegates are the same function reference.
        ValueError:
            Propagated if function signature introspection fails.

    Notes:
        - If a delegate function cannot be introspected (e.g. built-ins or
          C-extensions), it is accepted and any runtime errors are deferred
          until actual invocation.
    """
    if segment_replace is not None:
        try:
            params = inspect.signature(segment_replace).parameters
            pos = [
                p for p in params.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(pos) < 3:
                raise TypeError(
                    "segment_replace must accept (text:str, slots:List[DictSlot], cap:int). "
                    "Did you pass convert_union/union_replace by mistake?"
                )
        except (TypeError, ValueError):
            pass

    if union_replace is not None:
        try:
            params = inspect.signature(union_replace).parameters
            pos = [
                p for p in params.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(pos) < 2:
                raise TypeError(
                    "union_replace must accept (text:str, union:StarterUnion). "
                    "Did you pass convert_segment/segment_replace by mistake?"
                )
        except (TypeError, ValueError):
            pass

    if (
            segment_replace is not None
            and union_replace is not None
            and segment_replace is union_replace
    ):
        raise TypeError(
            "segment_replace and union_replace refer to the same function; they must differ."
        )


@dataclass
class DictRefs:
    """
    Wrap up to 3 rounds of dictionary application for multi-pass conversion.

    Each round can be:
      - A list of `(dict, max_len)` slots (legacy shape), or
      - A single `(dict, max_len)` slot, or
      - A `StarterUnion` (treated as one merged slot)

    This keeps `segment_replace(text, dictionaries, max_word_length)` unchanged:
      - `dictionaries` is the list[List[(dict, max_len)]] for that round
      - `max_word_length` is computed from the provided round content
    """

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
    def _is_starter_union_like(inp: object) -> bool:
        return (
                inp is not None
                and hasattr(inp, "merged_map")
                and hasattr(inp, "cap")
        )

    @staticmethod
    def _as_slots_and_cap(inp: RoundInput) -> Tuple[List[DictSlot], int]:
        """
        Normalize a round input into (list_of_slots, max_len).
        """
        if inp is None:
            return [], 0

        if DictRefs._is_starter_union_like(inp):
            u = cast(StarterUnionLike, inp)
            return [(u.merged_map, int(u.cap))], int(u.cap)

        if isinstance(inp, tuple) and len(inp) == 2 and isinstance(inp[0], dict):
            d, length = inp
            return [(d, int(length))], int(length)

        if isinstance(inp, list):
            slots: List[DictSlot] = []
            max_len = 0
            for item in inp:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
                    d, length = item
                    length = int(length)
                    slots.append((d, length))
                    if length > max_len:
                        max_len = length
                else:
                    raise TypeError(f"Round list contains non-slot entry: {type(item)}")
            return slots, max_len

        raise TypeError(f"Unsupported round input type: {type(inp)}")

    def _normalize(self) -> List[Tuple[List[DictSlot], int]]:
        """
        Normalize all configured dictionary rounds into a standard form.
        """
        if self._norm is not None:
            return self._norm

        rounds = [self.round_1, self.round_2, self.round_3]
        norm: List[Tuple[List[DictSlot], int]] = []

        for r in rounds:
            slots, cap = self._as_slots_and_cap(r)
            norm.append((slots, cap))

        self._norm = norm
        return norm

    def apply_segment_replace(
            self,
            input_text: str,
            *,
            segment_replace: Optional[Callable[[str, List[DictSlot], int], str]] = None,
            union_replace: Optional[Callable[[str, StarterUnionT], str]] = None,
            validate_delegates: bool = True,
    ) -> str:
        """
        Unified 3-round apply.

        You can pass:
          - segment_replace=opencc.segment_replace (legacy driver), or
          - segment_replace=opencc.convert_segment (direct core), or
          - union_replace=opencc.convert_union (StarterUnion fast path)

        Behavior:
          • If a round is a StarterUnion and union_replace exists → use it.
          • Else normalize to (slots, cap) and:
              - use segment_replace if provided, otherwise
              - merge slots → StarterUnion and use union_replace if provided,
              - otherwise skip the round.
        """
        if validate_delegates:
            _check_delegates(segment_replace, union_replace)

        starter_union_cls = starterUnionClass

        def _is_union(obj: object) -> bool:
            return (
                    obj is not None
                    and starter_union_cls is not None
                    and isinstance(obj, starter_union_cls)
            )

        def _ensure_index(union_object: StarterUnionLike) -> None:
            if not getattr(union_object, "_indexed", False):
                build = getattr(union_object, "build_starter_index", None)
                if callable(build):
                    build()

        def _merge_to_union(slot_list: List[DictSlot]) -> Optional[StarterUnionT]:
            if starter_union_cls is None:
                return None
            merger = getattr(starter_union_cls, "merge_precedence", None)
            if callable(merger):
                merged = merger(slot_list)
                if self._is_starter_union_like(merged):
                    return cast(StarterUnionT, merged)
            return None

        text = input_text

        for r in (self.round_1, self.round_2, self.round_3):
            if not r:
                continue

            if _is_union(r) and union_replace is not None:
                union_obj = cast(StarterUnionLike, r)
                _ensure_index(union_obj)
                text = union_replace(text, union_obj)
                continue

            try:
                slots, cap = self._as_slots_and_cap(r)
            except (TypeError, KeyError, ValueError):
                if _is_union(r):
                    union_obj = cast(object, r)
                    to_slots = getattr(union_obj, "to_slots", None)
                    max_cap = getattr(union_obj, "max_cap", 0)

                    if callable(to_slots):
                        raw_slots = to_slots()
                        slots = cast(List[DictSlot], raw_slots)
                        cap = int(max_cap) if max_cap else max((m for (_d, m) in slots), default=0)
                    else:
                        merged_map = getattr(union_obj, "merged_map", None)
                        if isinstance(merged_map, dict):
                            cap = int(max_cap) if max_cap else max((len(k) for k in merged_map), default=0)
                            slots = [(merged_map, cap)]
                        else:
                            slots, cap = [], 0
                else:
                    slots, cap = [], 0

            if not slots or cap <= 0:
                continue

            if segment_replace is not None:
                text = segment_replace(text, slots, cap)
                continue

            if union_replace is not None:
                union_obj = _merge_to_union(slots)
                if union_obj is not None:
                    _ensure_index(union_obj)
                    text = union_replace(text, union_obj)

        return text
