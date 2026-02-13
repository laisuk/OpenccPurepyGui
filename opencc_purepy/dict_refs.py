from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union, List, Optional, Callable
import inspect

DictSlot = Tuple[Dict[str, str], int]
"""Tuple representing a dictionary and its maximum word length.

The first element is a mapping of source strings to target strings,
and the second element is the maximum phrase length in that dictionary.
"""

StarterUnionT = Any  # type: ignore
"""Type placeholder for StarterUnion.

This is typically a union data structure used internally for
starter-index optimizations. Declared as `Any` to avoid circular
imports at type-check time.
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
        segment_replace: Optional[Callable],
        union_replace: Optional[Callable],
) -> None:
    """
        Validate delegate function signatures for segment and union replacement.

        This function inspects the callables provided for segment-based and
        union-based replacement to ensure they have the expected number of
        positional parameters. It raises a ``TypeError`` if a delegate
        appears to have been mis-specified.

        Args:
            segment_replace (Optional[Callable]):
                A function expected to accept three parameters:
                ``(text: str, slots: List[DictSlot], cap: int)``.
                Used for performing dictionary-based segment replacement.
            union_replace (Optional[Callable]):
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
            - If a delegate function cannot be introspected (e.g., built-ins or
              C-extensions), it is accepted and any runtime errors are deferred
              until actual invocation.
        """
    # segment_replace must accept (text, slots, cap)
    if segment_replace is not None:
        try:
            params = inspect.signature(segment_replace).parameters
            pos = [p for p in params.values()
                   if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if len(pos) < 3:
                raise TypeError(
                    "segment_replace must accept (text:str, slots:List[DictSlot], cap:int). "
                    "Did you pass convert_union/union_replace by mistake?"
                )
        except (TypeError, ValueError):
            pass  # not introspectable: allow and let runtime raise

    # union_replace must accept (text, union)
    if union_replace is not None:
        try:
            params = inspect.signature(union_replace).parameters
            pos = [p for p in params.values()
                   if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if len(pos) < 2:
                raise TypeError(
                    "union_replace must accept (text:str, union:StarterUnion). "
                    "Did you pass convert_segment/segment_replace by mistake?"
                )
        except (TypeError, ValueError):
            pass

    if segment_replace is not None and union_replace is not None and segment_replace is union_replace:
        raise TypeError("segment_replace and union_replace refer to the same function; they must differ.")


@dataclass
class DictRefs:
    """
    Wrap up to 3 rounds of dictionary application for multi-pass conversion.

    Each round can be:
      - A list of `(dict, max_len)` slots (legacy shape), or
      - A single `(dict, max_len)` slot, or
      - A `StarterUnion` (we'll treat it as one merged slot)

    This keeps `segment_replace(text, dictionaries, max_word_length)` unchanged:
      - `dictionaries` is the list[List[(dict, max_len)]] for that round
      - `max_word_length` is computed from the provided round content
    """
    round_1: RoundInput
    round_2: Optional[RoundInput] = None
    round_3: Optional[RoundInput] = None

    # Cached normalized form: a list of (dictionaries, max_len) per round
    _norm: Optional[List[Tuple[List[DictSlot], int]]] = None

    # ----- Fluent builders -----
    def with_round_2(self, round_2: RoundInput) -> "DictRefs":
        self.round_2 = round_2
        self._norm = None
        return self

    def with_round_3(self, round_3: RoundInput) -> "DictRefs":
        self.round_3 = round_3
        self._norm = None
        return self

    # ----- Normalization -----
    @staticmethod
    def _as_slots_and_cap(inp: RoundInput) -> Tuple[List[DictSlot], int]:
        """
        Normalize a round input into (list_of_slots, max_len).
        """
        if inp is None:
            return [], 0

        # StarterUnion -> single merged slot
        if "StarterUnion" in str(type(inp)):
            u = inp  # type: ignore
            # If caller wants masks/caps, they should have built them already.
            return [(u.merged_map, int(u.cap))], int(u.cap)

        # Single (dict, max_len) slot
        if isinstance(inp, tuple) and len(inp) == 2 and isinstance(inp[0], dict):
            d, L = inp
            return [(d, int(L))], int(L)

        # List of slots
        if isinstance(inp, list):
            slots: List[DictSlot] = []
            max_len = 0
            for item in inp:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
                    d, L = item
                    L = int(L)
                    slots.append((d, L))
                    if L > max_len:
                        max_len = L
                else:
                    raise TypeError(f"Round list contains non-slot entry: {type(item)}")
            return slots, max_len

        raise TypeError(f"Unsupported round input type: {type(inp)}")

    def _normalize(self) -> List[Tuple[List[DictSlot], int]]:
        """
            Normalize all configured dictionary rounds into a standard form.

            This method consolidates up to three rounds of dictionary data
            (`round_1`, `round_2`, `round_3`) into a uniform representation:
            a list of tuples, where each tuple contains:

              - A list of ``DictSlot`` items (dictionaries with max length).
              - An integer representing the maximum word length (`cap`) across
                those dictionaries.

            Results are cached in ``self._norm`` to avoid recomputation.

            Returns:
                List[Tuple[List[DictSlot], int]]:
                    A list of normalized rounds in the order
                    ``[round_1, round_2, round_3]``.

                    Each element is a 2-tuple:
                    ``(slots, cap)`` where:
                      * ``slots`` is a list of dictionaries for that round.
                      * ``cap`` is the maximum word length for those dictionaries.

            Notes:
                - If a round is ``None`` or empty, it is normalized to
                  ``([], 0)``.
                - Subsequent calls reuse the cached ``self._norm`` value.
                - Internal services ``_as_slots_and_cap`` performs the actual
                  unpacking and maximum-length calculation.

            Example:
                >>> d = DictRefs(round_1=[({"a":"b"}, 1)])
                >>> d._normalize()
                [([({"a":"b"}, 1)], 1), ([], 0), ([], 0)]
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

    # ----- Application -----

    def apply_segment_replace(
            self,
            input_text: str,
            *,
            segment_replace: Optional[Callable[[str, List[DictSlot], int], str]] = None,
            union_replace: Optional[Callable[[str, "StarterUnionT"], str]] = None,  # e.g. opencc.convert_union
    ) -> str:
        """
        Unified 3-round apply. You can pass:
          - segment_replace=opencc.segment_replace (legacy driver), or
          - segment_replace=opencc.convert_segment (direct core), or
          - union_replace=opencc.convert_union (StarterUnion fast path),
        or mix them; per round we pick the most natural path.

        Behavior:
          • If a round is a StarterUnion and union_replace exists → use it (ensure index built).
          • Else normalize to (slots, cap) and:
              - use segment_replace if provided, otherwise
              - merge slots → StarterUnion and use union_replace if provided,
              - otherwise skip the round (no delegate).
        """
        _check_delegates(segment_replace, union_replace)

        # lazy import to avoid cycles
        try:
            from .starter_union import StarterUnion  # type: ignore
        except (TypeError, KeyError, ValueError):
            StarterUnion = None  # type: ignore

        def _is_union(obj) -> bool:
            return bool(obj) and StarterUnion is not None and isinstance(obj, StarterUnion)

        def _ensure_index(_u):
            if not getattr(_u, "_indexed", False):
                build = getattr(_u, "build_starter_index", None)
                if callable(build):
                    build()

        def _merge_to_union(_slots: List[DictSlot]):
            if StarterUnion is None:
                return None
            merger = getattr(StarterUnion, "merge_precedence", None)
            return merger(_slots) if callable(merger) else None

        text = input_text
        for r in (self.round_1, self.round_2, self.round_3):
            if not r:
                continue

            # Native union + union delegate → fastest path
            if _is_union(r) and union_replace is not None:
                u = r
                _ensure_index(u)
                text = union_replace(text, u)
                continue

            # Normalize to legacy slots/cap
            try:
                slots, cap = self._as_slots_and_cap(r)
            except (TypeError, KeyError, ValueError):
                # try to salvage if r looks like a union
                if _is_union(r):
                    to_slots = getattr(r, "to_slots", None)
                    max_cap = getattr(r, "max_cap", 0)
                    if callable(to_slots):
                        slots: List[Tuple[dict, int]] = to_slots() if callable(to_slots) else []
                        cap = int(max_cap) if max_cap else max((m for (_d, m) in slots), default=0)
                    else:
                        merged_map = getattr(r, "merged_map", None)
                        max_cap = getattr(r, "max_cap", 0)
                        if isinstance(merged_map, dict):
                            slots = [(merged_map, int(max_cap) if max_cap else 0)]
                            cap = int(max_cap) if max_cap else max((len(k) for k in merged_map), default=0)
                        else:
                            slots, cap = [], 0
                else:
                    slots, cap = [], 0

            if not slots or cap <= 0:
                continue

            # Use legacy driver or direct core convert_segment
            if segment_replace is not None:
                text = segment_replace(text, slots, cap)
                continue

            # Or merge to union and use convert_union
            if union_replace is not None:
                u = _merge_to_union(slots)
                if u is not None:
                    _ensure_index(u)
                    text = union_replace(text, u)

        return text
