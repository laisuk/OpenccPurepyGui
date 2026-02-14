from __future__ import annotations

from typing import Dict, Optional

# =============================================================================
# Punctuation / Dialog / Brackets
#
# Design principles:
# - Single source of truth for pairs (open→close).
# - Precomputed sets/dicts for O(1) membership and matching.
# - Helpers stay tiny and predictable (no hidden trimming rules except where stated).
# =============================================================================

# -----------------------------------------------------------------------------
# Clause / sentence end punctuation
# -----------------------------------------------------------------------------

CJK_PUNCT_END: tuple[str, ...] = (
    # CJK / full-width / typography
    "。", "！", "？", "；", "：", "…", "—",
    "”", "’", "」", "』",
    "）", "】", "》", "〗", "〕", "〉", "］", "｝", "＞",
    # ASCII
    ".", "!", "?", ")", ":",
)

_CJK_PUNCT_END_SET: set[str] = set(CJK_PUNCT_END)


def is_clause_or_end_punct(ch: str) -> bool:
    """Clause-ending or sentence-ending punctuation."""
    return ch in _CJK_PUNCT_END_SET


# -----------------------------------------------------------------------------
# Dialog quotes (single source of truth)
# -----------------------------------------------------------------------------

DIALOG_OPEN_TO_CLOSE: Dict[str, str] = {
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "﹁": "﹂",
    "﹃": "﹄",
}

DIALOG_CLOSE_TO_OPEN: Dict[str, str] = {close: open_ for open_, close in DIALOG_OPEN_TO_CLOSE.items()}

DIALOG_OPENERS: tuple[str, ...] = tuple(DIALOG_OPEN_TO_CLOSE.keys())
DIALOG_CLOSERS: tuple[str, ...] = tuple(DIALOG_CLOSE_TO_OPEN.keys())

_DIALOG_OPENER_SET: set[str] = set(DIALOG_OPENERS)
_DIALOG_CLOSER_SET: set[str] = set(DIALOG_CLOSERS)


def is_dialog_opener(ch: str) -> bool:
    """Dialog opening mark."""
    return ch in _DIALOG_OPENER_SET


def is_dialog_closer(ch: str) -> bool:
    """Dialog closing mark."""
    return ch in _DIALOG_CLOSER_SET


def begins_with_dialog_opener(s: str) -> bool:
    """
    Return True if the first *non-space* char is a dialog opener.

    Trims only:
    - ASCII space: ' '
    - Full-width space: U+3000
    """
    i = 0
    n = len(s)
    while i < n and (s[i] == " " or s[i] == "\u3000"):
        i += 1
    return i < n and is_dialog_opener(s[i])


# -----------------------------------------------------------------------------
# Brackets (single source of truth: open → close)
# -----------------------------------------------------------------------------

BRACKET_PAIRS: tuple[tuple[str, str], ...] = (
    # Parentheses
    ("（", "）"),
    ("(", ")"),
    # Square brackets
    ("［", "］"),
    ("[", "]"),
    # Curly braces
    ("｛", "｝"),
    ("{", "}"),
    # Angle brackets
    ("＜", "＞"),
    ("<", ">"),
    ("⟨", "⟩"),
    ("〈", "〉"),
    # CJK brackets
    ("【", "】"),
    ("《", "》"),
    ("〔", "〕"),
    ("〖", "〗"),
)

_BRACKET_OPEN_TO_CLOSE: dict[str, str] = dict(BRACKET_PAIRS)
_BRACKET_CLOSE_TO_OPEN: dict[str, str] = {close: open_ for open_, close in BRACKET_PAIRS}
_BRACKET_OPEN_SET: set[str] = set(_BRACKET_OPEN_TO_CLOSE.keys())
_BRACKET_CLOSE_SET: set[str] = set(_BRACKET_CLOSE_TO_OPEN.keys())


def is_bracket_opener(ch: str) -> bool:
    return ch in _BRACKET_OPEN_SET


def is_bracket_closer(ch: str) -> bool:
    return ch in _BRACKET_CLOSE_SET


def is_matching_bracket(open_ch: str, close_ch: str) -> bool:
    return _BRACKET_OPEN_TO_CLOSE.get(open_ch) == close_ch


def try_get_matching_closer(open_ch: str) -> Optional[str]:
    return _BRACKET_OPEN_TO_CLOSE.get(open_ch)


def is_wrapped_by_matching_bracket(s: str, last_non_ws: str, min_len: int) -> bool:
    """
    True if string starts with an opening bracket and ends (at last_non_ws) with its matching closer.

    Notes
    -----
    - min_len=3 means at least: open + 1 char + close
    - `len()` is Unicode code points (good enough vs Rust `chars().count()` here).
    """
    if len(s) < min_len:
        return False
    return is_matching_bracket(s[0], last_non_ws)


# -----------------------------------------------------------------------------
# Punctuation helpers
# -----------------------------------------------------------------------------

_ALLOWED_POSTFIX_CLOSERS: set[str] = {")", "）"}
_STRONG_SENTENCE_END: set[str] = {"。", "！", "？", "!", "?"}
_COMMA_LIKE: set[str] = {"，", ",", "、"}
_COLON_LIKE: set[str] = {"：", ":"}
_ELLIPSIS_SUFFIXES: tuple[str, ...] = ("……", "...", "..", "…")


def is_allowed_postfix_closer(ch: str) -> bool:
    return ch in _ALLOWED_POSTFIX_CLOSERS


def ends_with_allowed_postfix_closer(s: str) -> bool:
    t = s.rstrip()
    return bool(t) and is_allowed_postfix_closer(t[-1])


def is_strong_sentence_end(ch: str) -> bool:
    return ch in _STRONG_SENTENCE_END


def is_comma_like(ch: str) -> bool:
    return ch in _COMMA_LIKE


def contains_any_comma_like(s: str) -> bool:
    return any(ch in _COMMA_LIKE for ch in s)


def is_colon_like(ch: str) -> bool:
    return ch in _COLON_LIKE


def ends_with_colon_like(s: str) -> bool:
    t = s.rstrip()
    return bool(t) and t[-1] in _COLON_LIKE


def ends_with_ellipsis(s: str) -> bool:
    t = s.rstrip()
    return t.endswith(_ELLIPSIS_SUFFIXES)
