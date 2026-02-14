from __future__ import annotations

from typing import Optional, Tuple

# =============================================================================
# Whitespace / tail inspection helpers
# =============================================================================

def last_non_whitespace(s: str) -> Optional[str]:
    """Return the last non-whitespace character, or None."""
    i = len(s) - 1
    while i >= 0:
        ch = s[i]
        if not ch.isspace():
            return ch
        i -= 1
    return None


def last_two_non_whitespace(s: str) -> Optional[Tuple[str, str]]:
    """Return (last, prev) non-whitespace characters, or None if not enough."""
    last: Optional[str] = None

    i = len(s) - 1
    while i >= 0:
        ch = s[i]
        if not ch.isspace():
            if last is None:
                last = ch
            else:
                return last, ch
        i -= 1

    return None


def find_last_non_whitespace_index(s: str) -> Optional[int]:
    """Return the index of the last non-whitespace char, or None."""
    i = len(s) - 1
    while i >= 0:
        if not s[i].isspace():
            return i
        i -= 1
    return None


def find_prev_non_whitespace_index(s: str, end_exclusive: int) -> Optional[int]:
    """
    Return the index of the previous non-whitespace char strictly before end_exclusive, or None.

    end_exclusive is a Python string index (like slicing).
    """
    i = min(end_exclusive, len(s)) - 1
    while i >= 0:
        if not s[i].isspace():
            return i
        i -= 1
    return None


# =============================================================================
# Character classification helpers (ASCII / CJK)
# =============================================================================

_ASCII_MAX = 0x7F

# CJK (BMP-focused) blocks used by your reflow heuristics
_CJK_EXT_A_START = 0x3400
_CJK_EXT_A_END = 0x4DBF
_CJK_UNIFIED_START = 0x4E00
_CJK_UNIFIED_END = 0x9FFF
_CJK_COMPAT_START = 0xF900
_CJK_COMPAT_END = 0xFAFF

# Digits
_ASCII_DIGIT_START = 0x30
_ASCII_DIGIT_END = 0x39
_FULLWIDTH_DIGIT_START = 0xFF10
_FULLWIDTH_DIGIT_END = 0xFF19


def is_all_ascii(s: str) -> bool:
    """True if all chars are ASCII (<= 0x7F)."""
    for ch in s:
        if ord(ch) > _ASCII_MAX:
            return False
    return True


def is_cjk(ch: str) -> bool:
    """
    Minimal CJK checker (BMP focused).
    Designed for reflow heuristics, not full Unicode linguistics.
    """
    c = ord(ch)
    if _CJK_EXT_A_START <= c <= _CJK_EXT_A_END:
        return True
    if _CJK_UNIFIED_START <= c <= _CJK_UNIFIED_END:
        return True
    return _CJK_COMPAT_START <= c <= _CJK_COMPAT_END


def contains_any_cjk_str(s: str) -> bool:
    return any(is_cjk(ch) for ch in s)


# =============================================================================
# String pattern helpers (digits / mixed scripts / mostly CJK)
# =============================================================================

_NEUTRAL_ASCII_MIXED = {" ", "-", "/", ":", "."}


def is_all_ascii_digits(s: str) -> bool:
    """
    Match C# IsAllAsciiDigits:

    - ASCII space ' ' is neutral (allowed)
    - ASCII digits '0'..'9' allowed
    - FULLWIDTH digits '０'..'９' allowed
    - Anything else rejects
    - Must contain at least one digit (ASCII or fullwidth)
    """
    has_digit = False

    for ch in s:
        if ch == " ":
            continue

        o = ord(ch)
        if _ASCII_DIGIT_START <= o <= _ASCII_DIGIT_END:
            has_digit = True
            continue

        if _FULLWIDTH_DIGIT_START <= o <= _FULLWIDTH_DIGIT_END:
            has_digit = True
            continue

        return False

    return has_digit


def is_mixed_cjk_ascii(s: str) -> bool:
    """
    Match C# IsMixedCjkAscii:

    - Neutral ASCII allowed but does not count as ASCII content: ' ', '-', '/', ':', '.'
    - ASCII letters/digits count as ASCII content, other ASCII punctuation rejects
    - FULLWIDTH digits count as ASCII content
    - CJK chars count as CJK content
    - Any other non-ASCII non-CJK rejects
    - Early return True once both seen
    """
    has_cjk = False
    has_ascii = False

    for ch in s:
        if ch in _NEUTRAL_ASCII_MIXED:
            continue

        o = ord(ch)

        if o <= _ASCII_MAX:
            # Only ASCII letters/digits are allowed (and count)
            if ("0" <= ch <= "9") or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                has_ascii = True
            else:
                return False

        elif _FULLWIDTH_DIGIT_START <= o <= _FULLWIDTH_DIGIT_END:
            has_ascii = True

        elif is_cjk(ch):
            has_cjk = True

        else:
            return False

        if has_cjk and has_ascii:
            return True

    return False


def is_mostly_cjk(s: str) -> bool:
    """
    Heuristic: count only meaningful letters:
    - whitespace: neutral
    - digits (ASCII / fullwidth): neutral
    - CJK: counts toward CJK
    - ASCII alphabetic: counts toward ASCII
    - other punctuation/symbols: neutral
    """
    cjk = 0
    ascii_ = 0

    for ch in s:
        if ch.isspace():
            continue

        o = ord(ch)

        # neutral digits
        if _ASCII_DIGIT_START <= o <= _ASCII_DIGIT_END:
            continue
        if _FULLWIDTH_DIGIT_START <= o <= _FULLWIDTH_DIGIT_END:
            continue

        if is_cjk(ch):
            cjk += 1
        elif o <= _ASCII_MAX and ch.isalpha():
            ascii_ += 1
        # else: symbols / punctuation -> neutral

    return cjk > 0 and cjk >= ascii_


# =============================================================================
# All-CJK helpers (Rust-equivalent API)
# =============================================================================

def is_all_cjk(s: str, allow_whitespace: bool) -> bool:
    """
    Rust-equivalent:

    - Iterate chars
    - If whitespace:
        - reject if allow_whitespace=False
        - otherwise ignore
    - For non-whitespace:
        - mark seen=True
        - must be CJK (via is_cjk)
    - Returns False for empty / whitespace-only strings
    """
    seen = False

    for ch in s:
        if ch.isspace():
            if not allow_whitespace:
                return False
            continue

        seen = True
        if not is_cjk(ch):
            return False

    return seen


def is_all_cjk_ignoring_ws(s: str) -> bool:
    return is_all_cjk(s, True)


def is_all_cjk_no_ws(s: str) -> bool:
    return is_all_cjk(s, False)
