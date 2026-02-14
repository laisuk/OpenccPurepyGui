from __future__ import annotations

"""
reflow_helper.py

CJK paragraph reflow helpers for PDF/plain text extraction pipelines.

Design notes
------------
- This module is deliberately independent from PDF extraction backends.
- It focuses on paragraph reflow, headings/metadata detection, and simple
  cleanup heuristics for noisy OCR/PDF text.
- Keep the rules deterministic and easy to tune.

The public entry point is:
    reflow_cjk_paragraphs_core(text, add_pdf_page_header=..., compact=...)
"""

import re
from typing import List, Optional, Sequence, Tuple, Dict


# =============================================================================
# Optional cleanup helpers (kept outside extraction)
# =============================================================================

def collapse_consecutive_duplicate_lines(text: str) -> str:
    """
    Collapse consecutive duplicate *non-empty* lines (whitespace-insensitive).

    Useful for removing repeated headers/footers that occasionally leak into
    extracted text streams.
    """
    out: List[str] = []
    prev: Optional[str] = None

    for line in text.splitlines():
        key = line.strip()
        if not key:
            out.append(line)
            prev = None
            continue
        if prev is not None and key == prev:
            continue
        out.append(line)
        prev = key

    return "\n".join(out)


# =============================================================================
# Shared constants (CJK / dialog / metadata)
# =============================================================================

# Tuple definition (readable)
CJK_PUNCT_END = (
    "。", "！", "？", "；", "：", "…", "—", "”", "」", "’", "』",
    "）", "】", "》", "〗", "〔", "〉", "］", "｝", "＞",
    ".", "!", "?", ")", ":"
)

# Precompute for O(1) membership
_CJK_PUNCT_END_SET = set(CJK_PUNCT_END)


def is_clause_or_end_punct(ch: str) -> bool:
    """Return True if character is clause-ending or sentence-ending punctuation."""
    return ch in _CJK_PUNCT_END_SET


TITLE_HEADING_REGEX = re.compile(
    r"^(?!.*[,，])(?=.{0,50}$)"
    r".{0,10}?(前言|序章|终章|尾声|后记|番外.{0,15}?|尾聲|後記|第.{0,5}?([章节部卷節回][^分合的])|[卷章][一二三四五六七八九十](?:$|.{0,20}?))"
)

DIALOG_OPEN_TO_CLOSE: Dict[str, str] = {
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "﹁": "﹂",
    "﹃": "﹄",
}

DIALOG_CLOSE_TO_OPEN: Dict[str, str] = {
    v: k for k, v in DIALOG_OPEN_TO_CLOSE.items()
}

DIALOG_OPENERS = tuple(DIALOG_OPEN_TO_CLOSE.keys())
DIALOG_CLOSERS = tuple(DIALOG_CLOSE_TO_OPEN.keys())

_DIALOG_OPENER_SET = set(DIALOG_OPENERS)
_DIALOG_CLOSER_SET = set(DIALOG_CLOSERS)


def is_dialog_opener(ch: str) -> bool:
    """Return True if character is a dialog opening mark."""
    return ch in _DIALOG_OPENER_SET


def is_dialog_closer(ch: str) -> bool:
    """Return True if character is a dialog closing mark."""
    return ch in _DIALOG_CLOSER_SET


def begins_with_dialog_opener(s: str) -> bool:
    # Trim only ASCII space and full-width space (U+3000)
    i = 0
    length = len(s)

    while i < length and (s[i] == " " or s[i] == "\u3000"):
        i += 1

    if i >= length:
        return False

    return is_dialog_opener(s[i])


METADATA_SEPARATORS = ("：", ":", "　", "·", "・")
METADATA_KEYS = {
    "書名", "书名",
    "作者",
    "譯者", "译者",
    "校訂", "校订",
    "出版社",
    "出版時間", "出版时间",
    "出版日期",
    "版權", "版权",
    "版權頁", "版权页",
    "版權信息", "版权信息",
    "責任編輯", "责任编辑",
    "編輯", "编辑",
    "責編", "责编",
    "定價", "定价",
    "前言",
    "序章",
    "終章", "终章",
    "尾聲", "尾声",
    "後記", "后记",
    "品牌方",
    "出品方",
    "授權方", "授权方",
    "電子版權", "数字版权",
    "掃描", "扫描",
    "OCR",
    "CIP",
    "在版編目", "在版编目",
    "分類號", "分类号",
    "主題詞", "主题词",
    "發行日", "发行日",
    "初版",
    "ISBN",
}

# ------------------------------------------------------------
# Bracket pairs (open → close)
# Single source of truth
# ------------------------------------------------------------

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

_BRACKET_OPEN_SET = {open_ for open_, _ in BRACKET_PAIRS}
_BRACKET_CLOSE_SET = {close for _, close in BRACKET_PAIRS}

_BRACKET_OPEN_TO_CLOSE = dict(BRACKET_PAIRS)
_BRACKET_CLOSE_TO_OPEN = {close: open_ for open_, close in BRACKET_PAIRS}


def is_bracket_opener(ch: str) -> bool:
    return ch in _BRACKET_OPEN_SET


def is_bracket_closer(ch: str) -> bool:
    return ch in _BRACKET_CLOSE_SET


def is_matching_bracket(open_ch: str, close_ch: str) -> bool:
    return _BRACKET_OPEN_TO_CLOSE.get(open_ch) == close_ch


def is_wrapped_by_matching_bracket(s: str, last_non_ws: str, min_len: int) -> bool:
    # min_len=3 means at least: open + 1 char + close
    if not s:
        return False

    open_ch = s[0]

    # Equivalent to Rust's `s.chars().count() >= min_len`
    # (Python len() counts Unicode code points)
    if len(s) < min_len:
        return False

    return is_matching_bracket(open_ch, last_non_ws)


def try_get_matching_closer(open_ch: str) -> Optional[str]:
    return _BRACKET_OPEN_TO_CLOSE.get(open_ch)


_ALLOWED_POSTFIX_CLOSERS = {")", "）"}


def is_allowed_postfix_closer(ch: str) -> bool:
    return ch in _ALLOWED_POSTFIX_CLOSERS


def ends_with_allowed_postfix_closer(s: str) -> bool:
    # Trim only trailing whitespace
    s = s.rstrip()
    if not s:
        return False
    # Last non-whitespace character
    return is_allowed_postfix_closer(s[-1])


# ------------------------------------------------------------
# Sentence / punctuation helpers
# ------------------------------------------------------------

_STRONG_SENTENCE_END = {"。", "！", "？", "!", "?"}

_COMMA_LIKE = {"，", ",", "、"}

_COLON_LIKE = {"：", ":"}


def is_strong_sentence_end(ch: str) -> bool:
    return ch in _STRONG_SENTENCE_END


def is_comma_like(ch: str) -> bool:
    return ch in _COMMA_LIKE


def contains_any_comma_like(s: str) -> bool:
    # Generator short-circuits like Rust's .any()
    return any(ch in _COMMA_LIKE for ch in s)


def is_colon_like(ch: str) -> bool:
    return ch in _COLON_LIKE


def ends_with_colon_like(s: str) -> bool:
    t = s.rstrip()  # trim right only
    return bool(t) and t[-1] in _COLON_LIKE


_ELLIPSIS_SUFFIXES = ("……", "...", "..", "…")


def ends_with_ellipsis(s: str) -> bool:
    t = s.rstrip()
    return t.endswith(_ELLIPSIS_SUFFIXES)


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
    last = None

    i = len(s) - 1
    while i >= 0:
        ch = s[i]
        if not ch.isspace():
            if last is None:
                last = ch
            else:
                return last, ch  # (last, prev)
        i -= 1

    return None


def find_last_non_whitespace_index(s: str) -> Optional[int]:
    """Return the Python string index of the last non-whitespace char, or None."""
    i = len(s) - 1
    while i >= 0:
        if not s[i].isspace():
            return i
        i -= 1
    return None


def find_prev_non_whitespace_index(s: str, end_exclusive: int) -> Optional[int]:
    """
    Return the index of the previous non-whitespace char strictly before end_exclusive,
    or None.

    end_exclusive is a Python string index (like slicing).
    """
    i = min(end_exclusive, len(s)) - 1
    while i >= 0:
        if not s[i].isspace():
            return i
        i -= 1
    return None


# =============================================================================
# Low-level helpers (indent / CJK / box line / repeats)
# =============================================================================

def is_all_ascii(s: str) -> bool:
    for ch in s:
        if ord(ch) > 0x7F:
            return False
    return True


def is_cjk(ch: str) -> bool:
    """
    Minimal CJK checker (BMP focused).
    Designed for reflow heuristics, not full Unicode linguistics.
    """
    c = ord(ch)
    if 0x3400 <= c <= 0x4DBF:  # Extension A
        return True
    if 0x4E00 <= c <= 0x9FFF:  # Unified Ideographs
        return True
    return 0xF900 <= c <= 0xFAFF  # Compatibility Ideographs


def contains_any_cjk_str(s: str) -> bool:
    return any(is_cjk(ch) for ch in s)


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
        if 0x30 <= o <= 0x39:  # '0'..'9'
            has_digit = True
            continue
        if 0xFF10 <= o <= 0xFF19:  # '０'..'９'
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
        # Neutral ASCII
        if ch in (" ", "-", "/", ":", "."):
            continue

        o = ord(ch)
        if o <= 0x7F:
            # Only ASCII letters/digits are allowed (and count)
            if ("0" <= ch <= "9") or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                has_ascii = True
            else:
                return False
        elif 0xFF10 <= o <= 0xFF19:  # FULLWIDTH digits
            has_ascii = True
        elif is_cjk(ch):
            has_cjk = True
        else:
            return False

        if has_cjk and has_ascii:
            return True

    return False


def is_mostly_cjk(s: str) -> bool:
    cjk = 0
    ascii_ = 0

    for ch in s:
        # Neutral: whitespace
        if ch.isspace():
            continue

        o = ord(ch)

        # Neutral: ASCII digits
        if 0x30 <= o <= 0x39:
            continue

        # Neutral: FULLWIDTH digits
        if 0xFF10 <= o <= 0xFF19:
            continue

        if is_cjk(ch):
            cjk += 1
        elif o <= 0x7F and ch.isalpha():
            ascii_ += 1
        # else: symbols / punctuation → neutral (ignored)

    return cjk > 0 and cjk >= ascii_


def is_all_cjk_no_whitespace(s: str) -> bool:
    if not s:
        return False
    for ch in s:
        if ch.isspace():
            return False
        if not is_cjk(ch):
            return False
    return True


def is_all_cjk_ignoring_whitespace(s: str) -> bool:
    """
    Match C# IsAllCjkIgnoringWhitespace:
    - Ignore any Unicode whitespace
    - If any non-whitespace ASCII is present => false
    - Otherwise true (even if empty / whitespace-only)
    """
    for ch in s:
        if ch.isspace():
            continue
        if ord(ch) <= 0x7F:
            return False
    return True


def is_visual_divider_line(s: str) -> bool:
    """
    Detect visual divider lines (box drawing / ASCII separators).

    If True, we force a paragraph break.
    """
    if not s or s.isspace():
        return False

    total = 0
    for ch in s:
        if ch.isspace():
            continue
        total += 1

        if "\u2500" <= ch <= "\u257F":  # box drawing range
            continue

        if ch in ("-", "=", "_", "~", "·", "•", "*"):
            continue

        return False

    return total >= 3


IDEOGRAPHIC_SPACE = "\u3000"

# Common indent regex (raw_line based)
_INDENT_RE = re.compile(r"^\s{2,}")


def strip_half_width_indent_keep_fullwidth(s: str) -> str:
    """
    Strip ASCII/half-width indentation, but keep full-width IDEOGRAPHIC_SPACE.
    """
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]
        if ch == IDEOGRAPHIC_SPACE:
            break
        if ch.isspace() and ord(ch) <= 0x7F:
            i += 1
            continue
        break

    return s[i:]


def strip_all_left_indent_for_probe(s: str) -> str:
    """
    Probe indentation stripping: remove both half- and full-width indents.
    """
    return s.lstrip(" \t\r\n\u3000")


def collapse_repeated_segments(line: str) -> str:
    """
    Collapse repeated word sequences and repeated tokens for OCR noise.
    """
    if not line:
        return line
    parts = line.strip().split()
    if not parts:
        return line
    parts2 = collapse_repeated_word_sequences(parts)
    parts3 = [collapse_repeated_token(tok) for tok in parts2]
    return " ".join(parts3)


def collapse_repeated_word_sequences(parts: Sequence[str]) -> List[str]:
    min_repeats = 3
    max_phrase_len = 8

    n = len(parts)
    if n < min_repeats:
        return list(parts)

    for start in range(n):
        for phrase_len in range(1, max_phrase_len + 1):
            if start + phrase_len > n:
                break

            count = 1
            while True:
                next_start = start + count * phrase_len
                if next_start + phrase_len > n:
                    break

                equal = True
                for k in range(phrase_len):
                    if parts[start + k] != parts[next_start + k]:
                        equal = False
                        break
                if not equal:
                    break

                count += 1

            if count >= min_repeats:
                result: List[str] = []
                result.extend(parts[:start])
                result.extend(parts[start:start + phrase_len])
                tail_start = start + count * phrase_len
                result.extend(parts[tail_start:])
                return result

    return list(parts)


def collapse_repeated_token(token: Optional[str]) -> Optional[str]:
    """
    Collapse repeated unit patterns in a token, e.g.
    'ABC...ABC...ABC...' → 'ABC...'

    Only applies to medium-length tokens (4..200 chars) and unit sizes 4..10.
    """
    if token is None:
        return None

    length = len(token)
    if length < 4 or length > 200:
        return token

    for unit_len in range(4, 11):
        if unit_len > length // 3:
            break
        if length % unit_len != 0:
            continue

        unit = token[:unit_len]
        all_match = True
        for pos in range(0, length, unit_len):
            if token[pos:pos + unit_len] != unit:
                all_match = False
                break

        if all_match:
            return unit

    return token


# =============================================================================
# Dialog state
# =============================================================================

class DialogState:
    """
    Track unclosed dialog quotes across concatenated lines.
    """
    __slots__ = ("counts",)

    def __init__(self) -> None:
        self.counts = dict.fromkeys(DIALOG_OPEN_TO_CLOSE.keys(), 0)

    def reset(self) -> None:
        for k in self.counts:
            self.counts[k] = 0

    def update(self, s: str) -> None:
        for ch in s:
            if ch in DIALOG_OPEN_TO_CLOSE:
                self.counts[ch] += 1
            elif ch in DIALOG_CLOSE_TO_OPEN:
                open_ch = DIALOG_CLOSE_TO_OPEN[ch]
                if self.counts[open_ch] > 0:
                    self.counts[open_ch] -= 1

    @property
    def is_unclosed(self) -> bool:
        return any(v > 0 for v in self.counts.values())


# =============================================================================
# Reflow rule helpers (kept out of inner loops)
# =============================================================================

def has_unclosed_bracket(s: str) -> bool:
    """
    Strict bracket safety check (Rust-style):

    - Track openers on a stack.
    - If we see a closer with no opener => unsafe => True.
    - If opener/closer mismatch => unsafe => True.
    - At end: True if we saw any bracket and stack not empty.
    """
    if not s:
        return False

    stack: list[str] = []
    seen_bracket = False

    for ch in s:
        if is_bracket_opener(ch):
            seen_bracket = True
            stack.append(ch)
            continue

        if is_bracket_closer(ch):
            seen_bracket = True

            # STRICT: stray closer => unsafe
            if not stack:
                return True

            open_ch = stack.pop()
            if not is_matching_bracket(open_ch, ch):
                return True

    return seen_bracket and bool(stack)


def is_heading_like(s: str) -> bool:
    """
    Heuristic for detecting heading-like lines (aligned with your C# port).
    """
    if s is None:
        return False

    s = s.strip()
    if not s:
        return False

    # Page markers are not headings
    if s.startswith("=== ") and s.endswith("==="):
        return False

    # Unbalanced bracket lines are not headings
    if has_unclosed_bracket(s):
        return False

    length = len(s)
    if length < 2:
        return False

    last_ch = s[-1]

    # Bracket-wrapped titles: （xxx）, 【xxx】, etc.
    if is_wrapped_by_matching_bracket(s, last_ch, 3):
        return True

    max_len = 18 if is_all_ascii(s) or is_mixed_cjk_ascii(s) else 8

    # Short-circuit for item title-like: "物品准备："
    last = s[-1] if s else None
    if last is not None:
        # 1) Item-title like: "物品准备："
        if is_colon_like(last) and length < max_len:
            body = s[:-1]  # strip_last_char(s)
            if is_all_cjk_no_whitespace(body):
                return True

        # 2) Allowed postfix closer: ... ) / ） and no comma-like anywhere
        if is_allowed_postfix_closer(last):
            if not contains_any_comma_like(s):
                return True

        # 3) Ends with clause/sentence punctuation => not a heading
        if is_clause_or_end_punct(last):
            return False

    # Reject comma-ish headings
    if contains_any_comma_like(s):
        return False

    if length <= max_len:
        # Any embedded ending punct inside short heading => reject
        for p in CJK_PUNCT_END:
            if p in s:
                return False

        has_non_ascii = False
        all_ascii = True
        has_letter = False
        all_ascii_digits = True

        for ch in s:
            if ord(ch) > 0x7F:
                has_non_ascii = True
                all_ascii = False
                all_ascii_digits = False
                continue

            if not ch.isdigit():
                all_ascii_digits = False
            if ch.isalpha():
                has_letter = True

        if all_ascii_digits or all_ascii:
            return True
        if has_non_ascii:
            return True
        if all_ascii and has_letter:
            return True

    return False


def is_metadata_line(line: str) -> bool:
    """
    Port of C# IsMetadataLine().
    Caller should pass the probe (left indent removed).
    """
    if not line:
        return False

    s = line.strip()
    if not s:
        return False

    # Fast length gate
    if len(s) > 30:
        return False

    # Find the earliest separator among allowed ones, idx in (0..10)
    idx = -1
    for sep in METADATA_SEPARATORS:
        i = s.find(sep)
        if 0 < i <= 10 and (idx < 0 or i < idx):
            idx = i

    if idx < 0:
        return False

    key = s[:idx].strip()
    if key not in METADATA_KEYS:
        return False

    # Skip whitespace after separator
    n = len(s)
    j = idx + 1
    while j < n and s[j].isspace():
        j += 1
    if j >= n:
        return False

    # Reject dialog opener right after "Key: "
    return not is_dialog_opener(s[j])


# -------------------------------
# Sentence Boundary start
# -------------------------------

def ends_with_sentence_boundary(s: str) -> bool:
    """
    Level-2 normalized sentence boundary detection.

    Includes OCR artifacts (ASCII '.' / ':'), but does NOT treat a bare
    bracket closer as a sentence boundary (avoid false flush: "（亦作肥）").
    """
    if not s or not s.strip():
        return False

    last2 = last_two_non_whitespace_idx(s)
    if last2 is None:
        # < 2 non-whitespace chars; still may match strong end on single char
        last = last_non_whitespace(s)
        return (last is not None) and is_strong_sentence_end(last)

    (last_i, last), (prev_i, prev) = last2

    # 1) Strong sentence enders.
    if is_strong_sentence_end(last):
        return True

    # 2) OCR '.' / ':' at line end (mostly-CJK).
    if (last == "." or last == ":") and is_ocr_cjk_ascii_punct_at_line_end(s, last_i):
        return True

    # 3) Quote closers + Allowed postfix closer after strong end,
    #    plus OCR artifact `.“”` / `.」` / `.）`.
    if is_dialog_closer(last) or is_allowed_postfix_closer(last):
        if is_strong_sentence_end(prev):
            return True

        if prev == "." and is_ocr_cjk_ascii_punct_before_closers(s, prev_i):
            return True

    # 4) Full-width colon as a weak boundary (common: "他说：" then dialog next line)
    if is_colon_like(last) and is_mostly_cjk(s):
        return True

    # 5) Ellipsis as weak boundary.
    if ends_with_ellipsis(s):
        return True

    return False


def is_ocr_cjk_ascii_punct_at_line_end(s: str, punct_index: int) -> bool:
    """
    Strict OCR: punct itself is at end-of-line (only whitespace after it),
    and preceded by CJK in a mostly-CJK line.
    """
    if punct_index <= 0:
        return False
    if not is_at_line_end_ignoring_whitespace(s, punct_index):
        return False

    prev = nth_char(s, punct_index - 1)
    return is_cjk(prev) and is_mostly_cjk(s)


def is_ocr_cjk_ascii_punct_before_closers(s: str, punct_index: int) -> bool:
    """
    Relaxed OCR: after punct, allow only whitespace and closers (quote/bracket).
    Enables `“.”` / `.」` / `.）` to count as sentence boundary.
    """
    if punct_index <= 0:
        return False
    if not is_at_end_allowing_closers(s, punct_index):
        return False

    prev = nth_char(s, punct_index - 1)
    return is_cjk(prev) and is_mostly_cjk(s)


def nth_char(s: str, idx: int) -> str:
    # Rust: s.chars().nth(idx).unwrap_or('\0')
    if 0 <= idx < len(s):
        return s[idx]
    return "\0"


def is_at_line_end_ignoring_whitespace(s: str, index: int) -> bool:
    # Rust: s.chars().skip(index + 1).all(|c| c.is_whitespace())
    i = index + 1
    while i < len(s):
        if not s[i].isspace():
            return False
        i += 1
    return True


def is_at_end_allowing_closers(s: str, index: int) -> bool:
    # Rust: after punct, allow only whitespace and dialog/bracket closers
    i = index + 1
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if is_dialog_closer(ch) or is_bracket_closer(ch):
            i += 1
            continue
        return False
    return True


def last_two_non_whitespace_idx(s: str) -> Optional[Tuple[Tuple[int, str], Tuple[int, str]]]:
    """
    Returns ((last_i,last),(prev_i,prev)) in Python string indices.
    Equivalent role to Rust's last_two_non_whitespace_idx (byte indices there).
    """
    last: Optional[Tuple[int, str]] = None

    i = len(s) - 1
    while i >= 0:
        ch = s[i]
        if not ch.isspace():
            if last is None:
                last = (i, ch)
            else:
                return last, (i, ch)
        i -= 1

    return None


# -------------------------------
# Sentence Boundary end
# -------------------------------


# ------ Bracket Boundary start ------

def slice_inner_without_outer_pair(s: str) -> Optional[str]:
    """
    Returns the substring excluding the first and last character of `s`.
    Precondition: `s` is already trimmed and has at least 2 chars.
    """
    if len(s) < 2:
        return None
    return s[1:-1]


def is_bracket_type_balanced_str(s: str, open_ch: str) -> bool:
    close_ch = try_get_matching_closer(open_ch)
    if close_ch is None:
        # Same as Rust/C#: unrecognized opener treated as "balanced"
        return True

    depth = 0
    for ch in s:
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth < 0:
                return False

    return depth == 0


def ends_with_cjk_bracket_boundary(s: str) -> bool:
    """
    True if the string ends with a balanced CJK-style bracket boundary,
    e.g. （完）, 【番外】, 《後記》.
    """
    t = s.strip()
    if not t:
        return False

    # Need at least open+close
    if len(t) < 2:
        return False

    open_ch = t[0]

    # last non-whitespace char (t is stripped, so last char is correct)
    close_ch = t[-1]

    # 1) Must be one of our known pairs.
    if not is_matching_bracket(open_ch, close_ch):
        return False

    # Inner content (exclude outer pair)
    inner = slice_inner_without_outer_pair(t)
    if inner is None:
        return False
    inner = inner.strip()
    if not inner:
        return False

    # 2) Must be mostly CJK (reject "(test)", "[1.2]" etc.)
    if not is_mostly_cjk(inner):
        return False

    # ASCII bracket pairs suspicious → require at least one CJK inside
    if (open_ch == "(" or open_ch == "[") and (not contains_any_cjk_str(inner)):
        return False

    # 3) Ensure this bracket type is balanced inside the text
    return is_bracket_type_balanced_str(t, open_ch)


# ------ Bracket Boundary end ------


# =============================================================================
# Reflow core (public entry)
# =============================================================================

def reflow_cjk_paragraphs_core(
        text: str,
        *,
        add_pdf_page_header: bool,
        compact: bool,
) -> str:
    """
    Reflow extracted text into CJK-friendly paragraphs.

    Parameters
    ----------
    text:
        Extracted text (already Unicode).
    add_pdf_page_header:
        If True, page markers like "=== [Page 1/20] ===" are expected to exist and
        treated as hard paragraph boundaries.
    compact:
        If True, join segments with single newlines; otherwise join paragraphs
        with blank lines (double newlines).

    Returns
    -------
    str:
        Reflowed text.
    """
    if not text.strip():
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    segments: List[str] = []
    buffer = ""
    dialog_state = DialogState()

    for raw_line in lines:
        visual = raw_line.rstrip()
        # 1) Remove half-width indent but keep full-width indent
        stripped = strip_half_width_indent_keep_fullwidth(visual)
        # 2) Collapse style-layer repeats (per line)
        stripped = collapse_repeated_segments(stripped)
        # 3) Probe for detection (remove all indent, incl. full-width)
        probe = strip_all_left_indent_for_probe(stripped)

        # Divider line → ALWAYS force paragraph break
        if is_visual_divider_line(probe):
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            segments.append(probe)
            continue

        # Title / heading / metadata detection
        is_title_heading = bool(TITLE_HEADING_REGEX.search(probe))
        is_short_heading = is_heading_like(stripped)
        is_metadata = is_metadata_line(probe)

        buffer_has_unclosed_bracket = has_unclosed_bracket(buffer)
        dialog_unclosed = dialog_state.is_unclosed

        # 4) Empty line
        if not stripped:
            if (not add_pdf_page_header) and buffer:

                # NEW: If dialog or brackets are unclosed, blank line is treated as soft wrap.
                # Never flush mid-enclosure.
                if dialog_unclosed or buffer_has_unclosed_bracket:
                    continue

                # LIGHT rule: only flush on blank line if buffer ends with STRONG sentence end.
                last_ch = last_non_whitespace(buffer)
                ends_strong = (last_ch is not None) and is_strong_sentence_end(last_ch)

                if not ends_strong:
                    # Soft cross-page blank line: keep accumulating
                    continue

            # End paragraph → flush buffer (do not emit empty segments)
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            continue

        # Page markers like "=== [Page 1/20] ==="
        if stripped.startswith("=== ") and stripped.endswith("==="):
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            segments.append(stripped)
            continue

        # Strong headings (TitleHeadingRegex)
        if is_title_heading:
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            segments.append(stripped)
            continue

        # Metadata lines
        if is_metadata:
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            segments.append(stripped)
            continue

        # Weak heading-like (heuristic)
        if is_short_heading:
            is_all_cjk = is_all_cjk_ignoring_whitespace(stripped)
            current_looks_like_cont_marker = (
                    is_all_cjk
                    or ends_with_colon_like(stripped)
                    or ends_with_allowed_postfix_closer(stripped)
            )

            if not buffer:
                split_as_heading = True
            else:
                if buffer_has_unclosed_bracket:
                    split_as_heading = False
                else:
                    bt = buffer.rstrip()
                    if bt:
                        last = bt[-1]
                        if is_comma_like(last):
                            split_as_heading = False
                        elif current_looks_like_cont_marker and (not is_clause_or_end_punct(last)):
                            split_as_heading = False
                        else:
                            split_as_heading = True
                    else:
                        split_as_heading = True

            if split_as_heading:
                if buffer:
                    segments.append(buffer)
                    buffer = ""
                    dialog_state.reset()

                segments.append(stripped)
                continue

        # Final strong line punct ending check for line text
        if buffer and (not dialog_unclosed) and (not buffer_has_unclosed_bracket):
            last = last_non_whitespace(stripped)
            if (last is not None) and is_strong_sentence_end(last):
                # Merge current line into buffer, then flush immediately as a paragraph
                buffer += stripped
                segments.append(buffer)
                buffer = ""

                dialog_state.reset()
                dialog_state.update(stripped)
                continue

        # First line of a new paragraph
        if not buffer:
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        current_is_dialog_start = begins_with_dialog_opener(stripped)
        # If previous line ends with comma, do NOT flush even if new line starts dialog
        if current_is_dialog_start:
            trimmed_buffer = buffer.rstrip()

            if trimmed_buffer:
                last = trimmed_buffer[-1]

                # Rust: if !is_comma_like(ch) && !is_cjk_bmp(ch) { flush }
                if (not is_comma_like(last)) and (not is_cjk(last)):
                    segments.append(buffer)
                    buffer = ""
                    buffer += stripped

                    dialog_state.reset()
                    dialog_state.update(stripped)
                    continue
            else:
                # Buffer is whitespace-only → treat like empty and flush
                segments.append(buffer)
                buffer = ""
                buffer += stripped

                dialog_state.reset()
                dialog_state.update(stripped)
                continue

        # 🔸 9b) Dialog end line: ends with dialog closer.
        # Flush when the char before closer is strong end,
        # and bracket safety is satisfied (with a narrow typo override).
        last2 = last_two_non_whitespace(stripped)
        if last2 is not None:
            last_ch, prev_ch = last2

            if is_dialog_closer(last_ch):
                # Check punctuation right before the closer (e.g., “？” / “。”)
                punct_before_closer_is_strong = is_clause_or_end_punct(prev_ch)

                # Snapshot bracket safety BEFORE appending current line
                buffer_has_bracket_issue = buffer_has_unclosed_bracket
                line_has_bracket_issue = has_unclosed_bracket(stripped)

                # Append current line into buffer, then update dialog state
                buffer += stripped
                dialog_state.update(stripped)

                # Allow flush if:
                # - dialog is closed after this line
                # - punctuation before closer is a strong end
                # - and either:
                #   (a) buffer has no bracket issue, OR
                #   (b) buffer has bracket issue but this line itself is the culprit (OCR/typo)
                if (not dialog_unclosed) and punct_before_closer_is_strong and (
                        (not buffer_has_bracket_issue) or line_has_bracket_issue
                ):
                    segments.append(buffer)
                    buffer = ""
                    dialog_state.reset()

                continue

        # Colon + dialog continuation
        # if buffer.endswith(("：", ":")):
        #     after_indent = stripped.lstrip(" \u3000")
        #     if after_indent and after_indent[0] in DIALOG_OPENERS:
        #         buffer += stripped
        #         dialog_state.update(stripped)
        #         continue

        # 8a) Strong sentence boundary (handles 。！？, OCR . / :, “.”)
        if (not dialog_unclosed) and (not buffer_has_unclosed_bracket) and ends_with_sentence_boundary(
                buffer):
            segments.append(buffer)
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # 8b) Balanced CJK bracket boundary: （完）, 【番外】, 《後記》
        if (not dialog_unclosed) and ends_with_cjk_bracket_boundary(buffer):
            segments.append(buffer)
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # Ends with CJK punctuation → new paragraph (only if not inside unclosed dialog)
        # if buffer[-1] in CJK_PUNCT_END and not dialog_unclosed:
        #     segments.append(buffer)
        #     buffer = stripped
        #     dialog_state.reset()
        #     dialog_state.update(stripped)
        #     continue

        # Indentation → new paragraph (raw line based)
        # if _INDENT_RE.match(raw_line):
        #     segments.append(buffer)
        #     buffer = stripped
        #     dialog_state.reset()
        #     dialog_state.update(stripped)
        #     continue

        # Chapter-like endings
        if len(buffer) <= 12 and re.search(r"([章节部卷節])[】》〗〕〉」』）]*$", buffer):
            segments.append(buffer)
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # Default merge
        buffer += stripped
        dialog_state.update(stripped)

    if buffer:
        segments.append(buffer)

    return "\n".join(segments) if compact else "\n\n".join(segments)
