from __future__ import annotations

from pdf_module.cjk_text import (
    last_non_whitespace,
    last_two_non_whitespace,
    is_all_ascii,
    is_cjk,
    contains_any_cjk_str,
    is_mixed_cjk_ascii,
    is_mostly_cjk,
    is_all_cjk_ignoring_ws,
    is_all_cjk_no_ws,
)

from pdf_module.punct_sets import (
    CJK_PUNCT_END,
    DIALOG_OPEN_TO_CLOSE,
    DIALOG_CLOSE_TO_OPEN,

    is_clause_or_end_punct,
    is_dialog_opener,
    is_dialog_closer,
    begins_with_dialog_opener,

    is_bracket_opener,
    is_bracket_closer,
    is_matching_bracket,
    is_wrapped_by_matching_bracket,
    try_get_matching_closer,

    is_allowed_postfix_closer,
    ends_with_allowed_postfix_closer,

    is_strong_sentence_end,
    is_comma_like,
    contains_any_comma_like,
    is_colon_like,
    ends_with_colon_like,
    ends_with_ellipsis,
)

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
from typing import List, Optional, Sequence, Tuple

# =============================================================================
# Shared constants (CJK / dialog / metadata)
# =============================================================================

TITLE_HEADING_REGEX = re.compile(
    r"^(?!.*[,，])(?=.{0,50}$)"
    r".{0,10}?(前言|序章|终章|尾声|后记|番外.{0,15}?|尾聲|後記|第.{0,5}?([章节部卷節回][^分合的])|[卷章][一二三四五六七八九十](?:$|.{0,20}?))"
)

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

IDEOGRAPHIC_SPACE = "\u3000"
# Common indent regex (raw_line based)
_INDENT_RE = re.compile(r"^\s{2,}")


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
        # counts per opener
        self.counts = dict.fromkeys(DIALOG_OPEN_TO_CLOSE, 0)

    def reset(self) -> None:
        counts = self.counts
        for k in counts:
            counts[k] = 0

    def update(self, s: str) -> None:
        counts = self.counts
        open_to_close = DIALOG_OPEN_TO_CLOSE
        close_to_open = DIALOG_CLOSE_TO_OPEN

        for ch in s:
            if ch in open_to_close:
                counts[ch] += 1
            else:
                open_ch = close_to_open.get(ch)
                if open_ch is not None:
                    v = counts[open_ch]
                    if v > 0:
                        counts[open_ch] = v - 1

    def is_unclosed(self) -> bool:
        # Hot-path; avoid generator+any overhead
        for v in self.counts.values():
            if v > 0:
                return True
        return False


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
            if is_all_cjk_no_ws(body):
                return True

        # 2) Allowed postfix closer: ... () / ） and no comma-like anywhere
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

    # Hoist hot callables (outside loop, before `for raw_line in lines:`)
    append_seg = segments.append
    title_search = TITLE_HEADING_REGEX.search
    # chapter_search = re.compile(r"([章节部卷節])[】》〗〕〉」』）]*$").search  # compile once if possible

    is_unclosed = dialog_state.is_unclosed
    d_reset = dialog_state.reset
    d_update = dialog_state.update

    strip_half = strip_half_width_indent_keep_fullwidth
    collapse_rep = collapse_repeated_segments
    strip_probe = strip_all_left_indent_for_probe

    is_divider = is_visual_divider_line
    is_meta = is_metadata_line
    is_heading = is_heading_like

    for raw_line in lines:
        visual = raw_line.rstrip()
        stripped = strip_half(visual)
        stripped = collapse_rep(stripped)
        probe = strip_probe(stripped)

        # Divider line → ALWAYS force paragraph break
        if is_divider(probe):
            if buffer:
                append_seg(buffer)
                buffer = ""
                d_reset()
            append_seg(probe)
            continue

        # Title / heading / metadata detection
        is_title_heading = bool(title_search(probe))
        is_short_heading = is_heading(stripped)
        is_metadata = is_meta(probe)

        # Dialog state snapshot (bool!)
        dialog_unclosed = is_unclosed()

        # Buffer bracket snapshot (only meaningful if buffer exists)
        buffer_has_unclosed_bracket = has_unclosed_bracket(buffer) if buffer else False

        # 4) Empty line
        if not stripped:
            if (not add_pdf_page_header) and buffer:
                # If dialog or brackets are unclosed, blank line is treated as soft wrap.
                if dialog_unclosed or buffer_has_unclosed_bracket:
                    continue

                # LIGHT rule: only flush on blank line if buffer ends with STRONG sentence end.
                last_ch = last_non_whitespace(buffer)
                if (last_ch is None) or (not is_strong_sentence_end(last_ch)):
                    continue

            if buffer:
                append_seg(buffer)
                buffer = ""
                d_reset()
            continue

        # Page markers like "=== [Page 1/20] ==="
        if stripped.startswith("=== ") and stripped.endswith("==="):
            if buffer:
                append_seg(buffer)
                buffer = ""
                d_reset()
            append_seg(stripped)
            continue

        # Strong headings (TitleHeadingRegex)
        if is_title_heading:
            if buffer:
                append_seg(buffer)
                buffer = ""
                d_reset()
            append_seg(stripped)
            continue

        # Metadata lines
        if is_metadata:
            if buffer:
                append_seg(buffer)
                buffer = ""
                d_reset()
            append_seg(stripped)
            continue

        # Weak heading-like (heuristic)
        if is_short_heading:
            is_all_cjk = is_all_cjk_ignoring_ws(stripped)
            current_looks_like_cont_marker = (
                    is_all_cjk
                    or ends_with_colon_like(stripped)
                    or ends_with_allowed_postfix_closer(stripped)
            )

            if not buffer:
                split_as_heading = True
            elif buffer_has_unclosed_bracket:
                split_as_heading = False
            else:
                bt = buffer.rstrip()
                if not bt:
                    split_as_heading = True
                else:
                    last = bt[-1]
                    if is_comma_like(last):
                        split_as_heading = False
                    elif current_looks_like_cont_marker and (not is_clause_or_end_punct(last)):
                        split_as_heading = False
                    else:
                        split_as_heading = True

            if split_as_heading:
                if buffer:
                    append_seg(buffer)
                    buffer = ""
                    d_reset()
                append_seg(stripped)
                continue

        # Final strong line punct ending check for current line text
        if buffer and (not dialog_unclosed) and (not buffer_has_unclosed_bracket):
            last = last_non_whitespace(stripped)
            if (last is not None) and is_strong_sentence_end(last):
                buffer += stripped
                append_seg(buffer)
                buffer = ""
                d_reset()
                d_update(stripped)
                continue

        # First line of a new paragraph
        if not buffer:
            buffer = stripped
            d_reset()
            d_update(stripped)
            continue

        current_is_dialog_start = begins_with_dialog_opener(stripped)

        # If previous line ends with comma, do NOT flush even if new line starts dialog
        if current_is_dialog_start:
            tb = buffer.rstrip()
            if tb:
                last = tb[-1]
                if (not is_comma_like(last)) and (not is_cjk(last)):  # <-- FIX: is_cjk_bmp
                    append_seg(buffer)
                    buffer = stripped
                    d_reset()
                    d_update(stripped)
                    continue
            else:
                # Buffer is whitespace-only → treat like empty and flush
                append_seg(buffer)
                buffer = stripped
                d_reset()
                d_update(stripped)
                continue

        # 🔸 9b) Dialog end line: ends with dialog closer.
        last2 = last_two_non_whitespace(stripped)
        if last2 is not None:
            last_ch, prev_ch = last2
            if is_dialog_closer(last_ch):
                punct_before_closer_is_strong = is_clause_or_end_punct(prev_ch)

                # Snapshot bracket safety BEFORE appending current line
                buffer_has_bracket_issue = buffer_has_unclosed_bracket
                line_has_bracket_issue = has_unclosed_bracket(stripped)

                buffer += stripped
                d_update(stripped)

                # dialog_unclosed might have changed after update; re-check like Rust
                if (not is_unclosed()) and punct_before_closer_is_strong and (
                        (not buffer_has_bracket_issue) or line_has_bracket_issue
                ):
                    append_seg(buffer)
                    buffer = ""
                    d_reset()

                continue

        # 8a) Strong sentence boundary (handles 。！？, OCR . / :, “.”)
        if (not dialog_unclosed) and (not buffer_has_unclosed_bracket) and ends_with_sentence_boundary(buffer):
            append_seg(buffer)
            buffer = stripped
            d_reset()
            d_update(stripped)
            continue

        # 8b) Balanced CJK bracket boundary: （完）, 【番外】, 《後記》
        if (not dialog_unclosed) and ends_with_cjk_bracket_boundary(buffer):
            append_seg(buffer)
            buffer = stripped
            d_reset()
            d_update(stripped)
            continue

        # Chapter-like endings
        # if len(buffer) <= 12 and chapter_search(buffer):
        #     append_seg(buffer)
        #     buffer = stripped
        #     d_reset()
        #     d_update(stripped)
        #     continue

        # Default merge
        buffer += stripped
        d_update(stripped)

    if buffer:
        # segments.append(buffer)
        append_seg(buffer)

    return "\n".join(segments) if compact else "\n\n".join(segments)
