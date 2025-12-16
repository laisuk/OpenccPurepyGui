from __future__ import annotations

import re
from pathlib import Path
from typing import Callable
from typing import List, Sequence, Optional

import pymupdf

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int], None]  # (current_page, total_pages)
CancelCallback = Callable[[], bool]  # return True => cancel requested

# ---------------------------------------------------------------------------
# CJK punctuation / title rules (ported from MainWindow)
# ---------------------------------------------------------------------------

CJK_PUNCT_END = (
    '。', '！', '？', '；', '：', '…', '—', '”', '」', '’', '』',
    '）', '】', '》', '〗', '〕', '〉', '］', '｝',
    '.', '!', '?', ')', ":"
)

# General structural brackets (non-dialog)
OPEN_BRACKETS = "([{（【《〈｛"
CLOSE_BRACKETS = ")]}）】》〉｝"

# Title heading detection (same semantics as your C# TitleHeadingRegex)
TITLE_HEADING_REGEX = re.compile(
    r"^(?=.{0,50}$)"
    r".{0,10}?(前言|序章|终章|尾声|后记|番外.{0,15}?|尾聲|後記|第.{0,5}?([章节部卷節回][^分合]).{0,20}?)"
)


# ---------------------------------------------------------------------------
# Helper: collapse repeated segments (style-layer noise)
# ---------------------------------------------------------------------------

# def collapse_repeated_segments(line: str) -> str:
#     """
#     Split a line into tokens by whitespace and collapse each token separately.
#     Equivalent to the C# CollapseRepeatedSegments.
#     """
#     if not line:
#         return line
#
#     # Split on whitespace into manageable parts
#     parts = re.split(r"[ \t]+", line.strip())
#     if not parts:
#         return line
#
#     collapsed_parts = [collapse_repeated_token(tok) for tok in parts]
#
#     # Rejoin using a single space (same behavior as C#)
#     return " ".join(collapsed_parts)
#
#
# def collapse_repeated_token(token: str) -> str:
#     """
#     Collapse repeated subunit patterns inside a token.
#
#     Python port of the C# CollapseRepeatedToken:
#     - Ignore very short (<4) or very long (>200) tokens.
#     - Try repeating unit lengths between 2 and 20.
#     - Detect tokens made entirely of repeated units.
#     - Collapse to a single unit.
#     """
#     length = len(token)
#
#     # Very short or very large tokens are not treated as repeated patterns
#     if length < 4 or length > 200:
#         return token
#
#     # Try unit sizes between 2 and 20 chars
#     for unit_len in range(2, 21):
#         if unit_len > length // 2:
#             break
#
#         if length % unit_len != 0:
#             continue
#
#         unit = token[:unit_len]
#         # Check if token is made entirely of repeated unit
#         repeat_count = length // unit_len
#
#         if unit * repeat_count == token:
#             return unit  # collapse
#
#     return token


def collapse_repeated_segments(line: str) -> str:
    """
    Style-layer repeat collapse for PDF headings / title lines.

    Conceptually similar to the regex:

        (.{4,10}?)\1{2,3}

    but implemented in a token- and phrase-aware way so that CJK
    headings like:

        "背负着一切的麒麟 背负着一切的麒麟 背负着一切的麒麟 背负着一切的麒麟"

    collapse cleanly to a single phrase, while avoiding damage to
    normal text (e.g. '哈哈哈哈哈哈').
    """
    if not line:
        return line

    # Split on whitespace into tokens; re-join with single spaces.
    parts = line.strip().split()
    if not parts:
        return line

    # 1) Phrase-level collapse: repeated word sequences inside the line.
    parts = collapse_repeated_word_sequences(parts)

    # 2) Token-level collapse: repeated substring patterns inside a token.
    parts = [collapse_repeated_token(tok) for tok in parts]

    return " ".join(parts)


def collapse_repeated_word_sequences(parts: Sequence[str]) -> List[str]:
    """
    Collapse repeated *phrases* (sequences of tokens) within a single line.

    Example:
        ["背负着一切的麒麟", "背负着一切的麒麟",
         "背负着一切的麒麟", "背负着一切的麒麟"]

    becomes:
        ["背负着一切的麒麟"]

    The algorithm:
      - Scan for phrases of length 1..max_phrase_len tokens.
      - If the same phrase occurs consecutively at least min_repeats times,
        collapse all repeats into a single copy.
      - Prefix and suffix tokens are preserved.

    This is intentionally conservative and only fires on very obvious
    layout / styling repetition.
    """
    min_repeats = 3  # require at least 3 consecutive repeats
    max_phrase_len = 8  # typical heading / subtitle phrases are short

    n = len(parts)
    if n < min_repeats:
        return list(parts)

    # Scan from left to right for any repeating phrase.
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
                # Build collapsed list: [prefix] + [one phrase] + [tail]
                result: List[str] = []

                # prefix
                result.extend(parts[:start])

                # one copy of the repeated phrase
                result.extend(parts[start:start + phrase_len])

                # tail
                tail_start = start + count * phrase_len
                result.extend(parts[tail_start:])

                return result

    return list(parts)


def collapse_repeated_token(token: Optional[str]) -> Optional[str]:
    """
    Collapse a single token if it consists entirely of a repeated substring.

    Only applies when:
      - token length is between 4 and 200
      - base unit length is between 4 and 10
      - the token is exactly N consecutive repeats of that unit, with N >= 3

    Examples:
        "abcdabcdabcd" → "abcd"
        "第一季大结局第一季大结局第一季大结局" → "第一季大结局"

    Very short units (len < 4) are ignored on purpose, so patterns like
    "哈哈哈哈哈哈" are left intact.
    """
    if token is None:
        return None

    length = len(token)
    if length < 4 or length > 200:
        return token

    # Require at least 3 repeats (so unit_len <= length // 3).
    for unit_len in range(4, 11):  # 4..10 inclusive
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


# -----------------------------------------------
# Remove Invisible Chars rom extracted PDF text
# -----------------------------------------------

INVISIBLE_CHARS = (
    "\u200b",  # ZERO WIDTH SPACE
    "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "\u200e",  # LEFT-TO-RIGHT MARK
    "\u200f",  # RIGHT-TO-LEFT MARK
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
)


def sanitize_invisible(text: str) -> str:
    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, "")
    return text


# ---------------------------------------------------------------------------
# Progress block (ported from your worker / C#)
# ---------------------------------------------------------------------------

def get_progress_block(total_pages: int) -> int:
    """
    Adaptive progress update interval (port of C# GetProgressBlock).

    Behavior:
    - <= 20 pages  → update every page
    - <= 100 pages → update every 3 pages
    - <= 300 pages → update every 5 pages
    - > 300 pages  → update at ~5% intervals
    """
    if total_pages <= 20:
        return 1  # every page
    if total_pages <= 100:
        return 3  # every 3 pages
    if total_pages <= 300:
        return 5  # every 5 pages
    # large PDFs: ~5% intervals
    return max(1, total_pages // 20)


def build_progress_bar(current: int, total: int, width: int = 10) -> str:
    """
    Emoji-based progress bar, stable full-width squares.

    Example:
        [🟩🟩🟩🟩🟨🟨🟨🟨🟨🟨]
    """
    if total <= 0:
        return "[" + "🟨" * width + "]"

    filled = current * width // total
    filled = max(0, min(width, filled))

    # return "[" + "█" * filled + "░" * (width - filled) + "]"
    # return "[" + "🟩" * filled + "⬜" * (width - filled) + "]"
    return "[" + "🟩" * filled + "🟨" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Core PDF extraction (no Qt, reusable for batch)
# ---------------------------------------------------------------------------

def extract_pdf_text_core(
        filename: str,
        add_pdf_page_header: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCallback] = None,
) -> str:
    """
    Core PDF text extraction using pymupdf (GUI-free and batch-safe).

    Parameters
    ----------
    filename : str
        Path of the PDF.
    add_pdf_page_header : bool
        Whether to insert '=== [Page x/y] ===' markers.
    on_progress : Callable[[int,int],None] | None
        Optional callback for progress updates.
    is_cancelled : Callable[[],bool] | None
        Optional cancel predicate (return True if cancel is requested).

    Returns
    -------
    str
        Extracted full or partial text.
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        doc = pymupdf.open(str(path))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load PDF: {path} ({e})") from e

    try:
        total = doc.page_count
        if total <= 0:
            return ""

        parts: List[str] = []
        block = get_progress_block(total)

        for i in range(total):
            # Check cancel
            if is_cancelled is not None and is_cancelled():
                break

            # Load & extract
            page = doc[i]  # same as doc.load_page(i)
            text = page.get_text("text") or ""  # type: ignore

            # Optional header
            if add_pdf_page_header:
                parts.append(f"\n\n=== [Page {i + 1}/{total}] ===\n\n")

            parts.append(text)

            # Progress callback
            current = i + 1
            if (
                    current % block == 0
                    or current == 1
                    or current == total
            ):
                if on_progress is not None:
                    on_progress(current, total)

        return "".join(parts)

    finally:
        doc.close()


def collapse_consecutive_duplicate_lines(text: str) -> str:
    out = []
    prev = None

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


# ---------------------------------------------------------------------------
# CJK punctuation + title regex assumed to already exist in your module:
#   - CJK_PUNCT_END
#   - TITLE_HEADING_REGEX
#   - collapse_repeated_segments()
# ---------------------------------------------------------------------------

# Dialog brackets (Simplified / Traditional / JP-style)
DIALOG_OPEN_TO_CLOSE = {
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "﹁": "﹂",  # U+FE41 → U+FE42
    "﹃": "﹄",  # U+FE43 → U+FE44
}

DIALOG_CLOSE_TO_OPEN = {v: k for k, v in DIALOG_OPEN_TO_CLOSE.items()}
DIALOG_OPENERS = tuple(DIALOG_OPEN_TO_CLOSE.keys())


def is_dialog_opener(ch: str) -> bool:
    return ch in DIALOG_OPENERS


# -------------------------------------------------------------
# Metadata detection (port from C#)
# -------------------------------------------------------------

# Metadata separators
METADATA_SEPARATORS = ("：", ":", "　")  # full-width colon, ascii colon, ideographic space

# Metadata keys (exact match, same as C#)
METADATA_KEYS = {
    # ===== 1. Title / Author / Publishing =====
    "書名", "书名",
    "作者",
    "譯者", "译者",
    "校訂", "校订",
    "出版社",
    "出版時間", "出版时间",
    "出版日期",

    # ===== 2. Copyright / License =====
    "版權", "版权",
    "版權頁", "版权页",
    "版權信息", "版权信息",

    # ===== 3. Editor / Pricing =====
    "責任編輯", "责任编辑",
    "編輯", "编辑",
    "責編", "责编",
    "定價", "定价",

    # ===== 4. Descriptions / Forewords =====
    "前言",
    "序章",
    "終章", "终章",
    "尾聲", "尾声",
    "後記", "后记",

    # ===== 5. Digital Publishing =====
    "品牌方",
    "出品方",
    "授權方", "授权方",
    "電子版權", "数字版权",
    "掃描", "扫描",
    "OCR",

    # ===== 6. CIP =====
    "CIP",
    "在版編目", "在版编目",
    "分類號", "分类号",
    "主題詞", "主题词",

    # ===== 7. Publishing Cycle =====
    "發行日", "发行日",
    "初版",

    # ===== 8. Common keys =====
    "ISBN",
}


class DialogState:
    """
    Tracks unmatched dialog brackets for the *current paragraph buffer*.

    We update it incrementally as we append text to the buffer, so we never
    re-scan the whole buffer.
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


def is_all_ascii(s: str) -> bool:
    for ch in s:
        if ord(ch) > 0x7F:
            return False
    return True


def is_cjk(ch: str) -> bool:
    c = ord(ch)

    # CJK Unified Ideographs + Extension A
    if 0x3400 <= c <= 0x4DBF:
        return True
    if 0x4E00 <= c <= 0x9FFF:
        return True

    # Compatibility Ideographs
    return 0xF900 <= c <= 0xFAFF


def is_all_cjk(s: str) -> bool:
    if not s:
        return False

    for ch in s:
        # treat common full-width space as not CJK heading content
        if ch.isspace():
            return False

        if not is_cjk(ch):
            return False

    return True


def is_box_drawing_line(s: str) -> bool:
    if not s or s.isspace():
        return False

    total = 0

    for ch in s:
        if ch.isspace():
            continue

        total += 1

        # Unicode box drawing block (U+2500–U+257F)
        if '\u2500' <= ch <= '\u257F':
            continue

        # ASCII / bullet-style visual separators
        if ch in ('-', '=', '_', '~', '·', '•', '*'):
            continue

        return False

    return total >= 3


IDEOGRAPHIC_SPACE = "\u3000"  # full-width indent


def strip_half_width_indent_keep_fullwidth(s: str) -> str:
    """
    Remove left indentation made of ASCII/half-width whitespace,
    but keep full-width indent (U+3000) intact.
    """
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # Keep full-width indent
        if ch == IDEOGRAPHIC_SPACE:
            break

        # Strip ASCII/half-width whitespace (space, tab, etc.)
        # Note: '\u3000'.isspace() is True, but we handled it above.
        if ch.isspace() and ord(ch) <= 0x7F:
            i += 1
            continue

        break

    return s[i:]


def strip_all_left_indent_for_probe(s: str) -> str:
    # remove ASCII whitespace + ideographic space U+3000
    return s.lstrip(" \t\r\n\u3000")


def reflow_cjk_paragraphs_core(
        text: str,
        *,
        add_pdf_page_header: bool,
        compact: bool,
) -> str:
    """
    Reflows CJK text extracted from PDFs by merging artificial line breaks
    while preserving intentional paragraph / heading / dialog boundaries.
    """
    if not text.strip():
        return text

    # Normalize line endings for cross-platform stability
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    segments: List[str] = []
    buffer = ""
    dialog_state = DialogState()  # track dialog for current buffer

    def is_dialog_start(s: str) -> bool:
        """
        Returns True if the line logically starts with a dialog opener,
        ignoring leading half/full-width spaces.
        """
        s = s.lstrip(" \u3000")
        return bool(s) and s[0] in DIALOG_OPENERS

    def has_unclosed_bracket(s: str) -> bool:
        """
        Check whether the string contains any opening bracket
        without a corresponding closing bracket.
        """
        if not s:
            return False

        has_open = False
        has_close = False

        for char in s:
            if not has_open and char in OPEN_BRACKETS:
                has_open = True
            if not has_close and char in CLOSE_BRACKETS:
                has_close = True

            if has_open and has_close:
                break

        return has_open and not has_close

    def is_heading_like(s: str) -> bool:
        """
        Heuristic for detecting heading-like or emphasis lines in CJK text.

        Rules (aligned with C# version):
          - Keep page markers intact (=== Page ===)
          - If *ends* with CJK end punctuation → NOT heading
          - Reject headings with unclosed brackets
          - Reject any short line containing comma-like punctuation: "，", ",", "、"
          - For very short lines (≤ 10 chars):
              * If line contains ANY CJK punctuation → NOT heading
              * Rule C: pure ASCII digits → heading
              * Rule A: has any non-ASCII char → heading
              * Rule B: pure ASCII with at least one letter → heading
        """
        if s is None:
            return False

        s = s.strip()
        if not s:
            return False

        # Keep page markers intact
        if s.startswith("=== ") and s.endswith("==="):
            return False

        # Reject headings with unclosed brackets (simplified HasUnclosedBracket)
        if any(char in OPEN_BRACKETS for char in s) and not any(char in CLOSE_BRACKETS for char in s):
            return False

        length = len(s)
        max_len = 18 if is_all_ascii(s) else 8
        last_ch = s[-1]

        # Short circuit for item title-like: "物品准备："
        if (last_ch == ':' or last_ch == '：') and length <= max_len and is_dialog_start(s[:-1]):
            return True

        # If *ends* with CJK end punctuation → not heading
        if last_ch in CJK_PUNCT_END:
            return False

        # NEW: reject any line containing comma-like punctuation
        # Short headings almost never contain commas.
        if '，' in s or ',' in s or '、' in s:
            return False

        # Match C#: only apply the short-line heuristic to len <= 10
        if length <= max_len:
            # NEW: short line containing ANY CJK punctuation → NOT heading
            for p in CJK_PUNCT_END:
                if p in s:
                    return False

            has_non_ascii = False
            all_ascii = True
            has_letter = False
            all_ascii_digits = True

            for char in s:
                if ord(char) > 0x7F:
                    has_non_ascii = True
                    all_ascii = False
                    all_ascii_digits = False
                    continue

                if not char.isdigit():
                    all_ascii_digits = False

                if char.isalpha():
                    has_letter = True

            # Rule C: pure ASCII digits → heading (e.g. "1", "007")
            if all_ascii_digits or all_ascii:
                return True

            # Rule A: short line with any non-ASCII (CJK/mixed) → heading
            if has_non_ascii:
                return True

            # Rule B: short pure ASCII line with at least one letter → heading
            if all_ascii and has_letter:
                return True

        return False

    # ----- Metadata -----

    def is_metadata_line(line: str) -> bool:
        """Port of C# IsMetadataLine()"""

        if not line or line.strip() == "":
            return False

        stripped_line = line.strip()

        # A) length limit
        if len(stripped_line) > 30:
            return False

        # B) find first separator (equivalent to IndexOfAny)
        idx = min((stripped_line.find(sep) for sep in METADATA_SEPARATORS if stripped_line.find(sep) >= 0), default=-1)
        if idx <= 0 or idx > 10:
            return False

        # C) extract key
        key = stripped_line[:idx].strip()
        if key not in METADATA_KEYS:
            return False

        # D) find next non-space char after separator
        j = idx + 1
        while j < len(stripped_line) and stripped_line[j].isspace():
            j += 1

        if j >= len(stripped_line):
            return False

        # E) must NOT start with a dialog opener
        if is_dialog_opener(stripped_line[j]):
            return False

        return True

    for raw_line in lines:
        # 1) Visual form: trim right-side whitespace only
        stripped = raw_line.rstrip()

        # 2) Remove half-width indent but keep full-width indent (for output/layout)
        stripped = strip_half_width_indent_keep_fullwidth(stripped)

        # --- PROBE (no left indent at all) for structural detection ---
        probe = strip_all_left_indent_for_probe(stripped)

        # 2.5) Visual divider line → ALWAYS force paragraph break
        if is_box_drawing_line(probe):
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            # keep divider as its own segment (optional; you can also drop it)
            segments.append(probe if probe else stripped)
            continue

        # 3) Collapse style-layer repeated segments (per line)
        stripped = collapse_repeated_segments(stripped)

        # logical probe for title detection (no left indent)
        stripped_left = stripped.lstrip()

        # Title detection (e.g. 前言 / 第X章 / 番外 ...)
        is_title_heading = bool(TITLE_HEADING_REGEX.search(stripped_left))
        is_short_heading = is_heading_like(stripped)
        is_metadata = is_metadata_line(stripped)

        # 1) Empty line
        if not stripped:
            if (not add_pdf_page_header) and buffer:
                last_char = buffer[-1]
                # Page-break-like blank line without ending punctuation → skip
                if last_char not in CJK_PUNCT_END:
                    continue

            # End of a paragraph → flush buffer, do NOT add an empty segment
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            continue

        # 2) Page markers like "=== [Page 1/20] ==="
        if stripped.startswith("=== ") and stripped.endswith("==="):
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            segments.append(stripped)
            continue

        # 3) 強制 TitleHeading
        if is_title_heading:
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()
            segments.append(stripped)
            continue

        # 3b) Metadata
        if is_metadata:
            if buffer:
                segments.append(buffer)
                buffer = ""
                dialog_state.reset()

            segments.append(stripped)
            continue

        # # 3c) 弱 heading-like
        # if is_short_heading:
        #     if buffer:
        #         bt = buffer.rstrip()
        #         if bt and bt[-1] in ("，", ","):
        #             # 逗號結尾 → 視作續句
        #             pass
        #         else:
        #             segments.append(buffer)
        #             buffer = stripped
        #             dialog_state.reset()
        #             dialog_state.update(stripped)
        #             continue
        #     else:
        #         # 無前文 → 直接當 heading
        #         segments.append(stripped)
        #         continue
        # 3c) Weak heading-like:
        #     Only takes effect when the “previous paragraph is safe”
        #     AND “the previous paragraph’s ending looks like a sentence boundary”.
        if is_short_heading:
            # Determine whether the current line is “all CJK” (ignoring whitespace)
            all_cjk = True
            for ch in stripped:
                if ch.isspace():
                    continue
                if ord(ch) > 0x7F:
                    continue
                all_cjk = False
                break

            if buffer:
                buf_text = buffer  # keep original (don't rstrip yet, we may want exact)
                # 🔐 1) If previous paragraph has unclosed brackets / book-title marks,
                #        it must be a continuation line and must NOT be treated as a heading.
                if has_unclosed_bracket(buf_text):
                    pass  # fall through → treat as normal line (merge logic below will handle)
                else:
                    bt = buf_text.rstrip()
                    if bt:
                        last = bt[-1]

                        # 🔸 2) Previous line ends with a comma → continuation, not heading.
                        if last in ("，", ",", "、"):
                            pass  # fall through
                        # 🔸 3) For “all-CJK short heading-like” lines:
                        #        if previous line does NOT end with a CJK sentence-ending punct,
                        #        treat as continuation (do not split).
                        elif all_cjk and (last not in CJK_PUNCT_END):
                            pass  # fall through
                        else:
                            # ✅ True heading-like:
                            # flush previous paragraph, then add heading as independent segment
                            segments.append(buf_text)
                            buffer = ""
                            dialog_state.reset()
                            segments.append(stripped)
                            continue
                    else:
                        # Buffer exists but is whitespace only → treat directly as a heading.
                        segments.append(stripped)
                        continue
            else:
                # Buffer is empty → allow a short heading to stand alone.
                segments.append(stripped)
                continue

        current_is_dialog_start = is_dialog_start(stripped)

        # 4) First line of a new paragraph
        if not buffer:
            # first line in buffer → start new paragraph
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # We already have some text in buffer
        buffer_text = buffer

        if buffer_text:
            trimmed = buffer_text.rstrip()
            last = trimmed[-1] if trimmed else "\0"

            # NEW RULE: if previous line ends with comma "，" or ","
            # do NOT flush even if this line starts a dialog.
            if last in ("，", ","):
                # fall through → treat as continuation
                pass
            elif current_is_dialog_start:
                # Dialog start and safe to flush previous paragraph
                segments.append(buffer_text)
                buffer = stripped
                dialog_state.reset()
                dialog_state.update(stripped)
                continue
        else:
            # buffer is logically empty (very rare here, but keep parity with C#)
            if current_is_dialog_start:
                buffer = stripped
                dialog_state.reset()
                dialog_state.update(stripped)
                continue

        # Colon + dialog continuation:
        # e.g. "她写了一行字：" + "「如果连自己都不相信……」"
        if buffer_text.endswith(("：", ":")):
            after_indent = stripped.lstrip(" \u3000")
            if after_indent and after_indent[0] in DIALOG_OPENERS:
                buffer += stripped
                dialog_state.update(stripped)
                continue

        # 5) Ends with CJK punctuation → new paragraph,
        #    but only if we are NOT inside an unclosed dialog.
        if buffer_text[-1] in CJK_PUNCT_END and not dialog_state.is_unclosed:
            segments.append(buffer_text)
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # 6) (old "previous is heading-like" rule REMOVED)
        #    Now all heading-like detection is done on the *current* line
        #    above, via: is_title_heading or is_heading_like(stripped).

        # 7) Indentation → new paragraph
        if re.match(r"^\s{2,}", raw_line):
            segments.append(buffer_text)
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # 8) Chapter-like endings: 章 / 节 / 部 / 卷 (with possible trailing brackets)
        if len(buffer_text) <= 12 and re.search(
                r"([章节部卷節])[】》〗〕〉」』）]*$", buffer_text
        ):
            segments.append(buffer_text)
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # 9) Default merge (soft line break)
        buffer += stripped
        dialog_state.update(stripped)

    # Flush last buffer
    if buffer:
        segments.append(buffer)

    # Formatting:
    return "\n".join(segments) if compact else "\n\n".join(segments)
