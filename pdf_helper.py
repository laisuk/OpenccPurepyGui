from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pymupdf

# =============================================================================
# Types
# =============================================================================

ProgressCallback = Callable[[int, int], None]  # (current_page, total_pages)
CancelCallback = Callable[[], bool]  # return True => cancel requested


# =============================================================================
# Extraction helpers (top)
# =============================================================================

def get_progress_block(total_pages: int) -> int:
    if total_pages <= 20:
        return 1
    if total_pages <= 100:
        return 3
    if total_pages <= 300:
        return 5
    return max(1, total_pages // 20)


def build_progress_bar(current: int, total: int, width: int = 10) -> str:
    if total <= 0:
        return "[" + "🟨" * width + "]"
    filled = current * width // total
    filled = max(0, min(width, filled))
    return "[" + "🟩" * filled + "🟨" * (width - filled) + "]"


# -----------------------------------------------
# Remove Invisible Chars from extracted PDF text
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
# Core PDF extraction (no Qt, reusable for batch)
# ---------------------------------------------------------------------------

def extract_pdf_text_core(
    filename: str,
    add_pdf_page_header: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    is_cancelled: Optional[CancelCallback] = None,
) -> str:
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
            if is_cancelled is not None and is_cancelled():
                break

            page = doc[i]
            text = page.get_text("text") or ""  # type: ignore

            if add_pdf_page_header:
                parts.append(f"\n\n=== [Page {i + 1}/{total}] ===\n\n")

            parts.append(text)

            current = i + 1
            if current % block == 0 or current == 1 or current == total:
                if on_progress is not None:
                    on_progress(current, total)

        return "".join(parts)

    finally:
        doc.close()


# =============================================================================
# Reflow-layer optional cleanup (kept outside extraction)
# =============================================================================

def collapse_consecutive_duplicate_lines(text: str) -> str:
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

CJK_PUNCT_END = (
    "。", "！", "？", "；", "：", "…", "—", "”", "」", "’", "』",
    "）", "】", "》", "〗", "〕", "〉", "］", "｝",
    ".", "!", "?", ")", ":"
)

OPEN_BRACKETS = "([{（【《〈｛"
CLOSE_BRACKETS = ")]}）】》〉｝"

TITLE_HEADING_REGEX = re.compile(
    r"^(?=.{0,50}$)"
    r".{0,10}?(前言|序章|终章|尾声|后记|番外.{0,15}?|尾聲|後記|第.{0,5}?([章节部卷節回][^分合]).{0,20}?)"
)

DIALOG_OPEN_TO_CLOSE = {
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "﹁": "﹂",
    "﹃": "﹄",
}
DIALOG_CLOSE_TO_OPEN = {v: k for k, v in DIALOG_OPEN_TO_CLOSE.items()}
DIALOG_OPENERS = tuple(DIALOG_OPEN_TO_CLOSE.keys())


def is_dialog_opener(ch: str) -> bool:
    return ch in DIALOG_OPENERS


METADATA_SEPARATORS = ("：", ":", "　")
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


# =============================================================================
# Low-level helpers (indent / CJK / box line / repeats)
# =============================================================================

def is_all_ascii(s: str) -> bool:
    for ch in s:
        if ord(ch) > 0x7F:
            return False
    return True


def is_cjk(ch: str) -> bool:
    c = ord(ch)
    if 0x3400 <= c <= 0x4DBF:
        return True
    if 0x4E00 <= c <= 0x9FFF:
        return True
    return 0xF900 <= c <= 0xFAFF


def is_all_cjk(s: str) -> bool:
    if not s:
        return False
    for ch in s:
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

        if "\u2500" <= ch <= "\u257F":
            continue

        if ch in ("-", "=", "_", "~", "·", "•", "*"):
            continue

        return False

    return total >= 3


IDEOGRAPHIC_SPACE = "\u3000"


def strip_half_width_indent_keep_fullwidth(s: str) -> str:
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
    return s.lstrip(" \t\r\n\u3000")


def collapse_repeated_segments(line: str) -> str:
    if not line:
        return line
    parts = line.strip().split()
    if not parts:
        return line
    parts = collapse_repeated_word_sequences(parts)
    parts = [collapse_repeated_token(tok) for tok in parts]
    return " ".join(parts)


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
# Reflow rule helpers (moved out from inner functions)
# =============================================================================

def is_dialog_start(line: str) -> bool:
    """
    Returns True if the line logically starts with a dialog opener,
    ignoring leading half/full-width spaces.
    """
    s = line.lstrip(" \u3000")
    return bool(s) and s[0] in DIALOG_OPENERS


def has_unclosed_bracket(s: str) -> bool:
    """
    True if we see any OPEN_BRACKETS but no CLOSE_BRACKETS.
    (Same intent as your C# HasUnclosedBracket)
    """
    if not s:
        return False

    has_open = False
    has_close = False

    for ch in s:
        if not has_open and ch in OPEN_BRACKETS:
            has_open = True
        if not has_close and ch in CLOSE_BRACKETS:
            has_close = True
        if has_open and has_close:
            break

    return has_open and not has_close


def is_heading_like(s: str) -> bool:
    """
    Heuristic for detecting heading-like lines (aligned with C# version).
    """
    if s is None:
        return False

    s = s.strip()
    if not s:
        return False

    if s.startswith("=== ") and s.endswith("==="):
        return False

    if any(ch in OPEN_BRACKETS for ch in s) and not any(ch in CLOSE_BRACKETS for ch in s):
        return False

    length = len(s)
    max_len = 18 if is_all_ascii(s) else 8
    last_ch = s[-1]

    # Short circuit for item title-like: "物品准备："
    if (last_ch == ":" or last_ch == "：") and length <= max_len and is_dialog_start(s[:-1]):
        return True

    if last_ch in CJK_PUNCT_END:
        return False

    if "，" in s or "," in s or "、" in s:
        return False

    if length <= max_len:
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
    Note: caller should pass the probe (left indent removed).
    """
    if not line or line.strip() == "":
        return False

    stripped_line = line.strip()

    if len(stripped_line) > 30:
        return False

    idx = min(
        (stripped_line.find(sep) for sep in METADATA_SEPARATORS if stripped_line.find(sep) >= 0),
        default=-1,
    )
    if idx <= 0 or idx > 10:
        return False

    key = stripped_line[:idx].strip()
    if key not in METADATA_KEYS:
        return False

    j = idx + 1
    while j < len(stripped_line) and stripped_line[j].isspace():
        j += 1
    if j >= len(stripped_line):
        return False

    if is_dialog_opener(stripped_line[j]):
        return False

    return True


# =============================================================================
# Reflow core (bottom)
# =============================================================================

def reflow_cjk_paragraphs_core(
    text: str,
    *,
    add_pdf_page_header: bool,
    compact: bool,
) -> str:
    if not text.strip():
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    segments: List[str] = []
    buffer = ""
    dialog_state = DialogState()

    for raw_line in lines:
        visual = raw_line.rstrip()

        # 2) Remove half-width indent but keep full-width indent
        stripped = strip_half_width_indent_keep_fullwidth(visual)

        # 3) Collapse style-layer repeats (per line)
        stripped = collapse_repeated_segments(stripped)

        # 4) Probe (no left indent at all, incl. full-width) for detection
        probe = strip_all_left_indent_for_probe(stripped)

        # 2.5) Visual divider line → ALWAYS force paragraph break
        if is_box_drawing_line(probe):
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

        # 1) Empty line
        if not stripped:
            if (not add_pdf_page_header) and buffer:
                last_char = buffer[-1]
                if last_char not in CJK_PUNCT_END:
                    continue

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

        # 3) Force TitleHeading
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

        # 3c) Weak heading-like (your C# port)
        if is_short_heading:
            all_cjk = True
            for ch in stripped:
                if ch.isspace():
                    continue
                if ord(ch) > 0x7F:
                    continue
                all_cjk = False
                break

            if buffer:
                buf_text = buffer

                if has_unclosed_bracket(buf_text):
                    pass
                else:
                    bt = buf_text.rstrip()
                    if bt:
                        last = bt[-1]
                        if last in ("，", ",", "、"):
                            pass
                        elif all_cjk and (last not in CJK_PUNCT_END):
                            pass
                        else:
                            segments.append(buf_text)
                            buffer = ""
                            dialog_state.reset()
                            segments.append(stripped)
                            continue
                    else:
                        segments.append(stripped)
                        continue
            else:
                segments.append(stripped)
                continue

        current_is_dialog_start = is_dialog_start(stripped)

        # 4) First line of a new paragraph
        if not buffer:
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        buffer_text = buffer

        if buffer_text:
            trimmed = buffer_text.rstrip()
            last = trimmed[-1] if trimmed else "\0"

            # NEW RULE: if previous line ends with comma, do NOT flush even if dialog starts
            if last in ("，", ","):
                pass
            elif current_is_dialog_start:
                segments.append(buffer_text)
                buffer = stripped
                dialog_state.reset()
                dialog_state.update(stripped)
                continue
        else:
            if current_is_dialog_start:
                buffer = stripped
                dialog_state.reset()
                dialog_state.update(stripped)
                continue

        # Colon + dialog continuation
        if buffer_text.endswith(("：", ":")):
            after_indent = stripped.lstrip(" \u3000")
            if after_indent and after_indent[0] in DIALOG_OPENERS:
                buffer += stripped
                dialog_state.update(stripped)
                continue

        # Ends with CJK punctuation → new paragraph (only if not inside unclosed dialog)
        if buffer_text[-1] in CJK_PUNCT_END and not dialog_state.is_unclosed:
            segments.append(buffer_text)
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # Indentation → new paragraph
        if re.match(r"^\s{2,}", raw_line):
            segments.append(buffer_text)
            buffer = stripped
            dialog_state.reset()
            dialog_state.update(stripped)
            continue

        # Chapter-like endings
        if len(buffer_text) <= 12 and re.search(r"([章节部卷節])[】》〗〕〉」』）]*$", buffer_text):
            segments.append(buffer_text)
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
