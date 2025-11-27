from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, List
import re

import pymupdf

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int], None]        # (current_page, total_pages)
CancelCallback = Callable[[], bool]                 # return True => cancel requested


# ---------------------------------------------------------------------------
# CJK punctuation / title rules (ported from MainWindow)
# ---------------------------------------------------------------------------

CJK_PUNCT_END = (
    '。', '！', '？', '；', '：', '…', '—', '”', '」', '’', '』',
    '）', '】', '》', '〗', '〕', '〉', '］', '｝', '章', '节', '部', '卷', '節'
)

# Title heading detection (same semantics as your C# TitleHeadingRegex)
TITLE_HEADING_REGEX = re.compile(
    r"^(?=.{0,60}$)"
    r"(前言|序章|终章|尾声|后记|番外|尾聲|後記|第.{0,10}?([章节部卷節]))"
)


# ---------------------------------------------------------------------------
# Helper: collapse repeated segments (style-layer noise)
# ---------------------------------------------------------------------------

def collapse_repeated_segments(line: str) -> str:
    """
    Split a line into tokens by whitespace and collapse each token separately.
    Equivalent to the C# CollapseRepeatedSegments.
    """
    if not line:
        return line

    # Split on whitespace into manageable parts
    parts = re.split(r"[ \t]+", line.strip())
    if not parts:
        return line

    collapsed_parts = [collapse_repeated_token(tok) for tok in parts]

    # Rejoin using a single space (same behavior as C#)
    return " ".join(collapsed_parts)


def collapse_repeated_token(token: str) -> str:
    """
    Collapse repeated subunit patterns inside a token.

    Python port of the C# CollapseRepeatedToken:
    - Ignore very short (<4) or very long (>200) tokens.
    - Try repeating unit lengths between 2 and 20.
    - Detect tokens made entirely of repeated units.
    - Collapse to a single unit.
    """
    length = len(token)

    # Very short or very large tokens are not treated as repeated patterns
    if length < 4 or length > 200:
        return token

    # Try unit sizes between 2 and 20 chars
    for unit_len in range(2, 21):
        if unit_len > length // 2:
            break

        if length % unit_len != 0:
            continue

        unit = token[:unit_len]
        # Check if token is made entirely of repeated unit
        repeat_count = length // unit_len

        if unit * repeat_count == token:
            return unit  # collapse

    return token


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

def build_progress_bar(current: int, total: int, width: int = 20) -> str:
    """
    Simple text progress bar like [██████░░░░░░░░░░].
    """
    if total <= 0:
        return "[" + "░" * width + "]"
    ratio = current / total
    filled = int(ratio * width)
    filled = max(0, min(width, filled))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


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
            text = page.get_text("text") or "" # type: ignore

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


# ---------------------------------------------------------------------------
# Core CJK reflow (ported from MainWindow.reflow_cjk_paragraphs)
# ---------------------------------------------------------------------------

def reflow_cjk_paragraphs_core(
    text: str,
    *,
    add_pdf_page_header: bool,
    compact: bool,
) -> str:
    """
    Reflows CJK text extracted from PDFs by merging artificial line breaks
    while preserving intentional paragraph / heading boundaries.

    Parameters
    ----------
    text : str
        Raw text (usually extracted from PDF) to be reflowed.
    add_pdf_page_header : bool
        If False, try to skip page-break-like blank lines that are not
        preceded by CJK punctuation (layout gaps between pages).
        If True, keep those gaps.
    compact : bool
        If True, join paragraphs with a single newline ("p1\\np2\\np3").
        If False, join with blank lines ("p1\\n\\np2\\n\\np3").

    Returns
    -------
    str
        Reflowed text.
    """
    if not text.strip():
        return text

    # Normalize line endings for cross-platform stability
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    segments: List[str] = []
    buffer = ""

    def is_heading_like(s: str) -> bool:
        """Heuristic: short, mostly CJK, no punctuation, not page marker."""
        s = s.strip()
        if not s:
            return False

        # keep page markers intact
        if s.startswith("=== ") and s.endswith("==="):
            return False

        # if ends with CJK punctuation, it's not a heading
        if any(ch in CJK_PUNCT_END for ch in s):
            return False

        # short + mostly CJK = heading/title
        if len(s) <= 8 and any(ord(ch) > 0x7F for ch in s):
            return True

        return False

    for raw_line in lines:
        stripped = raw_line.rstrip()
        stripped_left = stripped.lstrip()

        # Title detection (e.g. 前言 / 第X章 / 番外 ...)
        is_title_heading = bool(TITLE_HEADING_REGEX.search(stripped_left))

        # Collapse style-layer repeated titles
        if is_title_heading:
            stripped = collapse_repeated_segments(stripped)

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
            continue

        # 2) Page markers like "=== [Page 1/20] ==="
        if stripped.startswith("=== ") and stripped.endswith("==="):
            if buffer:
                segments.append(buffer)
                buffer = ""
            segments.append(stripped)
            continue

        # 3) Title heading (chapter, 前言, 番外, etc.)
        if is_title_heading:
            if buffer:
                segments.append(buffer)
                buffer = ""
            segments.append(stripped)
            continue

        # 4) First line of a new paragraph
        if not buffer:
            buffer = stripped
            continue

        buffer_text = buffer

        # 5) Buffer ends with CJK punctuation → new paragraph
        if buffer_text[-1] in CJK_PUNCT_END:
            segments.append(buffer_text)
            buffer = stripped
            continue

        # 6) Previous buffer looks like a heading-like short title
        if is_heading_like(buffer_text):
            segments.append(buffer_text)
            buffer = stripped
            continue

        # 7) Indentation → new paragraph
        if re.match(r"^\s{2,}", raw_line):
            segments.append(buffer_text)
            buffer = stripped
            continue

        # 8) Chapter-like endings: 章 / 节 / 部 / 卷 (with possible trailing brackets)
        if len(buffer_text) <= 15 and re.search(
            r"([章节部卷])[】》〗〕〉」』）]*$", buffer_text
        ):
            segments.append(buffer_text)
            buffer = stripped
            continue

        # 9) Default merge (soft line break)
        buffer += stripped

    # Flush last buffer
    if buffer:
        segments.append(buffer)

    # Formatting:
    # compact → "p1\np2\np3"
    # novel   → "p1\n\np2\n\np3"
    return "\n".join(segments) if compact else "\n\n".join(segments)
