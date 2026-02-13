from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

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

_INVISIBLE_MAP = {
    0x200b: None,  # ZERO WIDTH SPACE
    0xfeff: None,  # BOM
    0x200e: None,  # LTR mark
    0x200f: None,  # RTL mark
}


def sanitize_invisible(text: str) -> str:
    return text.translate(_INVISIBLE_MAP)


# ---------------------------------------------------------------------------
# Core PDF extraction (no Qt, reusable for batch)
# ---------------------------------------------------------------------------

def extract_pdf_text_core(
        filename: str,
        add_pdf_page_header: bool = False,
        on_progress: Optional["ProgressCallback"] = None,
        is_cancelled: Optional["CancelCallback"] = None,
) -> str:
    """
    Core PDF extraction using Pdfium (ctypes backend).

    Keeps identical external behavior to the old PyMuPDF version.
    """
    from pdf_module.pdfium_helper import extract_pdf_pages_with_callback_pdfium
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    parts: List[str] = []
    cancelled = False

    def _callback(page: int, total: int, text: str) -> None:
        nonlocal cancelled

        if is_cancelled is not None and is_cancelled():
            cancelled = True
            return

        if add_pdf_page_header:
            parts.append(f"\n\n=== [Page {page}/{total}] ===\n\n")

        parts.append(text)

        # Progress throttling (same as your previous block logic)
        block = get_progress_block(total)
        if page % block == 0 or page == 1 or page == total:
            if on_progress is not None:
                on_progress(page, total)

    # Run Pdfium extraction
    extract_pdf_pages_with_callback_pdfium(str(path), _callback)

    if cancelled:
        return ""

    return "".join(parts)
