from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtPdf import QPdfDocument


class PdfExtractWorker(QObject):
    """
    Worker object that runs in a background QThread and extracts text
    from a PDF using QPdfDocument.
    """

    progress = Signal(int, int)      # (current_page, total_pages)
    finished = Signal(str, str, bool)   # (text, filename, cancelled)
    error = Signal(str)             # error message

    def __init__(self, filename: str, add_pdf_page_header: bool, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._filename = filename
        self._add_pdf_page_header = add_pdf_page_header
        self._cancel_requested = False

    @Slot()
    def run(self) -> None:
        """
        Main worker entry point. Runs entirely in the worker thread.
        """
        path = Path(self._filename)
        if not path.is_file():
            self.error.emit(f"PDF not found: {path}")
            return

        doc = QPdfDocument()
        err = doc.load(str(path))

        if err != QPdfDocument.Error.None_:
            self.error.emit(f"Failed to load PDF: {path} (error={err})")
            doc.close()
            return

        try:
            page_count = doc.pageCount()
            if page_count <= 0:
                # No pages → finished with empty text, not cancelled
                self.finished.emit("", False)
                return

            parts: List[str] = []
            block = get_progress_block(page_count)
            cancelled = False

            for i in range(page_count):
                if self._cancel_requested:
                    cancelled = True
                    break

                selection = doc.getAllText(i)
                page_text = selection.text() if selection is not None else ""

                if self._add_pdf_page_header:
                    parts.append(f"\n\n=== [Page {i + 1}/{page_count}] ===\n\n")

                parts.append(page_text)

                current = i + 1
                if current % block == 0 or current == 1 or current == page_count:
                    self.progress.emit(current, page_count)

            text = "".join(parts)
            self.finished.emit(text, self._filename, cancelled)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            doc.close()

    @Slot()
    def request_cancel(self) -> None:
        """
        Called (indirectly) from the GUI thread to ask the worker to stop.
        This slot runs in the worker thread (queued connection).
        """
        self._cancel_requested = True

def get_progress_block(total_pages: int) -> int:
    """
    Adaptive progress update interval (port of C# GetProgressBlock).
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
