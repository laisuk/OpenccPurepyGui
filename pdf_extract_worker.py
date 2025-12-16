from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from pdf_helper import extract_pdf_text_core  # adjust import path as needed


class PdfExtractWorker(QObject):
    """
    Worker object that runs in a background QThread and extracts text
    from a PDF using PyMuPDF (pymupdf).
    """

    progress = Signal(int, int)  # (current_page, total_pages)
    finished = Signal(str, str, bool)  # (text, filename, cancelled)
    error = Signal(str)  # error message

    def __init__(
            self,
            filename: str,
            add_pdf_page_header: bool,
            parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._filename = filename
        self._add_pdf_page_header = add_pdf_page_header
        self._cancel_requested = False

    @Slot()
    def run(self) -> None:
        """
        Main worker entry point. Runs entirely in the worker thread.
        """
        # Keep the "file not found" behavior identical to old version
        path = Path(self._filename)
        if not path.is_file():
            self.error.emit(f"PDF not found: {path}")
            # finished with empty text, not canceled
            self.finished.emit("", self._filename, False)
            return

        try:
            text = extract_pdf_text_core(
                self._filename,
                add_pdf_page_header=self._add_pdf_page_header,
                on_progress=lambda cur, total: self.progress.emit(cur, total),
                is_cancelled=lambda: self._cancel_requested,
            )
        except FileNotFoundError as e:
            # Redundant with pre-check, but kept for safety
            self.error.emit(str(e))
            self.finished.emit("", self._filename, False)
            return
        except Exception as e:  # noqa: BLE001
            # Match old behavior: emit error, no finished() on load/other errors
            self.error.emit(str(e))
            return

        cancelled = self._cancel_requested
        self.finished.emit(text, self._filename, cancelled)

    @Slot()
    def request_cancel(self) -> None:
        """
        Called (indirectly) from the GUI thread to ask the worker to stop.
        This slot runs in the worker thread (queued connection).
        """
        self._cancel_requested = True
