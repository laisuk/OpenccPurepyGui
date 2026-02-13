from __future__ import annotations

from pathlib import Path
from typing import Optional
from PySide6.QtCore import QObject, Signal, Slot
from opencc_purepy.office_helper import OFFICE_FORMATS, convert_office_doc
# reuse your existing helpers
from pdf_module.pdf_helper import extract_pdf_text_core, reflow_cjk_paragraphs_core, sanitize_invisible


class BatchWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(bool)
    error = Signal(str)

    def __init__(
            self,
            files: list[str],
            out_dir: Path,
            converter,
            config: str,
            is_punctuation: bool,
            add_pdf_page_header: bool,
            auto_reflow_pdf: bool,
            compact_pdf: bool,
            convert_filename: bool,
            parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._files = [Path(p) for p in files]
        self._out_dir = out_dir
        self._converter = converter
        self._config = config
        self._is_punctuation = is_punctuation

        # PDF options (copied from UI at start)
        self._add_pdf_page_header = add_pdf_page_header
        self._auto_reflow_pdf = auto_reflow_pdf
        self._compact_pdf = compact_pdf
        self.convert_filename = convert_filename

        self._cancel_requested = False

    @Slot()
    def run(self) -> None:
        total = len(self._files)
        if total == 0:
            self.finished.emit(False)
            return

        self._out_dir.mkdir(parents=True, exist_ok=True)

        for idx, file_path in enumerate(self._files, start=1):
            if self._cancel_requested:
                self.log.emit("Batch cancelled.")
                self.finished.emit(True)
                return

            if not file_path.exists():
                self.log.emit(f"{idx}: {file_path} -> File not found.")
                self.progress.emit(idx, total)
                continue

            try:
                self._process_one_file(idx, total, file_path)
            except Exception as e:  # noqa: BLE001
                self.error.emit(f"{idx}: {file_path} -> Error: {e}")
            finally:
                self.progress.emit(idx, total)

        self.finished.emit(False)

    def _process_one_file(self, idx: int, total: int, file_path: Path) -> None:
        ext = file_path.suffix.lower()
        ext_no_dot = ext.lstrip(".")
        base = file_path.stem

        # filename conversion
        basename = self._converter.convert(base, self._is_punctuation) if self.convert_filename else base
        out_dir = self._out_dir

        # PDF
        if ext_no_dot == "pdf":
            self._process_pdf(idx, total, file_path, basename)
            return

        # Office
        output = out_dir / f"{basename}_{self._config}{ext}"
        if ext_no_dot in OFFICE_FORMATS:
            success, message = convert_office_doc(
                str(file_path),
                str(output),
                ext_no_dot,
                self._converter,
                self._is_punctuation,
                True,
            )
            if success:
                self.log.emit(f"{idx}: {output} -> {message} -> Done.")
            else:
                self.log.emit(f"{idx}: {file_path} -> Skip: {message}.")
            return

        # Plain text
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                input_text = f.read()
        except UnicodeDecodeError:
            input_text = ""

        if input_text:
            input_text = sanitize_invisible(input_text)
            converted_text = self._converter.convert(
                input_text,
                self._is_punctuation,
            )
            with open(output, "w", encoding="utf-8") as f:
                f.write(converted_text)
            self.log.emit(f"{idx}: {output} -> Done.")
        else:
            self.log.emit(f"{idx}: {file_path} -> Skip: Not text or valid file.")

    def _process_pdf(self, idx: int, total: int, file_path: Path, basename: str) -> None:
        add_header = self._add_pdf_page_header
        auto_reflow = self._auto_reflow_pdf
        compact = self._compact_pdf

        output = self._out_dir / f"{basename}_{self._config}.txt"
        self.log.emit(f"Processing PDF ({idx}/{total})... Please wait...")

        raw_text = extract_pdf_text_core(
            str(file_path),
            add_pdf_page_header=add_header,
            on_progress=None,
            is_cancelled=lambda: self._cancel_requested,
        )
        if not raw_text:
            self.log.emit(f"{idx}: {file_path} -> Skip: Empty or non-text PDF.")
            return

        raw_text = sanitize_invisible(raw_text)

        if auto_reflow:
            raw_text = reflow_cjk_paragraphs_core(
                raw_text,
                compact=compact,
                add_pdf_page_header=add_header,
            )

        converted_text = self._converter.convert(
            raw_text,
            self._is_punctuation,
        )
        with open(output, "w", encoding="utf-8") as f:
            f.write(converted_text)

        self.log.emit(f"{idx}: {output} -> Done.")

    @Slot()
    def request_cancel(self) -> None:
        self._cancel_requested = True
