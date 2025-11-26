# This Python file uses the following encoding: utf-8
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, Slot, QThread
from PySide6.QtGui import QGuiApplication
from PySide6.QtPdf import QPdfDocument
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QPushButton

from opencc_purepy import OpenCC
from opencc_purepy.office_helper import OFFICE_FORMATS, convert_office_doc
from pdf_extract_worker import PdfExtractWorker, build_progress_bar, get_progress_block
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_MainWindow

# Put this somewhere global (or wherever you keep CJK_PUNCT_END)
CJK_PUNCT_END = (
    '。', '！', '？', '；', '：', '…', '—', '”', '」', '’', '』',
    '）', '】', '》', '〗', '〕', '〉', '］', '｝', '章', '节', '部', '卷', '節'
)

# Title heading detection (same semantics as your C# TitleHeadingRegex)
TITLE_HEADING_REGEX = re.compile(
    r"^(?=.{0,60}$)"
    r"(前言|序章|终章|尾声|后记|番外|尾聲|後記|第.{0,10}?([章节部卷節]))"
)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_pdf_extraction = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self._pdf_thread: QThread | None = None
        self._pdf_worker: PdfExtractWorker | None = None  # type: ignore
        self._cancel_pdf_button: QPushButton | None = None  # type: ignore

        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.btnCopy.clicked.connect(self.btn_copy_click)
        self.ui.btnPaste.clicked.connect(self.btn_paste_click)
        self.ui.btnOpenFile.clicked.connect(self.btn_openfile_click)
        self.ui.btnSaveAs.clicked.connect(self.btn_savefile_click)
        self.ui.btnProcess.clicked.connect(self.btn_process_click)
        self.ui.btnExit.clicked.connect(btn_exit_click)
        self.ui.btnReflow.clicked.connect(self.reflow_cjk_paragraphs)
        self.ui.btnClearTbSource.clicked.connect(self.btn_clear_tb_source_clicked)
        self.ui.btnClearTbDestination.clicked.connect(self.btn_clear_tb_destination_clicked)
        self.ui.tbSource.textChanged.connect(self.update_char_count)
        self.ui.rbStd.clicked.connect(self.std_hk_select)
        self.ui.rbHK.clicked.connect(self.std_hk_select)
        self.ui.rbZhTw.clicked.connect(self.zhtw_select)
        self.ui.tabWidget.currentChanged[int].connect(self.tab_bar_changed)
        self.ui.cbZhTw.clicked[bool].connect(self.cbzhtw_clicked)
        self.ui.btnAdd.clicked.connect(self.btn_add_clicked)
        self.ui.btnRemove.clicked.connect(self.btn_remove_clicked)
        self.ui.btnClear.clicked.connect(self.btn_clear_clicked)
        self.ui.btnPreview.clicked.connect(self.btn_preview_clicked)
        self.ui.btnPreviewClear.clicked.connect(self.btn_preview_clear_clicked)
        self.ui.btnOutDir.clicked.connect(self.btn_out_directory_clicked)
        self.ui.cbManual.activated.connect(self.cb_manual_activated)
        self.ui.actionAbout.triggered.connect(self.action_about_triggered)
        self.ui.actionExit.triggered.connect(btn_exit_click)
        self.ui.tbSource.fileDropped.connect(self._on_tbSource_fileDropped)

        self.converter = OpenCC()

    def start_pdf_extraction(self, filename: str) -> None:
        """
        Kick off PDF text extraction in a background QThread.
        """
        # If an extraction is already running, you may want to ignore or cancel it
        if self._pdf_thread is not None:
            # optional: show a message / ignore
            self.statusBar().showMessage("PDF extraction already in progress.")
            return

        add_header = self.ui.actionAddPdfPageHeader.isChecked()

        # Create worker + thread
        self._pdf_thread = QThread(self)
        self._pdf_worker = PdfExtractWorker(filename, add_header)
        self._pdf_worker.moveToThread(self._pdf_thread)

        # Wire thread start → worker.run
        self._pdf_thread.started.connect(self._pdf_worker.run)  # type: ignore

        # Worker signals → MainWindow slots
        self._pdf_worker.progress.connect(self.on_pdf_progress)
        self._pdf_worker.finished.connect(self.on_pdf_finished)
        self._pdf_worker.error.connect(self.on_pdf_error)

        # Cleanup connections
        self._pdf_worker.finished.connect(self._pdf_thread.quit)
        self._pdf_worker.error.connect(self._pdf_thread.quit)
        self._pdf_thread.finished.connect(self._pdf_worker.deleteLater)  # type: ignore
        self._pdf_thread.finished.connect(self._on_pdf_thread_finished)  # type: ignore

        # Disable Reflow button while loading
        self.ui.btnReflow.setEnabled(False)

        # Add Cancel button to the right side of the status bar
        self._cancel_pdf_button = QPushButton("Cancel", self)
        self._cancel_pdf_button.setAutoDefault(False)
        self._cancel_pdf_button.setDefault(False)
        self._cancel_pdf_button.setFlat(True)
        self._cancel_pdf_button.setStyleSheet("""
            QPushButton {
                padding: 2px 8px;
                margin: 0px;
            }
        """)
        # IMPORTANT: connect to MainWindow slot, not directly to worker
        self._cancel_pdf_button.clicked.connect(self.on_pdf_cancel_clicked)  # type: ignore
        self.statusBar().addPermanentWidget(self._cancel_pdf_button)

        # Start the background thread
        self.statusBar().showMessage("Loading PDF...")
        self._pdf_thread.start()

    @Slot(int, int)
    def on_pdf_progress(self, current: int, total: int) -> None:
        """
        Called from worker thread via signal: update status bar.
        """
        percent = int(current / total * 100)
        bar = build_progress_bar(current, total, width=20)
        self.statusBar().showMessage(f"Loading PDF {bar}  {percent}%")

    @Slot(str, str, bool)
    def on_pdf_finished(self, text: str, filename: str, cancelled: bool) -> None:
        """
        Extraction finished (success or cancelled). Runs in GUI thread.
        """
        # Remove cancel button if present
        if self._cancel_pdf_button is not None:
            self.statusBar().removeWidget(self._cancel_pdf_button)
            self._cancel_pdf_button.deleteLater()
            self._cancel_pdf_button = None

        # Re-enable Reflow button
        self.ui.btnReflow.setEnabled(True)

        # Put extracted text into tbSource (even if partially cancelled)
        if text:
            self.ui.tbSource.setPlainText(text)

        self.detect_source_text_info()
        # stash the original filename (even for PDF)
        self.ui.tbSource.content_filename = filename

        if cancelled:
            self.statusBar().showMessage("PDF loading cancelled.")
        else:
            self.statusBar().showMessage("PDF loaded: " + filename)

    @Slot(str)
    def on_pdf_error(self, message: str) -> None:
        """
        Extraction encountered an error.
        """
        if self._cancel_pdf_button is not None:
            self.statusBar().removeWidget(self._cancel_pdf_button)
            self._cancel_pdf_button.deleteLater()
            self._cancel_pdf_button = None

        self.ui.btnReflow.setEnabled(True)
        self.statusBar().showMessage(f"Error loading PDF: {message}")
        # Optional: QMessageBox.critical(self, "PDF Error", message)

    @Slot()
    def _on_pdf_thread_finished(self) -> None:
        """
        Thread finished; clear references so another extraction can be started.
        """
        self._pdf_thread.deleteLater()
        self._pdf_thread = None
        self._pdf_worker = None

    @Slot(bool)
    def on_pdf_cancel_clicked(self, _checked: bool = False) -> None:
        """
        Called when the Cancel button in the status bar is clicked.
        Forwards the cancel request to the worker, if running.
        """
        if self._pdf_worker is not None:
            # Call the worker's slot; Qt will queue this into the worker thread
            self._pdf_worker.request_cancel()
            self.statusBar().showMessage("Cancelling PDF loading...")

    def _on_tbSource_fileDropped(self, path: str):
        self.detect_source_text_info()
        if not path:
            self.statusBar().showMessage("Text contents dropped")

    def action_about_triggered(self):
        QMessageBox.about(self, "About", "OpenccPurepyGui version 1.0.0 (c) 2025 Laisuk")

    def tab_bar_changed(self, index: int) -> None:
        if index == 0:
            self.ui.btnOpenFile.setEnabled(True)
            self.ui.lblFilename.setEnabled(True)
            self.ui.btnSaveAs.setEnabled(True)
        elif index == 1:
            self.ui.btnOpenFile.setEnabled(False)
            self.ui.lblFilename.setEnabled(False)
            self.ui.btnSaveAs.setEnabled(False)

    def update_char_count(self):
        self.ui.lblCharCount.setText(f"[ {len(self.ui.tbSource.document().toPlainText()):,} chars ]")

    def detect_source_text_info(self):
        text = self.ui.tbSource.toPlainText()
        if not text:
            return

        text_code = self.converter.zho_check(text)
        if text_code == 1:
            self.ui.lblSourceCode.setText("zh-Hant (繁体)")
            self.ui.rbT2s.setChecked(True)
        elif text_code == 2:
            self.ui.lblSourceCode.setText("zh-Hans (简体)")
            self.ui.rbS2t.setChecked(True)
        else:
            self.ui.lblSourceCode.setText("Non-zh (其它)")

        filename = getattr(self.ui.tbSource, "content_filename", None)
        if filename:
            base = os.path.basename(filename)
            self.ui.lblFilename.setText(base)
            self.statusBar().showMessage(f"File: {filename}")

    def extract_pdf_text(self, filename: str) -> str:
        """
        Extracts text from a PDF using QPdfDocument (Qt PDF).

        - Shows a text-based progress bar in the status bar.
        - Adds a temporary [Cancel] button on the right side.
        - If Cancel is clicked, stops early and returns the pages extracted so far.
        """
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        # --- Disable Reflow button during extraction ---
        self.ui.btnReflow.setEnabled(False)
        # --- Cancellation flag + button setup ---
        self._cancel_pdf_extraction = False

        cancel_button = QPushButton("Cancel")
        cancel_button.setAutoDefault(False)
        cancel_button.setDefault(False)
        cancel_button.setFlat(True)
        cancel_button.setStyleSheet("""
            QPushButton {
                padding: 2px 8px;
                margin: 0px;
            }
        """)

        def on_cancel_clicked():
            self._cancel_pdf_extraction = True
            # give immediate feedback
            self.statusBar().showMessage("Cancelling PDF loading...")

        cancel_button.clicked.connect(on_cancel_clicked)  # type: ignore

        # Add the cancel button to the right side of the status bar
        self.statusBar().addPermanentWidget(cancel_button)

        doc = QPdfDocument(self)
        err = doc.load(str(path))

        try:
            if err != QPdfDocument.Error.None_:
                raise RuntimeError(f"Failed to load PDF: {filename} (error={err})")

            page_count = doc.pageCount()
            if page_count <= 0:
                self.statusBar().showMessage("PDF has no pages.")
                return ""

            parts: List[str] = []

            block = get_progress_block(page_count)

            for i in range(page_count):
                # Check for cancel request
                if getattr(self, "_cancel_pdf_extraction", False):
                    self.statusBar().showMessage(
                        f"PDF loading cancelled at page {i}/{page_count}."
                    )
                    break

                selection = doc.getAllText(i)
                page_text = selection.text() if selection is not None else ""

                if self.ui.actionAddPdfPageHeader.isChecked():
                    parts.append(f"\n\n=== [Page {i + 1}/{page_count}] ===\n\n")

                parts.append(page_text)

                current = i + 1

                # Adaptive status update (like C#)
                if current % block == 0 or current == 1 or current == page_count:
                    percent = int(current / page_count * 100)
                    bar = build_progress_bar(current, page_count, width=20)
                    self.statusBar().showMessage(
                        f"Loading PDF {bar}  {percent}%"
                    )
                    QApplication.processEvents()

            # Only show "loaded" if not cancelled
            if not getattr(self, "_cancel_pdf_extraction", False):
                self.statusBar().showMessage("PDF loaded.")

            return "".join(parts)

        finally:
            # Always clean up button + reset flag
            try:
                self.statusBar().removeWidget(cancel_button)
            except (RuntimeError, AttributeError):
                # RuntimeError: underlying C++ object might already be deleted
                # AttributeError: unexpected missing method
                pass
            cancel_button.deleteLater()
            # --- Re-enable Reflow button ---
            self.ui.btnReflow.setEnabled(True)
            self._cancel_pdf_extraction = False
            doc.close()

    def reflow_cjk_paragraphs(self) -> None:
        """
        Reflows CJK text extracted from PDFs by merging artificial line breaks
        while preserving intentional paragraph / heading boundaries.

        Variables
        ----------
        add_pdf_page_header : bool
            If False, try to skip page-break-like blank lines that are not
            preceded by CJK punctuation (i.e., layout gaps between pages).
            If True, keep those gaps.
        compact : bool
            If True, join paragraphs with a single newline ("p1\\np2\\np3").
            If False (default), join with blank lines ("p1\\n\\np2\\n\\np3").
        """
        text = self.ui.tbSource.toPlainText()
        compact = self.ui.actionCompactPdfText.isChecked()
        add_pdf_page_header = self.ui.actionAddPdfPageHeader.isChecked()

        if not text.strip():
            return

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
            if len(buffer_text) <= 15 and re.search(r"([章节部卷])[】》〗〕〉」』）]*$", buffer_text):
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
        result = "\n".join(segments) if compact else "\n\n".join(segments)

        self.ui.tbSource.setPlainText(result)
        self.statusBar().showMessage("Reflow complete (CJK-aware)")

    def std_hk_select(self):
        self.ui.cbZhTw.setCheckState(Qt.CheckState.Unchecked)

    def zhtw_select(self):
        self.ui.cbZhTw.setCheckState(Qt.CheckState.Checked)

    def cbzhtw_clicked(self, status: bool) -> None:
        if status:
            self.ui.rbZhTw.setChecked(True)

    def btn_paste_click(self):
        if not QGuiApplication.clipboard().text():
            self.ui.statusbar.showMessage("Clipboard empty")
            return
        self.ui.tbSource.clear()
        self.ui.tbSource.paste()
        self.ui.tbSource.content_filename = ""
        self.ui.lblFilename.setText("")
        self.detect_source_text_info()
        self.ui.statusbar.showMessage("Clipboard contents pasted to source box")

    def btn_copy_click(self):
        text = self.ui.tbDestination.toPlainText()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        self.ui.statusbar.showMessage("Contents copied to clipboard")

    def btn_openfile_click(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            (
                "Text Files (*.txt);;"
                "Subtitle Files (*.srt *.vtt *.ass *.ttml2 *.xml);;"
                "XML Files (*.xml *.ttml2);;"
                "PDF Files (*.pdf);;"
                "All Files (*.*)"
            ),
        )
        if not filename:
            return

        contents = ""

        try:
            if filename.lower().endswith(".pdf"):
                if self.ui.actionUsePdfTextExtractWorker.isChecked():
                    self.start_pdf_extraction(filename)
                else:
                    contents = self.extract_pdf_text(filename)
            else:
                with open(filename, "r", encoding="utf-8") as f:
                    contents = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open/parse file:\n{e}")
            return

        self.ui.tbSource.setPlainText(contents)
        # stash the original filename (even for PDF)
        self.ui.tbSource.content_filename = filename
        self.detect_source_text_info()
        self.statusBar().showMessage(f"File: {filename}")

    def get_current_config(self):
        if self.ui.rbManual.isChecked():
            return self.ui.cbManual.currentText().split(' ')[0]

        if self.ui.rbS2t.isChecked():
            if self.ui.rbHK.isChecked():
                return "s2hk"
            if self.ui.rbStd.isChecked():
                return "s2t"
            return "s2twp" if self.ui.cbZhTw.isChecked() else "s2tw"

        if self.ui.rbT2s.isChecked():
            if self.ui.rbHK.isChecked():
                return "hk2s"
            if self.ui.rbStd.isChecked():
                return "t2s"
            return "tw2sp" if self.ui.cbZhTw.isChecked() else "tw2s"

        return "s2tw"

    def btn_process_click(self):
        config = self.get_current_config()
        is_punctuation = self.ui.cbPunct.isChecked()
        self.converter.set_config(config)

        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.tbDestination.clear()
            if not self.ui.tbSource.document().toPlainText():
                self.ui.statusbar.showMessage("Nothing to convert: Empty content.")
                return
            input_text = self.ui.tbSource.document().toPlainText()

            start_time = time.perf_counter()
            converted_text = self.converter.convert(input_text, is_punctuation)
            elapsed_ms = (time.perf_counter() - start_time) * 1000  # in milliseconds

            self.ui.tbDestination.document().setPlainText(converted_text)

            if self.ui.rbManual.isChecked():
                self.ui.lblDestinationCode.setText(self.ui.cbManual.currentText())
            else:
                if "Non" not in self.ui.lblSourceCode.text():
                    self.ui.lblDestinationCode.setText(
                        "zh-Hant (繁体)" if self.ui.rbS2t.isChecked() else "zh-Hans (简体)")
                else:
                    self.ui.lblDestinationCode.setText(self.ui.lblSourceCode.text())

            self.ui.statusbar.showMessage(f"Process completed in {elapsed_ms:.1f} ms ( {config} )")

        if self.ui.tabWidget.currentIndex() == 1:
            if self.ui.listSource.count() == 0:
                self.ui.statusbar.showMessage("Nothing to convert: Empty file list.")
                return

            out_dir = self.ui.lineEditDir.text()
            if not os.path.exists(out_dir):
                msg = QMessageBox(QMessageBox.Icon.Information, "Attention", "Invalid output directory.")
                msg.setInformativeText("Output directory:\n" + out_dir + "\nnot found.")
                msg.exec()
                self.ui.lineEditDir.setFocus()
                self.ui.statusbar.showMessage("Invalid output directory.")
            else:
                self.ui.tbPreview.clear()
                out_dir = Path(self.ui.lineEditDir.text())
                for index in range(self.ui.listSource.count()):
                    file_path = Path(self.ui.listSource.item(index).text())
                    # For single extension behavior (like splitext):
                    base = file_path.stem  # 'basename'
                    ext = file_path.suffix.lower()  # '.txt' or ''
                    ext_no_dot = ext.lstrip(".")

                    # If we want to preserve multipart extensions like .tar.gz, use:
                    # ext = ''.join(s.lower() for s in file_path.suffixes)

                    basename = (
                        self.converter.convert(base, is_punctuation)
                        if self.ui.actionConvert_filename.isChecked()
                        else base
                    )

                    if file_path.exists():
                        out_dir.mkdir(parents=True, exist_ok=True)  # make sure dir exists
                        output = out_dir / f"{basename}_{config}{ext}"
                        input_filename = str(file_path)
                        output_filename = str(output)

                        if ext_no_dot in OFFICE_FORMATS:
                            # Convert Office documents
                            success, message = convert_office_doc(input_filename, output_filename, ext_no_dot,
                                                                  self.converter,
                                                                  is_punctuation, True)
                            if success:
                                self.ui.tbPreview.appendPlainText(
                                    f"{index + 1}: {output_filename} -> {message} -> Done.")
                            else:
                                self.ui.tbPreview.appendPlainText(f"{index + 1}: {input_filename} -> Skip: {message}.")
                            continue

                        else:
                            # Convert plain text files
                            input_text = ""
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    input_text = f.read()
                            except UnicodeDecodeError:
                                input_text = ""

                            if input_text:
                                converted_text = self.converter.convert(input_text, self.ui.cbPunct.isChecked())

                                with open(output_filename, "w", encoding="utf-8") as f:
                                    f.write(converted_text)
                                self.ui.tbPreview.appendPlainText(f"{index + 1}: {output_filename} -> Done.")
                            else:
                                self.ui.tbPreview.appendPlainText(
                                    f"{index + 1}: {input_filename} -> Skip: Not text or valid file.")
                    else:
                        self.ui.tbPreview.appendPlainText(f"{index + 1}: {file_path} -> File not found.")
                self.ui.statusbar.showMessage("Process completed")

    def btn_savefile_click(self):
        filename = QFileDialog.getSaveFileName(
            self,
            "Save Text File",
            f"{self.ui.cbSaveTarget.currentText()}.txt",
            "Text File (*.txt);;All Files (*.*)")

        if not filename[0]:
            return

        target = self.ui.cbSaveTarget.currentText()
        with open(filename[0], "w", encoding="utf-8") as f:
            if self.ui.cbSaveTarget.currentIndex() == 0:
                f.write(self.ui.tbSource.toPlainText())
            else:
                f.write(self.ui.tbDestination.toPlainText())
        self.ui.statusbar.showMessage(f"{target} contents saved to {filename[0]}")

    def btn_add_clicked(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Files",
            "",
            "Text Files (*.txt);;Office Files (*.docx *.xlsx *.pptx *.odt *.ods *.odp *.epub);;All Files (*.*)"
        )
        if files:
            self.display_file_list(files)
            self.ui.statusbar.showMessage("File(s) added.")

    def display_file_list(self, files):
        existing = {self.ui.listSource.item(i).text() for i in range(self.ui.listSource.count())}
        for file in files:
            if file not in existing:
                self.ui.listSource.addItem(file)
                existing.add(file)

    def btn_remove_clicked(self):
        selected_items = self.ui.listSource.selectedItems()
        if selected_items:
            for selected_item in selected_items:
                self.ui.listSource.takeItem(self.ui.listSource.row(selected_item))
            self.ui.statusbar.showMessage("File(s) removed.")

    def btn_clear_clicked(self):
        self.ui.listSource.clear()
        self.ui.statusbar.showMessage("File list cleared.")

    def btn_preview_clicked(self):
        selected_items = self.ui.listSource.selectedItems()
        # Initialize contents to a default value
        contents = ""
        if selected_items:
            selected_item = selected_items[0]
            file_path = selected_item.text()
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    contents = f.read()
                self.ui.statusbar.showMessage(f"File preview: {selected_items[0].text()}")
            except UnicodeDecodeError:
                contents = "❌ Not a valid text file"  # Already initialized, but good to explicitly handle for clarity
                self.ui.statusbar.showMessage(f"{file_path}: Not a valid text file.")
            except FileNotFoundError:  # Add this to handle non-existent files
                contents = "❌ File not found"
                self.ui.statusbar.showMessage(f"{file_path}: File not found.")
            except Exception as e:  # Catch other potential errors
                contents = "❌ Error opening file"
                self.ui.statusbar.showMessage(f"Error opening {file_path}: {e}")

        self.ui.tbPreview.setPlainText(contents)

    def btn_out_directory_clicked(self):
        directory = QFileDialog.getExistingDirectory(self, "Select output directory")
        if directory:
            self.ui.lineEditDir.setText(directory)
            self.ui.statusbar.showMessage(f"Output directory set: {directory}")

    def btn_preview_clear_clicked(self):
        self.ui.tbPreview.clear()
        self.ui.statusbar.showMessage("File preview cleared.")

    def btn_clear_tb_source_clicked(self):
        self.ui.tbSource.clear()
        self.ui.lblSourceCode.setText("")
        self.ui.tbSource.content_filename = ""
        self.ui.lblFilename.setText("")
        self.ui.statusbar.showMessage("Source contents cleared.")

    def btn_clear_tb_destination_clicked(self):
        self.ui.tbDestination.clear()
        self.ui.lblDestinationCode.setText("")
        self.ui.statusbar.showMessage("Destination contents cleared.")

    def cb_manual_activated(self):
        self.ui.rbManual.setChecked(True)


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


def btn_exit_click():
    QApplication.quit()


if __name__ == "__main__":
    app = QApplication()
    app.setStyle("WindowsVista")
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
