# This Python file uses the following encoding: utf-8
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Callable

from PySide6.QtCore import Qt, Slot, QThread
from PySide6.QtGui import QGuiApplication, QTextCursor
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QPushButton

from batch_worker import BatchWorker
from opencc_purepy import OpenCC
from pdf_extract_worker import PdfExtractWorker
from pdf_helper import build_progress_bar, reflow_cjk_paragraphs_core, extract_pdf_text_core
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._batch_worker = None
        self._batch_thread = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # state
        self._pdf_thread: QThread | None = None
        self._pdf_worker: Optional[PdfExtractWorker] = None
        self._cancel_button: Optional[QPushButton] = None
        self._cancel_pdf_extraction = None
        self._pdf_sequential_active = False

        # shared Cancel button (hidden by default)
        self._cancel_button = QPushButton("Cancel", self)
        self._cancel_button.setAutoDefault(False)
        self._cancel_button.setDefault(False)
        self._cancel_button.setFlat(True)
        self._cancel_button.setStyleSheet(
            "QPushButton { padding: 2px 8px; margin: 0px; }"
        )
        self._cancel_button.hide()
        # self._cancel_pdf_button.clicked.connect(self.on_pdf_cancel_clicked)  # type: ignore
        self.statusBar().addPermanentWidget(self._cancel_button)

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

    def show_cancel_button(self, handler):
        """Show Cancel button and connect to the given handler."""
        try:
            self._cancel_button.clicked.disconnect()  # type: ignore
        except Exception():
            pass  # ok if nothing connected yet

        self._cancel_button.clicked.connect(handler)  # type: ignore
        self._cancel_button.show()

    def hide_cancel_button(self):
        """Hide Cancel button and remove handlers."""
        try:
            self._cancel_button.clicked.disconnect()  # type: ignore
        except Exception():
            pass

        self._cancel_button.hide()

    def disable_process_ui(self):
        """Disable processing controls to prevent re-entry."""
        self.ui.btnProcess.setEnabled(False)
        self.ui.btnReflow.setEnabled(False)  # optional
        # self.ui.tabWidget.setEnabled(False)  # optional — prevents switching tabs mid-process

    def enable_process_ui(self):
        """Re-enable processing controls after batch or PDF task is finished."""
        self.ui.btnProcess.setEnabled(True)
        self.ui.btnReflow.setEnabled(True)
        # self.ui.tabWidget.setEnabled(True)

    # ====== Main Worker ======

    def start_pdf_extraction(self, filename: str) -> None:
        """
        Interactive single-PDF extraction entry point.
        Uses the core wiring and adds UI behaviour.
        """
        # Guard: only one PDF extraction at a time in interactive mode
        if self._pdf_thread is not None:
            self.statusBar().showMessage("PDF extraction already in progress.")
            return

        add_header = self.ui.actionAddPdfPageHeader.isChecked()

        # UI-specific bits
        self.ui.btnReflow.setEnabled(False)
        # self._cancel_pdf_button.show()
        self.show_cancel_button(self.on_pdf_cancel_clicked)
        self.disable_process_ui()
        self.statusBar().showMessage("Loading PDF...")

        # Reuse the core
        self.start_pdf_extraction_core(
            filename=filename,
            add_header=add_header,
            on_progress=self.on_pdf_progress,
            on_finished=self.on_pdf_finished,
            on_error=self.on_pdf_error,
        )

    def start_pdf_extraction_core(
            self,
            filename: str,
            add_header: bool,
            on_progress: Callable[[int, int], None],
            on_finished: Callable[[str, str, bool], None],
            on_error: Callable[[str], None],
    ) -> None:
        """
        Core wiring for PDF extraction in a background QThread.

        - No direct UI logic (no statusBar, no buttons).
        - Caller decides which slots to connect.
        - Reusable for both single-file UI and batch processing.
        """
        # Create worker + thread
        self._pdf_thread = QThread(self)
        self._pdf_worker = PdfExtractWorker(filename, add_header)
        self._pdf_worker.moveToThread(self._pdf_thread)

        # Thread start → worker.run
        self._pdf_thread.started.connect(self._pdf_worker.run)  # type: ignore

        # Connect worker signals → caller-provided handlers
        if on_progress is not None:
            self._pdf_worker.progress.connect(on_progress)
        if on_finished is not None:
            self._pdf_worker.finished.connect(on_finished)
        if on_error is not None:
            self._pdf_worker.error.connect(on_error)

        # Cleanup
        self._pdf_worker.finished.connect(self._pdf_thread.quit)
        self._pdf_worker.error.connect(self._pdf_thread.quit)
        self._pdf_thread.finished.connect(self._pdf_worker.deleteLater)  # type: ignore
        self._pdf_thread.finished.connect(self._on_pdf_thread_finished)  # type: ignore

        # Start background thread
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
        # Hide cancel button if present
        # self._cancel_pdf_button.hide()
        self.enable_process_ui()
        self.hide_cancel_button()
        # Re-enable Reflow button
        self.ui.btnReflow.setEnabled(True)
        if self.ui.actionAutoReflow.isChecked():
            addPageHeader = self.ui.actionAddPdfPageHeader.isChecked()
            compact = self.ui.actionCompactPdfText.isChecked()
            text = reflow_cjk_paragraphs_core(text,
                                              add_pdf_page_header=addPageHeader,
                                              compact=compact)
        # Put extracted text into tbSource (even if partially cancelled)
        if text:
            self.ui.tbSource.setPlainText(text)

        # stash the original filename (even for PDF)
        self.ui.tbSource.content_filename = filename
        self.detect_source_text_info()

        if cancelled:
            self.statusBar().showMessage("❌ PDF loading cancelled: " + filename)
        else:
            self.statusBar().showMessage(
                f"✅ PDF loaded{(' (Auto-Reflowed)' if self.ui.actionAutoReflow.isChecked() else '')}: " + filename)

    @Slot(str)
    def on_pdf_error(self, message: str) -> None:
        """
        Extraction encountered an error.
        """
        # self._cancel_pdf_button.hide()
        self.enable_process_ui()
        self.hide_cancel_button()
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
        Routes the cancel request to either:
        - the PDF worker (async mode), or
        - the sequential extractor (sync mode).
        """
        if self._pdf_worker is not None:
            # Worker mode: queue cancel into worker thread
            self._pdf_worker.request_cancel()
            self.statusBar().showMessage("Cancelling PDF loading (worker)...")

        elif getattr(self, "_pdf_sequential_active", False):
            # Sequential mode: flip the flag checked by extract_pdf_text()
            self._cancel_pdf_extraction = True
            self.statusBar().showMessage("Cancelling PDF loading (sequential)...")

    # ====== Main Worker End ======

    # ====== Batch Processing ======

    def on_batch_progress(self, current: int, total: int) -> None:
        self.ui.statusbar.showMessage(f"Processing {current}/{total}...")

    def on_batch_error(self, msg: str) -> None:
        self.ui.tbPreview.appendPlainText(f"[Error] {msg}")
        self.ui.statusbar.showMessage(msg)
        self.hide_cancel_button()
        self.enable_process_ui()

    def on_batch_finished(self, cancelled: bool) -> None:
        if cancelled:
            self.ui.tbPreview.appendPlainText("❌ Batch cancelled.")
            self.ui.statusbar.showMessage("❌ Batch cancelled.")
        else:
            self.ui.tbPreview.appendPlainText("✅ Batch conversion done.")
            self.ui.statusbar.showMessage("Batch completed.")
        self.hide_cancel_button()
        self.enable_process_ui()

    def _on_batch_thread_finished(self) -> None:
        self._batch_thread = None
        self._batch_worker = None

    def on_batch_cancel_clicked(self):
        if self._batch_worker is not None:
            self._batch_worker.request_cancel()
            self.ui.statusbar.showMessage("Cancelling batch...")

    # ====== Batch Processing End ======

    def _on_tbSource_fileDropped(self, path: str):
        self.detect_source_text_info()
        if not path:
            self.statusBar().showMessage("Text contents dropped")
        else:
            self.statusBar().showMessage("File dropped: " + path)

    def action_about_triggered(self):
        QMessageBox.about(self, "About", "OpenccPurepyGui version 1.0.0 (c) 2025 Laisuk")

    def tab_bar_changed(self, index: int) -> None:
        if index == 0:
            self.ui.btnOpenFile.setEnabled(True)
            self.ui.lblFilename.setEnabled(True)
            self.ui.btnSaveAs.setEnabled(True)
            self.ui.cbSaveTarget.setEnabled(True)
        elif index == 1:
            self.ui.btnOpenFile.setEnabled(False)
            self.ui.lblFilename.setEnabled(False)
            self.ui.btnSaveAs.setEnabled(False)
            self.ui.cbSaveTarget.setEnabled(False)

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
            # self.statusBar().showMessage(f"File: {filename}")

    def extract_pdf_text(self, filename: str) -> str:
        """
        Extracts text from a PDF using the core PDF helper.

        - Shows a text-based progress bar in the status bar.
        - Adds a temporary [Cancel] button on the right side.
        - If Cancel is clicked, stops early and returns the pages extracted so far.
        """
        self._pdf_sequential_active = True
        self._cancel_pdf_extraction = False
        self._cancel_button.show()
        self.ui.btnReflow.setEnabled(False)

        # Track last progress for nicer "cancelled at page X/Y" message
        last_page: int = 0
        total_pages: int = 0

        def on_progress(current: int, total: int) -> None:
            nonlocal last_page, total_pages
            last_page, total_pages = current, total
            percent = int(current / total * 100)
            bar = build_progress_bar(current, total, width=20)
            self.statusBar().showMessage(f"Loading PDF {bar}  {percent}%")
            QApplication.processEvents()

        def is_cancelled() -> bool:
            return bool(self._cancel_pdf_extraction)

        try:
            text = extract_pdf_text_core(
                filename,
                add_pdf_page_header=self.ui.actionAddPdfPageHeader.isChecked(),
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )
            # Decide final status message
            if self._cancel_pdf_extraction:
                if last_page and total_pages:
                    # Normal: cancelled after reading some pages
                    self.statusBar().showMessage(
                        f"❌ PDF loading cancelled at page {last_page}/{total_pages} - {filename}."
                    )
                elif text:
                    # Rare case: partial text but no progress callback fired
                    self.statusBar().showMessage(
                        f"❌ PDF loading cancelled (partial text extracted). ({filename})"
                    )
                else:
                    # Cancelled immediately before loading page 1
                    self.statusBar().showMessage(f"❌ PDF loading cancelled - {filename}.")
            else:
                # Not cancelled
                if not text:
                    self.statusBar().showMessage("❌ PDF has no pages.")
                else:
                    self.statusBar().showMessage("✅ PDF loaded successfully.")

            return text
        finally:
            self._pdf_sequential_active = False
            self._cancel_pdf_extraction = False
            self._cancel_button.hide()
            self.ui.btnReflow.setEnabled(True)

    def reflow_cjk_paragraphs(self) -> None:
        """
        Reflows CJK text extracted from PDFs by merging artificial line breaks
        while preserving intentional paragraph / heading boundaries.

        Behavior
        --------
        - If there is a selection in tbSource, only the selected text is reflowed.
        - If there is no selection, the entire document is reflowed.
        - The change is wrapped in a single edit block, so one Undo restores the
          pre-reflow state.

        Parameters
        ----------
        self.add_pdf_page_header : bool
            If False, try to skip page-break-like blank lines that are not
            preceded by CJK punctuation (i.e., layout gaps between pages).
            If True, keep those gaps.
        self.compact : bool
            If True, join paragraphs with a single newline ("p1\\np2\\np3").
            If False (default), join with blank lines ("p1\\n\\np2\\n\\np3").
        """
        edit = self.ui.tbSource
        cursor = edit.textCursor()
        has_selection = cursor.hasSelection()

        if has_selection:
            src = cursor.selection().toPlainText()
        else:
            src = edit.toPlainText()

        if not src.strip():
            self.statusBar().showMessage("Source text is empty. Nothing to reflow.")
            return

        compact = self.ui.actionCompactPdfText.isChecked()
        add_pdf_page_header = self.ui.actionAddPdfPageHeader.isChecked()

        result = reflow_cjk_paragraphs_core(
            src,
            add_pdf_page_header=add_pdf_page_header,
            compact=compact,
        )

        if has_selection:
            # Replace only the selected range, as one undoable step
            cursor.beginEditBlock()
            cursor.insertText(result)  # replaces selection
            cursor.endEditBlock()
            edit.setTextCursor(cursor)
        else:
            # Replace the entire document, also as one undoable step
            doc_cursor = QTextCursor(edit.document())
            doc_cursor.beginEditBlock()
            doc_cursor.select(QTextCursor.Document)
            doc_cursor.insertText(result)
            doc_cursor.endEditBlock()

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
                    # Only update the editor + metadata here, but DO NOT override the status bar
                    self.ui.tbSource.setPlainText(contents)
                    self.ui.tbSource.content_filename = filename
                    self.detect_source_text_info()

                return

            # --- Non-PDF branch ---
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

    def btn_process_click(self) -> None:
        """
        Shell / entry point for the Process button.
        Decides which processing mode to run based on the selected tab.
        """
        config = self.get_current_config()
        is_punctuation = self.ui.cbPunct.isChecked()
        self.converter.set_config(config)

        current_tab = self.ui.tabWidget.currentIndex()
        if current_tab == 0:
            self.main_process(config, is_punctuation)
        elif current_tab == 1:
            self.batch_process(config, is_punctuation)
        else:
            # Just in case more tabs are added in future
            self.ui.statusbar.showMessage("Unsupported tab for processing.")

    def main_process(self, config: str, is_punctuation: bool) -> None:
        """
        Single-text conversion (Tab 0).
        Converts the content of tbSource into tbDestination.
        """
        self.ui.tbDestination.clear()

        input_text = self.ui.tbSource.document().toPlainText()
        if not input_text:
            self.ui.statusbar.showMessage("Nothing to convert: Empty content.")
            return

        start_time = time.perf_counter()
        converted_text = self.converter.convert(input_text, is_punctuation)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0  # ms

        self.ui.tbDestination.document().setPlainText(converted_text)

        # Update destination language label
        if self.ui.rbManual.isChecked():
            self.ui.lblDestinationCode.setText(self.ui.cbManual.currentText())
        else:
            if "Non" not in self.ui.lblSourceCode.text():
                self.ui.lblDestinationCode.setText(
                    "zh-Hant (繁体)" if self.ui.rbS2t.isChecked() else "zh-Hans (简体)"
                )
            else:
                self.ui.lblDestinationCode.setText(self.ui.lblSourceCode.text())

        self.ui.statusbar.showMessage(
            f"Process completed in {elapsed_ms:.1f} ms ( {config} )"
        )

    def batch_process(self, config: str, is_punctuation: bool) -> None:
        """
        Batch file conversion (Tab 1).
        Starts a BatchWorker in a QThread to process all files in listSource.
        """
        if self.ui.listSource.count() == 0:
            self.ui.statusbar.showMessage("Nothing to convert: Empty file list.")
            return

        out_dir = self.ui.lineEditDir.text()
        if not os.path.exists(out_dir):
            msg = QMessageBox(
                QMessageBox.Icon.Information,
                "Attention",
                "Invalid output directory.",
            )
            msg.setInformativeText(
                "Output directory:\n" + out_dir + "\nnot found."
            )
            msg.exec()
            self.ui.lineEditDir.setFocus()
            self.ui.statusbar.showMessage("Invalid output directory.")
            return

        out_path = Path(out_dir)

        # Collect file list from ListBox
        files = [
            self.ui.listSource.item(i).text()
            for i in range(self.ui.listSource.count())
        ]

        # Snapshot PDF-related and filename options
        add_header = self.ui.actionAddPdfPageHeader.isChecked()
        auto_reflow = self.ui.actionAutoReflow.isChecked()
        compact = self.ui.actionCompactPdfText.isChecked()
        convert_filename = self.ui.actionConvert_filename.isChecked()

        self.ui.tbPreview.clear()
        self.ui.statusbar.showMessage("Starting batch conversion...")

        self.disable_process_ui()
        self.show_cancel_button(self.on_batch_cancel_clicked)

        # Create thread + worker
        self._batch_thread = QThread(self)
        self._batch_worker = BatchWorker(
            files=files,
            out_dir=out_path,
            converter=self.converter,
            config=config,
            is_punctuation=is_punctuation,
            add_pdf_page_header=add_header,
            auto_reflow_pdf=auto_reflow,
            compact_pdf=compact,
            convert_filename=convert_filename,
            parent=None,  # worker is thread-owned; no need to parent to MainWindow
        )
        self._batch_worker.moveToThread(self._batch_thread)

        # Connections
        self._batch_thread.started.connect(self._batch_worker.run)  # type: ignore
        self._batch_worker.log.connect(self.ui.tbPreview.appendPlainText)
        self._batch_worker.progress.connect(self.on_batch_progress)
        self._batch_worker.error.connect(self.on_batch_error)
        self._batch_worker.finished.connect(self.on_batch_finished)

        # Cleanup
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._batch_worker.deleteLater)  # type: ignore
        self._batch_thread.finished.connect(self._on_batch_thread_finished)  # type: ignore

        self._batch_thread.start()

    def btn_savefile_click(self):
        target = self.ui.cbSaveTarget.currentText()
        filename = QFileDialog.getSaveFileName(
            self,
            "Save Text File",
            f"{target}.txt",
            "Text File (*.txt);;All Files (*.*)")

        if not filename[0]:
            return

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
            "Text Files (*.txt);;"
            "Office Files (*.docx *.xlsx *.pptx *.odt *.ods *.odp *.epub);;"
            "PDF Files (*.pdf);;"
            "All Files (*.*)"
        )
        if files:
            self.display_file_list(files)
            self.ui.statusbar.showMessage("File(s) added.")

    def display_file_list(self, files):
        # 1) Collect existing items
        all_paths = []
        existing = set()

        for i in range(self.ui.listSource.count()):
            path = self.ui.listSource.item(i).text()
            all_paths.append(path)
            existing.add(path)

        # 2) Add new files (deduplicated)
        for file in files:
            if file not in existing:
                all_paths.append(file)
                existing.add(file)

        # 3) Re-group: non-PDF first, PDFs at bottom
        def is_pdf(pth: str) -> bool:
            return pth.lower().endswith(".pdf")

        non_pdfs = [p for p in all_paths if not is_pdf(p)]
        pdfs = [p for p in all_paths if is_pdf(p)]

        # 4) Rebuild the list widget
        self.ui.listSource.clear()
        for path in non_pdfs + pdfs:
            self.ui.listSource.addItem(path)

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


def btn_exit_click():
    QApplication.quit()


if __name__ == "__main__":
    app = QApplication()
    app.setStyle("WindowsVista")
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
