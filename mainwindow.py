# This Python file uses the following encoding: utf-8
import os
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

from opencc_purepy import OpenCC
from opencc_purepy.office_helper import OFFICE_FORMATS, convert_office_doc
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.btnCopy.clicked.connect(self.btn_copy_click)
        self.ui.btnPaste.clicked.connect(self.btn_paste_click)
        self.ui.btnOpenFile.clicked.connect(self.btn_openfile_click)
        self.ui.btnSaveAs.clicked.connect(self.btn_savefile_click)
        self.ui.btnProcess.clicked.connect(self.btn_process_click)
        self.ui.btnExit.clicked.connect(btn_exit_click)
        self.ui.btnClearTbSource.clicked.connect(self.btn_clear_tb_source_clicked)
        self.ui.btnClearTbDestination.clicked.connect(self.btn_clear_tb_destination_clicked)
        self.ui.tbSource.textChanged.connect(self.update_char_count)
        self.ui.rbStd.clicked.connect(self.std_hk_select)
        self.ui.rbHK.clicked.connect(self.std_hk_select)
        self.ui.rbZhTw.clicked.connect(self.zhtw_select)
        self.ui.btnDetect.clicked.connect(self.detect_source_text_info)
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

        self.converter = OpenCC()

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
        if text:
            self.update_source_code(get_text_code(text))
            self.ui.lblFilename.setText(os.path.basename(self.ui.tbSource.content_filename))
        if self.ui.tbSource.content_filename:
            self.statusBar().showMessage(f"File: {self.ui.tbSource.content_filename}")

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
        self.detect_source_text_info()
        self.ui.statusbar.showMessage("Clipboard contents pasted to source box")

    def btn_copy_click(self):
        text = self.ui.tbDestination.toPlainText()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        self.ui.statusbar.showMessage("Contents copied to clipboard")

    def btn_openfile_click(self):
        filename = QFileDialog.getOpenFileName(
            self,
            "Open Text File",
            "",
            "Text Files (*.txt);;"
            "Subtitle Files (*.srt *.vtt *.ass *.ttml2 *.xml));;"
            "XML Files (*.xml *.ttml2);;"
            "All Files (*.*)")
        if not filename[0]:
            return

        with open(filename[0], "r", encoding="utf-8") as f:
            contents = f.read()

        self.ui.tbSource.document().setPlainText(contents)
        self.ui.tbSource.content_filename = filename[0]
        self.detect_source_text_info()
        self.ui.statusbar.showMessage(f"File: {filename[0]}")

    def get_current_config(self):
        config = "s2tw"
        if self.ui.rbManual.isChecked():
            config = self.ui.cbManual.currentText().split(' ')[0]
        else:
            if self.ui.rbS2t.isChecked():
                if self.ui.rbHK.isChecked():
                    config = "s2hk"
                elif self.ui.rbStd.isChecked():
                    config = "s2t"
                else:
                    config = "s2twp" if self.ui.cbZhTw.isChecked() else "s2tw"
            elif self.ui.rbT2s.isChecked():
                if self.ui.rbHK.isChecked():
                    config = "hk2s"
                elif self.ui.rbStd.isChecked():
                    config = "t2s"
                else:
                    config = "tw2sp" if self.ui.cbZhTw.isChecked() else "tw2s"
        return config

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
                for index in range(self.ui.listSource.count()):
                    file_path: str = self.ui.listSource.item(index).text()
                    file_base_path, file_extension = os.path.splitext(file_path)
                    ext = file_extension.lower().lstrip(".")

                    if self.ui.actionConvert_filename.isChecked():
                        file_base_path = self.converter.convert(file_path, is_punctuation)

                    if os.path.exists(file_path):
                        output_filename = self.ui.lineEditDir.text() + f"/{os.path.basename(file_base_path)}_{config}.{ext}"

                        if ext in OFFICE_FORMATS:
                            # Convert Office documents
                            success, message = convert_office_doc(file_path, output_filename, ext, self.converter,
                                                                  is_punctuation, True)
                            if success:
                                self.ui.tbPreview.appendPlainText(
                                    f"{index + 1}: {output_filename} -> {message} -> Done.")
                            else:
                                self.ui.tbPreview.appendPlainText(f"{index + 1}: {file_path} -> Skip: {message}.")
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
                                self.ui.tbPreview.appendPlainText(f"{index + 1}: {file_path} -> Skip: Not text file.")
                    else:
                        self.ui.tbPreview.appendPlainText(f"{index + 1}: {file_path} -> File not found.")
                self.ui.statusbar.showMessage("Process completed")

    def update_source_code(self, text_code):
        if text_code == 1:
            self.ui.lblSourceCode.setText("zh-Hant (繁体)")
            self.ui.rbT2s.setChecked(True)
        elif text_code == 2:
            self.ui.lblSourceCode.setText("zh-Hans (简体)")
            self.ui.rbS2t.setChecked(True)
        else:
            self.ui.lblSourceCode.setText("Non-zh (其它)")

    def btn_savefile_click(self):
        filename = QFileDialog.getSaveFileName(
            self,
            "Save Text File",
            "./File.txt",
            "Text File (*.txt);;All Files (*.*)")

        if not filename[0]:
            return

        with open(filename[0], "w", encoding="utf-8") as f:
            f.write(self.ui.tbDestination.toPlainText())
        self.ui.statusbar.showMessage(f"File saved to {filename[0]}")

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
        for file in files:
            # Check if the file path is not already in the list box
            if not self.file_path_exists(file):
                self.ui.listSource.addItem(file)

    def file_path_exists(self, file_path):
        # Check if the file path is already in the list box
        for index in range(self.ui.listSource.count()):
            item = self.ui.listSource.item(index)
            if item.text() == file_path:
                return True
        return False

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
            except UnicodeDecodeError:
                contents = ""  # Already initialized, but good to explicitly handle for clarity
                self.ui.statusbar.showMessage(f"{file_path}: Not a valid text file.")
            except FileNotFoundError:  # Add this to handle non-existent files
                contents = ""
                self.ui.statusbar.showMessage(f"{file_path}: File not found.")
            except Exception as e:  # Catch other potential errors
                contents = ""
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


def get_text_code(text):
    return OpenCC().zho_check(text)


if __name__ == "__main__":
    app = QApplication()
    app.setStyle("WindowsVista")
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
