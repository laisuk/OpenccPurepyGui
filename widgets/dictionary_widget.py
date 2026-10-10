"""Custom dictionary configuration, persisted separately from active dictionaries."""
from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QHeaderView,
    QComboBox, QLineEdit, QPushButton, QFileDialog,
)


class DictionaryWidget(QWidget):
    apply_requested = Signal(list)

    def __init__(self, slots: list[str], parent=None):
        super().__init__(parent)
        self._slots = slots
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("Custom Dictionary", self)
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch()
        self.status = QLabel("Default dictionary")
        self.status.setObjectName("dictionaryStatus")
        self.status.setStyleSheet("""
            QLabel#dictionaryStatus {
                color: #0F6CBD;
                background-color: rgba(15, 108, 189, 20);
                border: 1px solid #0F6CBD;
                border-radius: 12px;
                padding: 4px 12px;
                font-weight: 600;
            }
        """)
        header.addWidget(self.status)
        layout.addLayout(header)
        hint = QLabel(
            "Append merges mappings into a slot; Override replaces its mappings. Rows apply in order.\n"
            "Use UTF-8 text with source and target separated by a TAB; the first target is used.\n"
            "Applying with no configured dictionary files restores the default base dictionary."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("dictionaryRows")
        self.table.setStyleSheet("QTableWidget#dictionaryRows { border: 2px solid #B0B0B0; }")
        self.table.setHorizontalHeaderLabels(["Slot", "Mode", "Dictionary file", "Remove"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        layout.addWidget(self.table)
        self.empty = QLabel("No custom dictionaries configured.")
        layout.addWidget(self.empty)
        actions = QHBoxLayout()
        add = QPushButton("✚ Add Custom Dictionary", self)
        add.setObjectName("dictionaryAdd")
        self.apply_button = QPushButton("✔ Apply to Current Converter", self)
        self.apply_button.setObjectName("dictionaryApply")
        button_font = add.font()
        button_font.setBold(True)
        add.setFont(button_font)
        self.apply_button.setFont(button_font)
        actions.addWidget(add)
        actions.addStretch()
        actions.addWidget(self.apply_button)
        layout.addLayout(actions)
        add.clicked.connect(lambda: self.add_row())
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(self.rows()))
        settings = QSettings()
        count = settings.beginReadArray("dictionary/rows")
        for index in range(count):
            settings.setArrayIndex(index)
            self.add_row(
                settings.value("slot", slots[0], type=str),
                settings.value("mode", "append", type=str),
                settings.value("path", "", type=str),
                save=False,
            )
        settings.endArray()

    def add_row(self, slot=None, mode="append", path="", *, save=True):
        index = self.table.rowCount()
        self.table.insertRow(index)
        slot_box = QComboBox(self.table)
        slot_box.addItems(self._slots)
        slot = slot or self._slots[0]
        if slot_box.findText(slot) < 0:
            slot_box.addItem(slot)
        slot_box.setCurrentText(slot)
        mode_box = QComboBox(self.table)
        mode_box.addItem("Append", "append")
        mode_box.addItem("Override", "override")
        if mode_box.findData(mode) < 0:
            mode_box.addItem(f"Invalid mode ({mode})", mode)
        mode_box.setCurrentIndex(mode_box.findData(mode))
        cell = QWidget(self.table)
        file_layout = QHBoxLayout(cell)
        file_layout.setContentsMargins(0, 0, 0, 0)
        path_edit = QLineEdit(path, cell)
        browse = QPushButton("Browse …", cell)
        file_layout.addWidget(path_edit)
        file_layout.addWidget(browse)
        remove = QPushButton("Remove", self.table)
        remove.setStyleSheet("QPushButton { color: #D32F2F; }")
        for column, widget in enumerate((slot_box, mode_box, cell, remove)):
            self.table.setCellWidget(index, column, widget)
        slot_box.currentTextChanged.connect(self.save_rows)
        mode_box.currentTextChanged.connect(self.save_rows)
        path_edit.textChanged.connect(self.save_rows)
        browse.clicked.connect(lambda: self._browse(path_edit))
        remove.clicked.connect(lambda: self._remove(remove))
        self.empty.hide()
        if save:
            self.save_rows()

    def _browse(self, path_edit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dictionary", path_edit.text(), "Dictionary text (*.txt);;All files (*)"
        )
        if path:
            path_edit.setText(path)

    def _remove(self, button):
        for index in range(self.table.rowCount()):
            if self.table.cellWidget(index, 3) is button:
                self.table.removeRow(index)
                break
        self.empty.setVisible(self.table.rowCount() == 0)
        self.save_rows()

    def rows(self):
        rows: list[dict[str, str]] = []
        for index in range(self.table.rowCount()):
            slot = self.table.cellWidget(index, 0)
            mode = self.table.cellWidget(index, 1)
            file_cell = self.table.cellWidget(index, 2)
            assert isinstance(slot, QComboBox)
            assert isinstance(mode, QComboBox)
            assert isinstance(file_cell, QWidget)
            path = file_cell.findChild(QLineEdit)
            assert isinstance(path, QLineEdit)
            mode_value = mode.currentData()
            assert isinstance(mode_value, str)
            rows.append({"slot": slot.currentText(), "mode": mode_value, "path": path.text()})
        return rows

    def save_rows(self, *_):
        settings = QSettings()
        rows = self.rows()
        settings.beginWriteArray("dictionary/rows", len(rows))
        for index, row in enumerate(rows):
            settings.setArrayIndex(index)
            for key, value in row.items():
                settings.setValue(key, value)
        settings.endArray()

    def set_active_count(self, count):
        self.status.setText(f"Custom dictionary ({count} slots)" if count else "Default dictionary")

    def set_apply_enabled(self, enabled):
        self.apply_button.setEnabled(enabled)
