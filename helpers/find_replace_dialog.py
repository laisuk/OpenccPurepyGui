"""Modeless Find / Replace dialog for a QPlainTextEdit source editor."""
from __future__ import annotations

from PySide6.QtCore import Qt, QRegularExpression, QSettings, QTimer
from PySide6.QtGui import QColor, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
    QTextEdit
)
from typing import Any, cast


class FindReplaceDialog(QDialog):
    HIGHLIGHT_LIMIT = 1000
    COUNT_LIMIT = 100000

    def __init__(self, editor: QPlainTextEdit, parent: QWidget | None = None):
        super().__init__(parent)
        self.editor = editor
        # self.setWindowTitle('Find / Replace')
        self.update_title()
        self.setWindowFlag(Qt.WindowType.Tool)
        self.setModal(False)
        self.setMinimumWidth(470)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        find_row = QHBoxLayout()
        self.find_edit = QLineEdit(self)
        self.find_edit.setPlaceholderText('Find in source…')
        self.find_edit.setClearButtonEnabled(True)
        self.previous_button = QPushButton('↑', self)
        self.next_button = QPushButton('↓', self)
        self.previous_button.setToolTip('Previous match (Shift+F3)')
        self.next_button.setToolTip('Next match (F3 / Enter)')
        self.previous_button.setFixedWidth(35)
        self.next_button.setFixedWidth(35)
        self.counter = QLabel('0 matches', self)
        self.counter.setMinimumWidth(75)
        find_row.addWidget(self.find_edit, 1)
        find_row.addWidget(self.previous_button)
        find_row.addWidget(self.next_button)
        find_row.addWidget(self.counter)
        layout.addLayout(find_row)

        self.replace_row = QWidget(self)
        replace_layout = QHBoxLayout(self.replace_row)
        replace_layout.setContentsMargins(0, 0, 0, 0)
        self.replace_edit = QLineEdit(self.replace_row)
        self.replace_edit.setPlaceholderText('Replace with…')
        self.replace_button = QPushButton('Replace', self.replace_row)
        self.replace_all_button = QPushButton('Replace All', self.replace_row)
        replace_layout.addWidget(self.replace_edit, 1)
        replace_layout.addWidget(self.replace_button)
        replace_layout.addWidget(self.replace_all_button)
        layout.addWidget(self.replace_row)

        options = QHBoxLayout()
        self.match_case = QCheckBox('Match case', self)
        self.whole_words = QCheckBox('Whole words', self)
        self.regex = QCheckBox('Regex', self)
        for checkbox in (self.match_case, self.whole_words, self.regex):
            options.addWidget(checkbox)
        options.addStretch()
        layout.addLayout(options)
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(120)
        self.refresh_timer.timeout.connect(self.refresh)
        self._query_start: QTextCursor | None = None
        self.find_edit.textChanged.connect(self._query_changed)
        for checkbox in (self.match_case, self.whole_words, self.regex):
            checkbox.toggled.connect(self._query_changed)
        self.previous_button.clicked.connect(lambda: self.navigate(True))
        self.next_button.clicked.connect(lambda: self.navigate(False))
        self.replace_button.clicked.connect(self.replace_one)
        self.replace_all_button.clicked.connect(self.replace_all)
        self.find_edit.returnPressed.connect(lambda: self.navigate(False))
        self.editor.document().contentsChanged.connect(self._editor_changed)
        self.editor.cursorPositionChanged.connect(self._editor_changed)

        settings = QSettings()
        self.match_case.setChecked(
            cast(bool, settings.value("findReplace/matchCase", False, type=bool))
        )
        self.whole_words.setChecked(
            cast(bool, settings.value("findReplace/wholeWords", False, type=bool))
        )
        self.regex.setChecked(
            cast(bool, settings.value("findReplace/regex", False, type=bool))
        )
        self.replace_row.hide()

    def set_editor(self, editor: QPlainTextEdit) -> None:
        """Switch the target editor without recreating the dialog."""
        if self.editor is editor:
            return

        self.editor.setExtraSelections([])
        self.editor.document().contentsChanged.disconnect(self._editor_changed)
        self.editor.cursorPositionChanged.disconnect(self._editor_changed)
        self._query_start = None
        self.editor = editor
        self.update_title()
        self.editor.document().contentsChanged.connect(self._editor_changed)
        self.editor.cursorPositionChanged.connect(self._editor_changed)
        self.refresh()

    def update_title(self) -> None:
        target = self.tr(
            "Source" if self.editor.objectName() == "tbSource"
            else "Destination"
        )
        self.setWindowTitle(f"{self.tr('Find / Replace')} - {target}")

    def open_find(self, replace_mode: bool = False) -> None:
        self.replace_row.setVisible(replace_mode)
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            selected = cursor.selectedText()
            if '\u2029' not in selected and '\n' not in selected and len(selected) <= 200:
                self.find_edit.setText(selected)
        self.show()
        self.raise_()
        self.activateWindow()
        self.find_edit.setFocus()
        self.find_edit.selectAll()
        self.navigate(False)
        self.schedule_refresh()

    def _flags(self, backward: bool = False) -> QTextDocument.FindFlag:
        flags = QTextDocument.FindFlag(0)
        if self.match_case.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        if self.whole_words.isChecked():
            flags |= QTextDocument.FindFlag.FindWholeWords
        if backward:
            flags |= QTextDocument.FindFlag.FindBackward
        return flags

    def _expression(self) -> QRegularExpression:
        expression = QRegularExpression(self.find_edit.text())
        if not self.match_case.isChecked():
            expression.setPatternOptions(QRegularExpression.PatternOption.CaseInsensitiveOption)
        return expression

    def _valid_query(self) -> bool:
        return bool(self.find_edit.text()) and (not self.regex.isChecked() or self._expression().isValid())

    def _search(self, start: QTextCursor, backward: bool = False) -> QTextCursor:
        if not self._valid_query():
            return QTextCursor()
        query = self._expression() if self.regex.isChecked() else self.find_edit.text()
        return self.editor.document().find(query, start, self._flags(backward))

    def _is_current_match(self) -> bool:
        current = self.editor.textCursor()
        if not self._valid_query() or not current.hasSelection():
            return False
        start = QTextCursor(self.editor.document())
        start.setPosition(current.selectionStart())
        found = self._search(start)
        return (not found.isNull() and found.selectionStart() == current.selectionStart()
                and found.selectionEnd() == current.selectionEnd())

    def _next_nonempty_match(self, scan: QTextCursor) -> QTextCursor:
        while True:
            found = self._search(scan)
            if found.isNull() or found.hasSelection():
                return found
            scan.setPosition(found.selectionEnd())
            if not scan.movePosition(QTextCursor.MoveOperation.NextCharacter):
                return QTextCursor()

    def navigate(self, backward: bool = False) -> None:
        self._query_start = None
        if not self._valid_query():
            return
        start = self.editor.textCursor()
        start.setPosition(start.selectionStart() if backward else start.selectionEnd())
        found = self._search(start, backward)
        if found.isNull():
            edge = QTextCursor(self.editor.document())
            edge.movePosition(QTextCursor.MoveOperation.End if backward else QTextCursor.MoveOperation.Start)
            found = self._search(edge, backward)
        if not found.isNull():
            self.editor.setTextCursor(found)
            self.editor.ensureCursorVisible()
        self.refresh()

    def replace_one(self) -> None:
        if not self._valid_query() or self.editor.isReadOnly():
            return
        if not self._is_current_match():
            self.navigate(False)
            return
        cursor = self.editor.textCursor()
        cursor.insertText(self.replace_edit.text())
        self.editor.setTextCursor(cursor)
        self.navigate(False)

    def replace_all(self) -> None:
        if not self._valid_query() or self.editor.isReadOnly():
            return
        scan = QTextCursor(self.editor.document())
        scan.movePosition(QTextCursor.MoveOperation.Start)
        edit = QTextCursor(self.editor.document())
        replacement = self.replace_edit.text()
        replaced = 0
        edit.beginEditBlock()
        try:
            while True:
                found = self._next_nonempty_match(scan)
                if found.isNull():
                    break
                start = found.selectionStart()
                found.insertText(replacement)
                replaced += 1
                # QTextCursor positions are UTF-16 offsets, not Python code-point counts.
                scan.setPosition(start + len(replacement.encode('utf-16-le')) // 2)
        finally:
            edit.endEditBlock()
        self.refresh()
        self.message.setText(f'Replaced {replaced} occurrence(s).')

    def _editor_changed(self) -> None:
        if self.isVisible():
            self.schedule_refresh()

    def _query_changed(self, *_args) -> None:
        if self._query_start is None:
            self._query_start = self.editor.textCursor()
            # Keep the same starting point while refining a selected match.
            self._query_start.setPosition(self._query_start.selectionStart())
        self.schedule_refresh()

    def schedule_refresh(self, *_args) -> None:
        if self.isVisible():
            self.refresh_timer.start()

    def refresh(self) -> None:
        valid = self._valid_query()
        if valid and self._query_start is not None:
            start = self._query_start
            self._query_start = None
            found = self._search(start)
            if found.isNull():
                found = self._search(QTextCursor(self.editor.document()))
            if not found.isNull():
                self.editor.setTextCursor(found)
                self.editor.ensureCursorVisible()
        self.previous_button.setEnabled(valid)
        self.next_button.setEnabled(valid)
        self.replace_button.setEnabled(valid and not self.editor.isReadOnly())
        self.replace_all_button.setEnabled(valid and not self.editor.isReadOnly())
        if not valid:
            self.editor.setExtraSelections([])
            self.counter.setText('0 matches')
            self.message.setText(
                self._expression().errorString() if self.regex.isChecked() and self.find_edit.text() else '')
            return

        highlights = []
        scan = QTextCursor(self.editor.document())
        scan.movePosition(QTextCursor.MoveOperation.Start)
        current = self.editor.textCursor()
        total = 0
        current_index = 0
        while total < self.COUNT_LIMIT:
            found = self._next_nonempty_match(scan)
            if found.isNull():
                break
            start, end = found.selectionStart(), found.selectionEnd()
            total += 1
            if current.hasSelection() and start == current.selectionStart() and end == current.selectionEnd():
                current_index = total
            if len(highlights) < self.HIGHLIGHT_LIMIT:
                selection = cast(Any, QTextEdit.ExtraSelection())
                selection.cursor = found
                selection.format.setBackground(QColor(255, 214, 102, 125))
                highlights.append(selection)
            scan.setPosition(end)
        self.editor.setExtraSelections(highlights)
        if total == self.COUNT_LIMIT:
            self.counter.setText(f'{self.COUNT_LIMIT}+ matches')
        elif current_index:
            self.counter.setText(f'{current_index} of {total}')
        else:
            self.counter.setText(f'{total} matches')
        self.message.setText(f'Highlighting the first {self.HIGHLIGHT_LIMIT} matches.'
                             if total > self.HIGHLIGHT_LIMIT else '')

    def _save_settings(self) -> None:
        settings = QSettings()
        settings.setValue('findReplace/matchCase', self.match_case.isChecked())
        settings.setValue('findReplace/wholeWords', self.whole_words.isChecked())
        settings.setValue('findReplace/regex', self.regex.isChecked())

    def closeEvent(self, event) -> None:
        self.refresh_timer.stop()
        self.editor.setExtraSelections([])
        self._save_settings()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F3:
            self.navigate(bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            event.accept()
            return
        super().keyPressEvent(event)
