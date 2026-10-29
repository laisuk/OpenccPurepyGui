"""Go to Line dialog for a QPlainTextEdit (PySide6)."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QDialog, QInputDialog, QPlainTextEdit, QWidget


class GoToLine:
    """Show a modal line-number prompt and navigate without changing text."""

    @staticmethod
    def show(editor: QPlainTextEdit | None, parent: QWidget | None = None) -> None:
        if editor is None:
            return

        max_line = editor.document().blockCount()
        current_line = editor.textCursor().blockNumber() + 1

        tr = lambda text: QCoreApplication.translate("GoToLine", text)
        target = {
            "tbSource": tr("Source"),
            "tbDestination": tr("Destination"),
        }.get(editor.objectName(), tr("Editor"))

        dialog = QInputDialog(parent if parent is not None else editor)
        dialog.setWindowTitle(tr("Go to Line - %1").replace("%1", target))
        dialog.setLabelText(tr("Line number (1–%1):").replace("%1", str(max_line)))
        dialog.setInputMode(QInputDialog.InputMode.IntInput)
        dialog.setIntRange(1, max_line)
        dialog.setIntStep(1)
        dialog.setIntValue(current_line)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        block = editor.document().findBlockByNumber(dialog.intValue() - 1)
        if not block.isValid():
            return

        editor.setTextCursor(QTextCursor(block))
        editor.setFocus()
        editor.centerCursor()
