"""Offscreen checks for gutter layout, editor behavior, and saved visibility."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from PySide6.QtCore import QSettings, QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent, QTextCursor
from PySide6.QtWidgets import QApplication, QTextEdit
from widgets.custom_widgets import TextEditWidget
from mainwindow import MainWindow


class LineNumberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, cls.settings_dir.name)
        cls.app.setOrganizationName("LineNumberTests")
        cls.app.setApplicationName("LineNumberTests")

    def setUp(self):
        QSettings().clear()

    def test_gutter_layout_and_drops(self):
        editor = TextEditWidget()
        try:
            editor.resize(300, 180)
            editor.show()
            self.app.processEvents()
            self.assertFalse(editor.lineNumbersVisible())
            self.assertEqual(editor.viewportMargins().left(), 0)
            editor.setLineNumbersVisible(True)
            editor.setPlainText("one")
            small = editor.viewportMargins().left()
            editor.setPlainText("line\n" * 110)
            self.app.processEvents()
            normal = editor.viewportMargins().left()
            self.assertGreater(normal, small)
            font = editor.font()
            font.setPointSize(24)
            editor.setFont(font)
            self.assertGreater(editor.viewportMargins().left(), normal)
            editor.verticalScrollBar().setValue(editor.verticalScrollBar().maximum())
            self.app.processEvents()
            self.assertGreater(editor.firstVisibleBlock().blockNumber(), 0)
            self.assertFalse(editor.grab().isNull())
            editor.setPlainText("wrapped words " * 100 + "\nlast")
            self.app.processEvents()
            self.assertEqual(editor.blockCount(), 2)
            self.assertGreater(editor.document().firstBlock().layout().lineCount(), 1)
            selection = QTextEdit.ExtraSelection()
            cursor = editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.movePosition(QTextCursor.MoveOperation.PreviousWord, QTextCursor.MoveMode.KeepAnchor)
            selection.cursor = cursor
            editor.setExtraSelections([selection])
            editor.setTextCursor(cursor)
            editor.ensureCursorVisible()
            editor.resize(180, 180)
            editor.setLineNumbersVisible(False)
            self.assertEqual(editor.viewportMargins().left(), 0)
            self.assertEqual(len(editor.extraSelections()), 1)
            self.assertEqual(editor.textCursor().selectedText(), "last")
            paths = []
            editor.fileDropped.connect(paths.append)
            mime = QMimeData()
            mime.setText("dropped text")
            event = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                               Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            editor.dropEvent(event)
            self.assertTrue(event.isAccepted())
            self.assertEqual(editor.toPlainText(), "dropped text")
            self.assertEqual(editor.content_filename, "")
            self.assertEqual(paths, [""])
            for extension, signal in [("txt", editor.fileDropped), ("pdf", editor.pdfDropped),
                                      ("docx", editor.openXmlDropped)]:
                received = []
                signal.connect(received.append)
                mime = QMimeData()
                path = "C:/example." + extension
                mime.setUrls([QUrl.fromLocalFile(path)])
                event = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                                   Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
                editor.dropEvent(event)
                self.assertTrue(event.isAccepted())
                self.assertEqual(received, [path])
                self.assertEqual(editor.content_filename, path)
        finally:
            editor.close()

    def test_action_and_persistence(self):
        for expected in (False, True):
            window = MainWindow()
            try:
                source, destination = window.ui.tbSource, window.ui.tbDestination
                action = window.ui.actionShowLineNumbers
                self.assertIsInstance(source, TextEditWidget)
                self.assertIsInstance(destination, TextEditWidget)
                self.assertEqual(action.isChecked(), expected)
                self.assertEqual(source.lineNumbersVisible(), expected)
                self.assertEqual(destination.lineNumbersVisible(), expected)
                self.assertTrue(destination.isReadOnly())
                self.assertFalse(destination.acceptDrops())
                action.setChecked(not expected)
                self.assertEqual(source.lineNumbersVisible(), not expected)
                self.assertEqual(destination.lineNumbersVisible(), not expected)
                self.assertEqual(QSettings().value("editor/showLineNumbers", type=bool), not expected)
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
