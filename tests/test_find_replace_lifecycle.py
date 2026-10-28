"""Offscreen regression checks for selection styling and editor connections."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import unittest

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QWidget
from shiboken6 import delete, isValid

from helpers.find_replace_dialog import FindReplaceDialog
from mainwindow import MainWindow


class CountingEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.style_updates = []

    def setStyleSheet(self, stylesheet):
        self.style_updates.append(stylesheet)
        super().setStyleSheet(stylesheet)


class CountingDialog(FindReplaceDialog):
    def __init__(self, editor, parent=None):
        self.editor_changes = 0
        super().__init__(editor, parent)

    def _editor_changed(self):
        self.editor_changes += 1
        super()._editor_changed()


class FindReplaceLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                          cls.settings_dir.name)
        cls.app.setOrganizationName('FindReplaceLifecycleTests')
        cls.app.setApplicationName('FindReplaceLifecycleTests')

    @classmethod
    def tearDownClass(cls):
        cls.settings_dir.cleanup()

    def setUp(self):
        QSettings().clear()
        self.window = QWidget()
        self.source = CountingEditor(self.window)
        self.source.setObjectName('tbSource')
        self.destination = CountingEditor(self.window)
        self.destination.setObjectName('tbDestination')
        self.source_style = 'color: #123456;'
        self.destination_style = 'QPlainTextEdit { color: #654321; }'
        self.source.setStyleSheet(self.source_style)
        self.destination.setStyleSheet(self.destination_style)
        self.dialog = CountingDialog(self.source, self.window)
        self.addCleanup(lambda: delete(self.window) if isValid(self.window) else None)
        self.addCleanup(lambda: self.dialog.close() if isValid(self.dialog) else None)

    def settle(self):
        for _ in range(5):
            self.app.processEvents()

    def activate_dialog(self):
        self.window.show()
        self.dialog.open_find()
        self.settle()
        self.assertTrue(self.dialog.isActiveWindow())

    def test_activation_switch_close_reopen_and_stylesheet_preservation(self):
        self.assertEqual(self.source.styleSheet(), self.source_style)
        self.activate_dialog()
        self.assertIn('#B3D7FF', self.source.styleSheet())
        self.assertIn('color: #123456;', self.source.styleSheet())
        self.assertEqual(self.destination.styleSheet(), self.destination_style)
        self.dialog.set_editor(self.destination)
        self.assertEqual(self.source.styleSheet(), self.source_style)
        self.assertIn('#B3D7FF', self.destination.styleSheet())
        self.assertIn('color: #654321;', self.destination.styleSheet())
        self.window.activateWindow()
        self.settle()
        self.assertFalse(self.dialog.isActiveWindow())
        self.assertEqual(self.destination.styleSheet(), self.destination_style)
        self.dialog.activateWindow()
        self.settle()
        self.assertIn('#B3D7FF', self.destination.styleSheet())
        self.dialog.close()
        self.assertEqual(self.destination.styleSheet(), self.destination_style)
        self.dialog.open_find()
        self.settle()
        self.assertIn('#B3D7FF', self.destination.styleSheet())
        self.dialog.hide()
        self.assertEqual(self.destination.styleSheet(), self.destination_style)
        self.dialog.open_find()
        self.settle()
        QTest.keyClick(self.dialog.find_edit, Qt.Key.Key_Escape)
        self.assertFalse(self.dialog.isVisible())
        self.assertEqual(self.destination.styleSheet(), self.destination_style)

    def test_same_state_does_not_reapply_stylesheet(self):
        self.activate_dialog()
        updates = len(self.source.style_updates)
        for _ in range(3):
            self.dialog.set_editor(self.source)
            self.dialog._update_selection_style(True)
        self.assertEqual(len(self.source.style_updates), updates)
        self.dialog.hide()
        updates = len(self.source.style_updates)
        self.dialog._update_selection_style(False)
        self.assertEqual(len(self.source.style_updates), updates)

    def test_repeated_switches_have_exactly_one_connection_per_signal(self):
        self.activate_dialog()
        for _ in range(5):
            self.dialog.set_editor(self.destination)
            self.dialog.set_editor(self.source)
            self.dialog.set_editor(self.source)
        self.dialog.refresh_timer.stop()
        self.dialog.editor_changes = 0
        self.destination.document().contentsChanged.emit()
        self.destination.cursorPositionChanged.emit()
        self.assertEqual(self.dialog.editor_changes, 0)
        self.assertFalse(self.dialog.refresh_timer.isActive())
        self.source.document().contentsChanged.emit()
        self.assertEqual(self.dialog.editor_changes, 1)
        self.source.cursorPositionChanged.emit()
        self.assertEqual(self.dialog.editor_changes, 2)
        self.assertTrue(self.dialog.refresh_timer.isActive())

    def test_switch_stops_pending_refresh_and_old_destruction_is_disconnected(self):
        self.activate_dialog()
        self.dialog.schedule_refresh()
        self.assertTrue(self.dialog.refresh_timer.isActive())
        self.dialog.set_editor(self.destination)
        self.assertFalse(self.dialog.refresh_timer.isActive())
        delete(self.source)
        self.assertIs(self.dialog.editor, self.destination)
        self.assertTrue(self.dialog.isVisible())

    def test_target_destruction_closes_safely_and_dialog_can_be_reused(self):
        self.activate_dialog()
        self.dialog.schedule_refresh()
        delete(self.source)
        self.assertIsNone(self.dialog.editor)
        self.assertFalse(self.dialog.isVisible())
        self.assertFalse(self.dialog.refresh_timer.isActive())
        self.dialog.find_edit.setText('cat')
        self.dialog.navigate()
        self.dialog.refresh()
        self.dialog.replace_one()
        self.dialog.replace_all()
        self.dialog.set_editor(self.destination)
        self.dialog.open_find()
        self.settle()
        self.assertIn('#B3D7FF', self.destination.styleSheet())

    def test_dialog_deletion_restores_surviving_editor(self):
        self.activate_dialog()
        self.dialog.deleteLater()
        QCoreApplication.sendPostedEvents(self.dialog, QEvent.Type.DeferredDelete)
        self.assertFalse(isValid(self.dialog))
        self.assertEqual(self.source.styleSheet(), self.source_style)
        self.source.appendPlainText('safe after dialog deletion')

    def test_yellow_highlights_navigation_and_replacement(self):
        self.source.setPlainText('cat cat')
        self.activate_dialog()
        self.dialog.find_edit.setText('cat')
        self.dialog.refresh()
        highlights = self.source.extraSelections()
        self.assertEqual(len(highlights), 2)
        self.assertEqual(highlights[0].format.background().color().getRgb(),
                         (255, 214, 102, 125))
        self.dialog.navigate()
        self.assertEqual(self.source.textCursor().selectionStart(), 4)
        self.dialog.replace_edit.setText('dog')
        self.dialog.replace_all()
        self.assertEqual(self.source.toPlainText(), 'dog dog')
        self.dialog.close()
        self.assertEqual(self.source.styleSheet(), self.source_style)

    def test_mainwindow_shortcuts_share_one_dialog_and_keep_replace_on_source(self):
        window = MainWindow()
        self.addCleanup(lambda: delete(window) if isValid(window) else None)
        window.show()
        source, destination = window.ui.tbSource, window.ui.tbDestination
        source.setPlainText('cat cat')
        destination.setPlainText('cat destination')
        window.activateWindow()
        source.setFocus()
        self.settle()
        QTest.keyClick(source, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        dialog = window._find_replace_dialog
        self.assertIsNotNone(dialog)
        self.assertIs(dialog.editor, source)
        dialog.close()
        window.activateWindow()
        destination.setFocus()
        self.settle()
        QTest.keyClick(destination, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertIs(window._find_replace_dialog, dialog)
        self.assertIs(dialog.editor, destination)
        dialog.close()
        window.activateWindow()
        destination.setFocus()
        self.settle()
        QTest.keyClick(destination, Qt.Key.Key_H, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertIs(window._find_replace_dialog, dialog)
        self.assertIs(dialog.editor, source)
        self.assertTrue(dialog.replace_row.isVisible())
        self.assertEqual(len(window.findChildren(FindReplaceDialog)), 1)
        dialog.find_edit.setText('cat')
        dialog.refresh()
        dialog.navigate(True)  # Wrap to the last match, then F3 wraps to the first.
        dialog.hide()
        window.activateWindow()
        source.setFocus()
        self.settle()
        QTest.keyClick(source, Qt.Key.Key_F3)
        self.assertEqual(source.textCursor().selectionStart(), 0)
        QTest.keyClick(source, Qt.Key.Key_F3, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(source.textCursor().selectionStart(), 4)
        self.assertEqual(len(window.findChildren(FindReplaceDialog)), 1)


if __name__ == '__main__':
    unittest.main()
