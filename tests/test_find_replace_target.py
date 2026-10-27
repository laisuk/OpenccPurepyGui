"""Regression coverage for Find/Replace editor targeting (offscreen Qt)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import unittest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from mainwindow import MainWindow


class FindReplaceTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, cls.settings_dir.name)
        cls.app.setOrganizationName('FindReplaceTargetTests')
        cls.app.setApplicationName('FindReplaceTargetTests')

    @classmethod
    def tearDownClass(cls):
        cls.settings_dir.cleanup()

    def setUp(self):
        QSettings().clear()

    def test_destination_focus_while_replace_is_open(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.show()
        source, destination = window.ui.tbSource, window.ui.tbDestination
        source.setPlainText('cat cat')
        destination.setPlainText('cat destination cat')
        window.show_find_replace(True)
        dialog = window._find_replace_dialog
        self.addCleanup(dialog.close)
        dialog.find_edit.setText('cat')
        dialog.replace_edit.setText('dog')
        dialog.refresh()

        window.activateWindow()
        destination.setFocus()
        self.app.processEvents()
        self.assertTrue(destination.hasFocus())
        self.assertIs(dialog.editor, source)
        self.assertTrue(dialog.replace_row.isVisible())
        dialog.replace_one()
        self.assertEqual(source.toPlainText(), 'dog cat')
        dialog.replace_all()
        self.assertEqual(source.toPlainText(), 'dog dog')
        self.assertEqual(destination.toPlainText(), 'cat destination cat')

        # Explicitly opening Find in Destination changes to find-only mode.
        destination.setFocus()
        window.show_find_replace(False)
        self.assertIs(dialog.editor, destination)
        self.assertFalse(dialog.replace_row.isVisible())
        self.assertFalse(dialog.replace_button.isEnabled())
        self.assertFalse(dialog.replace_all_button.isEnabled())
        dialog.replace_one()
        dialog.replace_all()
        self.assertEqual(destination.toPlainText(), 'cat destination cat')

        # Reopening Replace always returns to Source, preserving replacement text.
        window.show_find_replace(True)
        self.assertIs(dialog.editor, source)
        self.assertTrue(dialog.replace_row.isVisible())
        self.assertEqual(dialog.replace_edit.text(), 'dog')


if __name__ == '__main__':
    unittest.main()
