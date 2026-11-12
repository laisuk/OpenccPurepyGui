"""Offscreen checks for theme switching, persistence, and palette propagation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest

from PySide6.QtCore import QEvent, QSettings
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QPlainTextEdit

from helpers.theme import Theme, initialize, normalize_mode
from mainwindow import MainWindow


class ThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                          cls.settings_dir.name)
        cls.app.setOrganizationName('ThemeTests')
        cls.app.setApplicationName('ThemeTests')

    def setUp(self):
        QSettings().clear()
        self.theme = initialize()

    def tearDown(self):
        self.theme.apply(0)
        QSettings().clear()
        self.app.processEvents()

    def test_invalid_settings_and_single_controller(self):
        for value in (-1, 3, 'bad', None):
            self.assertEqual(normalize_mode(value), 0)
            self.theme.apply(value)
            self.assertEqual(self.theme.mode, 0)
        self.assertIs(initialize(), self.theme)

    def test_startup_restores_saved_mode(self):
        for saved, expected in ((2, 2), (1, 1), ('bad', 0), (99, 0)):
            QSettings().setValue('theme/themeMode', saved)
            controller = Theme(self.app)
            try:
                self.assertEqual(controller.mode, expected)
            finally:
                controller.deleteLater()
                self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_palette_propagation_with_local_selection_stylesheet(self):
        editor = QPlainTextEdit()
        sheet = 'QPlainTextEdit { selection-background-color: #B3D7FF; selection-color: #000000; }'
        editor.setStyleSheet(sheet)
        editor.show()
        try:
            for mode, base, text in ((2, '#1e1e1e', '#eeeeee'),
                                     (1, '#ffffff', '#202020')):
                self.theme.apply(mode)
                self.app.processEvents()
                self.assertEqual(editor.palette().color(QPalette.ColorRole.Base).name(), base)
                self.assertEqual(editor.palette().color(QPalette.ColorRole.Text).name(), text)
                self.assertEqual(editor.styleSheet(), sheet)
                self.assertEqual(editor.palette().color(QPalette.ColorRole.Highlight).name(), '#b3d7ff')
            self.theme.apply(0)
            self.app.processEvents()
            self.assertEqual(self.theme.mode, 0)
            self.assertEqual(self.app.palette().resolveMask(), 0)
        finally:
            editor.close()
            editor.deleteLater()

    def test_menu_persistence_and_restore(self):
        window = MainWindow()
        try:
            actions = window.ui.menuTheme.actions()
            self.assertEqual([a.text() for a in actions], ['Follow System', 'Light', 'Dark'])
            self.assertTrue(actions[0].isChecked())
            for mode in (2, 1, 0, 2):
                actions[mode].trigger()
                self.app.processEvents()
                self.assertEqual(self.theme.mode, mode)
                self.assertEqual(QSettings().value('theme/themeMode', type=int), mode)
                self.assertEqual([a.isChecked() for a in actions], [i == mode for i in range(3)])
            restored = MainWindow()
            try:
                self.assertTrue(restored.ui.menuTheme.actions()[2].isChecked())
                self.assertEqual(restored.ui.tbSource.palette().color(QPalette.ColorRole.Base).name(), '#1e1e1e')
            finally:
                restored.close()
                restored.deleteLater()
        finally:
            window.close()
            window.deleteLater()

    def test_system_notification_preserves_explicit_palette(self):
        self.theme.apply(2)
        self.theme._system_scheme_changed()
        self.assertEqual(self.app.palette().color(QPalette.ColorRole.Base).name(), '#1e1e1e')
        self.theme.apply(0)
        self.theme._system_scheme_changed()
        self.assertEqual(self.app.palette().resolveMask(), 0)


if __name__ == '__main__':
    unittest.main()
