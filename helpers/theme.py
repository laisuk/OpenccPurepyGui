"""Application themes ported from ZhoConverterQt's Theme implementation."""
from PySide6.QtCore import QObject, QSettings, Qt, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def normalize_mode(value) -> int:
    try:
        mode = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return mode if mode in (0, 1, 2) else 0


def explicit_palette(dark: bool) -> QPalette:
    palette = QPalette()
    colors = {
        'Window': ('#292929', '#f0f0f0'),
        'WindowText': ('#eeeeee', '#202020'),
        'Base': ('#1e1e1e', '#ffffff'),
        'AlternateBase': ('#333333', '#f5f5f5'),
        'Text': ('#eeeeee', '#202020'),
        'Button': ('#292929', '#f0f0f0'),
        'ButtonText': ('#eeeeee', '#202020'),
        'ToolTipBase': ('#1e1e1e', '#ffffff'),
        'ToolTipText': ('#eeeeee', '#202020'),
        'PlaceholderText': ('#aaaaaa', '#666666'),
        'Highlight': ('#80bfff', '#005a9e'),
        'HighlightedText': ('#101010', '#ffffff'),
        'Accent': ('#80bfff', '#005a9e'),
        'Link': ('#80bfff', '#005a9e'),
        'LinkVisited': ('#d6a8ff', '#703090'),
        'Light': ('#555555', '#ffffff'),
        'Midlight': ('#444444', '#e3e3e3'),
        'Mid': ('#666666', '#a0a0a0'),
        'Dark': ('#181818', '#707070'),
        'Shadow': ('#000000', '#404040'),
    }
    for name, pair in colors.items():
        # Accent was introduced in Qt 6.6.
        role = getattr(QPalette.ColorRole, name, None)
        if role is not None:
            palette.setColor(role, QColor(pair[0 if dark else 1]))
    disabled = QPalette.ColorGroup.Disabled
    for name in ('WindowText', 'Text', 'ButtonText', 'PlaceholderText'):
        palette.setColor(disabled, getattr(QPalette.ColorRole, name),
                         QColor('#aaaaaa' if dark else '#666666'))
    palette.setColor(disabled, QPalette.ColorRole.Highlight,
                     QColor('#505050' if dark else '#d0d0d0'))
    palette.setColor(disabled, QPalette.ColorRole.HighlightedText,
                     QColor('#eeeeee' if dark else '#202020'))
    return palette


class Theme(QObject):
    """One application-owned controller, including queued system theme updates."""

    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app
        self.mode = 0
        hints = app.styleHints()
        if hasattr(hints, 'colorSchemeChanged'):
            hints.colorSchemeChanged.connect(
                self._system_scheme_changed, Qt.ConnectionType.QueuedConnection
            )
        self.apply(QSettings().value('theme/themeMode', 0))

    def _refresh_stylesheets(self):
        # Local QSS caches the palette from when it was installed.
        for widget in self.app.allWidgets():
            sheet = widget.styleSheet()
            if sheet:
                widget.setStyleSheet('')
                widget.setStyleSheet(sheet)

    def apply(self, mode):
        self.mode = normalize_mode(mode)
        hints = self.app.styleHints()
        if self.mode == 0:
            if hasattr(hints, 'unsetColorScheme'):
                hints.unsetColorScheme()
            self.app.setPalette(QPalette())
        else:
            if hasattr(hints, 'setColorScheme'):
                hints.setColorScheme(
                    Qt.ColorScheme.Dark if self.mode == 2 else Qt.ColorScheme.Light
                )
            self.app.setPalette(explicit_palette(self.mode == 2))
        self._refresh_stylesheets()

    @Slot()
    def _system_scheme_changed(self):
        # Qt updates the platform palette after emitting colorSchemeChanged.
        if self.mode == 0:
            self.app.setPalette(QPalette())
            self._refresh_stylesheets()


_theme_controller: Theme | None = None


def initialize() -> Theme:
    global _theme_controller

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        raise RuntimeError("Create QApplication before initializing themes")

    if _theme_controller is None:
        _theme_controller = Theme(app)

    return _theme_controller
