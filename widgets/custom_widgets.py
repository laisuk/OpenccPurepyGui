from pathlib import Path
from typing import Iterable, List, Set

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPainter, QPalette, QPaintEvent
from PySide6.QtWidgets import QPlainTextEdit, QListWidget, QAbstractItemView, QWidget

RECURSIVE_FOLDERS = True
# Optional: restrict types (None = accept all files)
# ACCEPT_EXTENSIONS = None
ACCEPT_EXTENSIONS = {".txt", ".md", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub", ".pdf"}


class _LineNumberArea(QWidget):
    def __init__(self, editor: "TextEditWidget"):
        super().__init__(editor)
        self._editor = editor

    def paintEvent(self, event: QPaintEvent) -> None:
        self._editor.paint_line_numbers(event)


class TextEditWidget(QPlainTextEdit):
    # Emit file path on file drop; emit "" when plain text is dropped
    fileDropped = Signal(str)
    pdfDropped = Signal(str)
    openXmlDropped = Signal(str)  # docx/odt/epub (handled by MainWindow)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._line_numbers_visible = False
        self._line_number_area = _LineNumberArea(self)
        self._line_number_area.hide()
        self.blockCountChanged.connect(self._update_line_number_width)
        self.updateRequest.connect(self._update_line_number_area)
        self._update_line_number_width()

    def setLineNumbersVisible(self, visible):
        visible = bool(visible)
        if self._line_numbers_visible == visible:
            return
        self._line_numbers_visible = visible
        self._line_number_area.setVisible(visible)
        self._update_line_number_width()

    def lineNumbersVisible(self):
        return self._line_numbers_visible

    def _line_number_width(self):
        if not self._line_numbers_visible:
            return 0
        return 8 + self.fontMetrics().horizontalAdvance("9") * len(str(max(1, self.blockCount())))

    def _update_line_number_width(self, *_):
        self.setViewportMargins(self._line_number_width(), 0, 0, 0)
        self._update_line_number_geometry()
        self._line_number_area.update()

    def _update_line_number_geometry(self):
        rect = self.contentsRect()
        self._line_number_area.setGeometry(rect.left(), rect.top(),
                                           self._line_number_width(), rect.height())

    def _update_line_number_area(self, rect, dy):
        if not self._line_numbers_visible:
            return
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(), self._line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_width()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_line_number_geometry()

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "_line_number_area") and event.type() in (
                QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._update_line_number_width()

    def _paint_line_numbers(self, event):
        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), self.palette().alternateBase())
        painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
        painter.setFont(self.font())
        block = self.firstVisibleBlock()
        while block.isValid():
            top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
            if top > event.rect().bottom():
                break
            bottom = top + self.blockBoundingRect(block).height()
            if block.isVisible() and bottom >= event.rect().top():
                painter.drawText(0, round(top), self._line_number_area.width() - 4,
                                 self.fontMetrics().height(),
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                 str(block.blockNumber() + 1))
            block = block.next()
        painter.end()

    def paint_line_numbers(self, event: QPaintEvent) -> None:
        self._paint_line_numbers(event)

    content_filename = ""

    def dragEnterEvent(self, e: QDragEnterEvent):
        mime_data = e.mimeData()
        # Check if the dragged data contains text/uri-list
        if mime_data.hasUrls() or mime_data.hasText():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        mime_data = e.mimeData()
        # Check if the dropped data contains text/uri-list
        if mime_data.hasUrls():
            file_path = mime_data.urls()[0].toLocalFile()
            self.content_filename = file_path
            ext = Path(file_path).suffix.lower()

            if ext == ".pdf":
                self.pdfDropped.emit(file_path)
                e.acceptProposedAction()
                return

            if ext in (".docx", ".odt", ".epub"):
                self.openXmlDropped.emit(file_path)
                e.acceptProposedAction()
                return

            # Read the content of the file and set it to QTextEdit
            # self.load_file(file_path)
            self.fileDropped.emit(file_path)  # <-- emit with path
            e.acceptProposedAction()
        elif mime_data.hasText():
            self.document().setPlainText(mime_data.text())
            self.content_filename = ""
            self.fileDropped.emit("")
            e.acceptProposedAction()

    def load_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.document().setPlainText(content)
        except Exception as e:
            self.document().setPlainText(f"Error loading file: {e}")


def _iter_files_safe(root: Path) -> Iterable[Path]:
    """
    Yield files from `root`:
    - if root is a file: yield it
    - if root is a dir: yield contained files (recursive depending on flag)
    Never yields directories.
    """
    if root.is_file():
        yield root
        return

    if not root.is_dir():
        return

    it = root.rglob("*") if RECURSIVE_FOLDERS else root.glob("*")
    for p in it:
        # Avoid following weird entries; only real files
        try:
            if p.is_file():
                yield p
        except OSError:
            # Permission/IO issues while stating individual files
            continue


class DragListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        # IMPORTANT: disable Qt auto-sorting; we do our own "PDF to bottom" sort
        self.setSortingEnabled(False)

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        # Check if the dragged data contains text/uri-list
        if mime_data.hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        mime = event.mimeData()
        if not mime.hasUrls():
            return

        # IMPORTANT: ensure Qt won't auto-sort against us
        if self.isSortingEnabled():
            self.setSortingEnabled(False)

        # 1) Collect existing items
        all_paths: List[str] = []
        existing: Set[str] = set()
        for i in range(self.count()):
            s = self.item(i).text()
            all_paths.append(s)
            existing.add(s)

        # 2) Add dropped files / expand dropped folders (dedupe)
        for url in mime.urls():
            raw = url.toLocalFile()
            if not raw:
                continue

            root = Path(raw)
            try:
                for f in _iter_files_safe(root):
                    ext = f.suffix.lower()
                    if ACCEPT_EXTENSIONS is not None and ext not in ACCEPT_EXTENSIONS:
                        continue

                    s = str(f)
                    if s not in existing:
                        existing.add(s)
                        all_paths.append(s)
            except OSError:
                continue

        # 3) Sort: PDFs to bottom
        def sort_key(pth: str):
            file_ext = Path(pth).suffix.lower()
            return file_ext == ".pdf", pth.casefold()

        all_paths.sort(key=sort_key)

        # 4) Rebuild list
        self.clear()
        self.addItems(all_paths)

        event.acceptProposedAction()
