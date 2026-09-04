from pathlib import Path
from typing import Iterable, List, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QPlainTextEdit, QListWidget, QAbstractItemView

RECURSIVE_FOLDERS = True
# Optional: restrict types (None = accept all files)
# ACCEPT_EXTENSIONS = None
ACCEPT_EXTENSIONS = {".txt", ".md", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub", ".pdf"}


class TextEditWidget(QPlainTextEdit):
    # Emit file path on file drop; emit "" when plain text is dropped
    fileDropped = Signal(str)
    pdfDropped = Signal(str)
    openXmlDropped = Signal(str)  # docx/odt/epub (handled by MainWindow)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

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
