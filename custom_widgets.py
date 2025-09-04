from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QPlainTextEdit, QListWidget, QAbstractItemView, QListWidgetItem


class TextEditWidget(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    content_filename = ""

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        # Check if the dragged data contains text/uri-list
        if mime_data.hasUrls() or mime_data.hasText():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        mime_data = event.mimeData()
        # Check if the dropped data contains text/uri-list
        if mime_data.hasUrls():
            file_path = mime_data.urls()[0].toLocalFile()
            # Read the content of the file and set it to QTextEdit
            self.load_file(file_path)
            self.content_filename = file_path
        elif mime_data.hasText():
            self.document().setPlainText(mime_data.text())
            self.content_filename = ""

    def load_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.document().setPlainText(content)
        except Exception as e:
            self.document().setPlainText(f"Error loading file: {e}")


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
        self.setSortingEnabled(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        # Check if the dragged data contains text/uri-list
        if mime_data.hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            # Get the list of dropped file URLs
            file_urls = [url.toLocalFile() for url in mime_data.urls()]
            # Check if the file is not already in the list, then add it
            for file_url in file_urls:
                if not self.is_item_in_list(file_url):
                    item = QListWidgetItem(file_url)
                    self.addItem(item)
            event.acceptProposedAction()

    def is_item_in_list(self, item_text):
        # Check if the item with given text is already in the list
        items = self.findItems(item_text, Qt.MatchFlag.MatchExactly)
        return len(items) > 0
