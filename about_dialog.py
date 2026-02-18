from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)


@dataclass(frozen=True)
class AboutInfo:
    app_name: str
    version: str
    author: str
    year: str
    description: str
    website_text: str = "GitHub"
    website_url: str = ""
    license_text: str = "MIT"
    license_url: str = ""
    details: str = ""  # multi-line


class AboutDialog(QDialog):
    def __init__(self, info: AboutInfo, parent=None, icon: Optional[QIcon] = None) -> None:
        super().__init__(parent)
        self._info = info

        self.setWindowTitle("About")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setModal(True)
        self.setMinimumWidth(480)

        # --- Root layout
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(12)

        # --- Header row (icon + title)
        header = QHBoxLayout()
        header.setSpacing(14)

        icon_label = QLabel()
        icon_label.setFixedSize(64, 64)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        if icon is None:
            # icon = QApplication.windowIcon()
            icon = self.windowIcon()

        pm = icon.pixmap(64, 64) if not icon.isNull() else QPixmap()
        if not pm.isNull():
            icon_label.setPixmap(pm)
        header.addWidget(icon_label, 0)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)

        title = QLabel(f"{info.app_name}")
        title.setObjectName("AboutTitle")
        title_box.addWidget(title)

        subtitle = QLabel(
            f"<b>Version {info.version}</b>  •  © {info.year} {info.author}"
        )
        subtitle.setTextFormat(Qt.TextFormat.RichText)
        subtitle.setObjectName("AboutSubtitle")
        subtitle.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        title_box.addWidget(subtitle)

        desc = QLabel(info.description)
        desc.setWordWrap(True)
        desc.setObjectName("AboutDesc")
        title_box.addWidget(desc)

        header.addLayout(title_box, 1)
        root.addLayout(header)

        # --- Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        # --- Links + Details (rich)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setFrameShape(QFrame.Shape.NoFrame)
        browser.setObjectName("AboutBrowser")
        browser.setMinimumHeight(120)

        links_html = []
        if info.website_url:
            links_html.append(f'<a href="{info.website_url}">{info.website_text}</a>')
        if info.license_url:
            links_html.append(f'<a href="{info.license_url}">{info.license_text}</a>')

        links_line = " • ".join(links_html) if links_html else ""

        details_html = ""
        if info.details.strip():
            # show details in a subtle boxed block
            escaped = (
                info.details
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            details_html = f"""
            <div style="margin-top:10px;">
              <div style="font-weight:600; margin-bottom:6px;">Details</div>
              <pre style="
                margin:0;
                padding:10px 12px;
                border-radius:10px;
                background: rgba(0,0,0,0.03);
                border: 1px solid rgba(0,0,0,0.05);
                white-space: pre-wrap;
                font-family: ui-monospace, Consolas, Menlo, monospace;
              ">{escaped}</pre>
            </div>
            """

        browser.setHtml(f"""
        <div style="line-height:1.35;">
          <div>{links_line}</div>
          {details_html}
        </div>
        """.strip())
        root.addWidget(browser)

        # --- Buttons (Copy Info + OK)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        btn_ok.setText("OK")

        btn_copy = QPushButton("Copy Info")
        btn_copy.clicked.connect(self._copy_info)

        # Put Copy Info to the left of OK
        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_copy)
        btn_row.addStretch(1)
        btn_row.addWidget(buttons)
        root.addLayout(btn_row)

        buttons.accepted.connect(self.accept)

        self._apply_styles()

    def _copy_info(self) -> None:
        text = (
            f"{self._info.app_name} {self._info.version}\n"
            f"© {self._info.year} {self._info.author}\n"
        )
        if self._info.website_url:
            text += f"{self._info.website_text}: {self._info.website_url}\n"
        if self._info.license_url:
            text += f"{self._info.license_text}: {self._info.license_url}\n"
        if self._info.details.strip():
            text += "\n" + self._info.details.strip() + "\n"

        QGuiApplication.clipboard().setText(text)

    def _apply_styles(self) -> None:
        # Lightweight, modern-ish styling without fighting OS theme too much
        self.setStyleSheet("""
        QLabel#AboutTitle {
            font-size: 18px;
            font-weight: 700;
        }
        QLabel#AboutSubtitle {
            color: rgba(0,0,0,0.62);
        }
        QLabel#AboutDesc {
            color: rgba(0,0,0,0.75);
        }
        QTextBrowser#AboutBrowser {
            color: rgba(0,0,0,0.85);
        }
        """)
