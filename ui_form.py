# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox,
    QComboBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLayout, QLineEdit, QListWidgetItem,
    QMainWindow, QMenu, QMenuBar, QPlainTextEdit,
    QPushButton, QRadioButton, QSizePolicy, QSpacerItem,
    QStatusBar, QTabWidget, QVBoxLayout, QWidget)

from widgets.custom_widgets import (DragListWidget, TextEditWidget)
import resource_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1155, 831)
        icon = QIcon()
        icon.addFile(u":/images/resource/openccpurepygui.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        icon1 = QIcon()
        icon1.addFile(u":/images/resource/exit.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionExit.setIcon(icon1)
        self.actionExit.setMenuRole(QAction.MenuRole.NoRole)
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        icon2 = QIcon()
        icon2.addFile(u":/images/resource/information.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionAbout.setIcon(icon2)
        self.actionAbout.setMenuRole(QAction.MenuRole.NoRole)
        self.actionConvertFilename = QAction(MainWindow)
        self.actionConvertFilename.setObjectName(u"actionConvertFilename")
        self.actionConvertFilename.setCheckable(True)
        self.actionAddPdfPageHeader = QAction(MainWindow)
        self.actionAddPdfPageHeader.setObjectName(u"actionAddPdfPageHeader")
        self.actionAddPdfPageHeader.setCheckable(True)
        self.actionCompactPdfText = QAction(MainWindow)
        self.actionCompactPdfText.setObjectName(u"actionCompactPdfText")
        self.actionCompactPdfText.setCheckable(True)
        self.actionUsePdfTextExtractWorker = QAction(MainWindow)
        self.actionUsePdfTextExtractWorker.setObjectName(u"actionUsePdfTextExtractWorker")
        self.actionUsePdfTextExtractWorker.setCheckable(True)
        self.actionUsePdfTextExtractWorker.setChecked(True)
        self.actionAutoReflow = QAction(MainWindow)
        self.actionAutoReflow.setObjectName(u"actionAutoReflow")
        self.actionAutoReflow.setCheckable(True)
        self.actionAutoReflow.setChecked(True)
        self.actionSelectEditorFont = QAction(MainWindow)
        self.actionSelectEditorFont.setObjectName(u"actionSelectEditorFont")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_config = QHBoxLayout()
        self.horizontalLayout_config.setObjectName(u"horizontalLayout_config")
        self.horizontalLayout_config.setContentsMargins(10, -1, 10, -1)
        self.rbS2t = QRadioButton(self.centralwidget)
        self.buttonGroup_config = QButtonGroup(MainWindow)
        self.buttonGroup_config.setObjectName(u"buttonGroup_config")
        self.buttonGroup_config.addButton(self.rbS2t)
        self.rbS2t.setObjectName(u"rbS2t")
        font = QFont()
        font.setPointSize(12)
        self.rbS2t.setFont(font)
        self.rbS2t.setChecked(True)

        self.horizontalLayout_config.addWidget(self.rbS2t)

        self.rbT2s = QRadioButton(self.centralwidget)
        self.buttonGroup_config.addButton(self.rbT2s)
        self.rbT2s.setObjectName(u"rbT2s")
        self.rbT2s.setFont(font)

        self.horizontalLayout_config.addWidget(self.rbT2s)

        self.rbManual = QRadioButton(self.centralwidget)
        self.buttonGroup_config.addButton(self.rbManual)
        self.rbManual.setObjectName(u"rbManual")
        self.rbManual.setMaximumSize(QSize(150, 16777215))
        self.rbManual.setFont(font)

        self.horizontalLayout_config.addWidget(self.rbManual)

        self.cbManual = QComboBox(self.centralwidget)
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.addItem("")
        self.cbManual.setObjectName(u"cbManual")
        self.cbManual.setMaximumSize(QSize(180, 16777215))
        font1 = QFont()
        font1.setPointSize(11)
        self.cbManual.setFont(font1)

        self.horizontalLayout_config.addWidget(self.cbManual)


        self.verticalLayout_3.addLayout(self.horizontalLayout_config)

        self.horizontalLayout_config_region = QHBoxLayout()
        self.horizontalLayout_config_region.setObjectName(u"horizontalLayout_config_region")
        self.horizontalLayout_config_region.setContentsMargins(10, -1, 10, -1)
        self.horizontalLayout_region = QHBoxLayout()
        self.horizontalLayout_region.setObjectName(u"horizontalLayout_region")
        self.rbStd = QRadioButton(self.centralwidget)
        self.buttonGroup_region = QButtonGroup(MainWindow)
        self.buttonGroup_region.setObjectName(u"buttonGroup_region")
        self.buttonGroup_region.addButton(self.rbStd)
        self.rbStd.setObjectName(u"rbStd")
        self.rbStd.setFont(font1)
        self.rbStd.setChecked(True)

        self.horizontalLayout_region.addWidget(self.rbStd)

        self.rbZhTw = QRadioButton(self.centralwidget)
        self.buttonGroup_region.addButton(self.rbZhTw)
        self.rbZhTw.setObjectName(u"rbZhTw")
        self.rbZhTw.setFont(font1)
        self.rbZhTw.setChecked(False)

        self.horizontalLayout_region.addWidget(self.rbZhTw)

        self.rbHK = QRadioButton(self.centralwidget)
        self.buttonGroup_region.addButton(self.rbHK)
        self.rbHK.setObjectName(u"rbHK")
        self.rbHK.setFont(font1)

        self.horizontalLayout_region.addWidget(self.rbHK)


        self.horizontalLayout_config_region.addLayout(self.horizontalLayout_region)

        self.horizontalLayout_idioms = QHBoxLayout()
        self.horizontalLayout_idioms.setObjectName(u"horizontalLayout_idioms")
        self.cbZhTw = QCheckBox(self.centralwidget)
        self.cbZhTw.setObjectName(u"cbZhTw")
        self.cbZhTw.setEnabled(False)
        self.cbZhTw.setFont(font1)

        self.horizontalLayout_idioms.addWidget(self.cbZhTw)

        self.cbPunct = QCheckBox(self.centralwidget)
        self.cbPunct.setObjectName(u"cbPunct")
        self.cbPunct.setFont(font1)
        self.cbPunct.setChecked(True)

        self.horizontalLayout_idioms.addWidget(self.cbPunct)


        self.horizontalLayout_config_region.addLayout(self.horizontalLayout_idioms)

        self.horizontalLayout_config_region.setStretch(0, 3)
        self.horizontalLayout_config_region.setStretch(1, 2)

        self.verticalLayout_3.addLayout(self.horizontalLayout_config_region)

        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setFont(font)
        self.tabWidget.setTabShape(QTabWidget.TabShape.Rounded)
        self.tabWidget.setIconSize(QSize(20, 20))
        self.tab_main = QWidget()
        self.tab_main.setObjectName(u"tab_main")
        self.verticalLayout_2 = QVBoxLayout(self.tab_main)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_textBox = QHBoxLayout()
        self.horizontalLayout_textBox.setSpacing(12)
        self.horizontalLayout_textBox.setObjectName(u"horizontalLayout_textBox")
        self.horizontalLayout_textBox.setContentsMargins(0, 0, 0, 0)
        self.frameSource = QFrame(self.tab_main)
        self.frameSource.setObjectName(u"frameSource")
        self.frameSource.setFrameShape(QFrame.Shape.Box)
        self.frameSource.setFrameShadow(QFrame.Shadow.Sunken)
        self.frameSource.setLineWidth(2)
        self.verticalLayout_frame = QVBoxLayout(self.frameSource)
        self.verticalLayout_frame.setSpacing(0)
        self.verticalLayout_frame.setObjectName(u"verticalLayout_frame")
        self.verticalLayout_frame.setContentsMargins(0, 0, 0, 0)
        self.tbSource = TextEditWidget(self.frameSource)
        self.tbSource.setObjectName(u"tbSource")
        self.tbSource.setToolTipDuration(-1)
        self.tbSource.setMidLineWidth(0)
        font2 = QFont()
        font2.setFamilies([u"Microsoft YaHei"])
        font2.setPointSize(13)
        font2.setBold(False)
        self.tbSource.setFont(font2)
        self.tbSource.setFrameShape(QFrame.Shape.NoFrame)
        self.tbSource.setLineWidth(0)
        self.tbSource.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self.verticalLayout_frame.addWidget(self.tbSource)

        self.lineframeFooter = QFrame(self.frameSource)
        self.lineframeFooter.setObjectName(u"lineframeFooter")
        self.lineframeFooter.setFrameShape(QFrame.Shape.HLine)
        self.lineframeFooter.setFrameShadow(QFrame.Shadow.Sunken)
        self.lineframeFooter.setLineWidth(2)

        self.verticalLayout_frame.addWidget(self.lineframeFooter)

        self.verticalLayout_frameSourceFooter = QVBoxLayout()
        self.verticalLayout_frameSourceFooter.setSpacing(6)
        self.verticalLayout_frameSourceFooter.setObjectName(u"verticalLayout_frameSourceFooter")
        self.verticalLayout_frameSourceFooter.setContentsMargins(8, 8, 8, 8)
        self.horizontalLayout_source_info = QHBoxLayout()
        self.horizontalLayout_source_info.setObjectName(u"horizontalLayout_source_info")
        self.horizontalLayout_source_info.setContentsMargins(0, 0, 0, 0)
        self.lblSource = QLabel(self.frameSource)
        self.lblSource.setObjectName(u"lblSource")
        font3 = QFont()
        font3.setPointSize(10)
        font3.setBold(True)
        self.lblSource.setFont(font3)
        self.lblSource.setFrameShape(QFrame.Shape.NoFrame)
        self.lblSource.setFrameShadow(QFrame.Shadow.Plain)
        self.lblSource.setLineWidth(1)
        self.lblSource.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.lblSource.setMargin(3)

        self.horizontalLayout_source_info.addWidget(self.lblSource)

        self.lblSourceCode = QLabel(self.frameSource)
        self.lblSourceCode.setObjectName(u"lblSourceCode")
        self.lblSourceCode.setMinimumSize(QSize(0, 28))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblSourceCode.sizePolicy().hasHeightForWidth())
        self.lblSourceCode.setSizePolicy(sizePolicy)
        font4 = QFont()
        font4.setPointSize(12)
        font4.setBold(False)
        self.lblSourceCode.setFont(font4)
        self.lblSourceCode.setMargin(1)

        self.horizontalLayout_source_info.addWidget(self.lblSourceCode)

        self.horizontalSpacer_source_info = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_source_info.addItem(self.horizontalSpacer_source_info)

        self.lblCharCount = QLabel(self.frameSource)
        self.lblCharCount.setObjectName(u"lblCharCount")
        sizePolicy.setHeightForWidth(self.lblCharCount.sizePolicy().hasHeightForWidth())
        self.lblCharCount.setSizePolicy(sizePolicy)
        font5 = QFont()
        font5.setPointSize(10)
        font5.setBold(False)
        self.lblCharCount.setFont(font5)
        self.lblCharCount.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.lblCharCount.setMargin(1)

        self.horizontalLayout_source_info.addWidget(self.lblCharCount)

        self.horizontalLayout_source_info.setStretch(2, 1)

        self.verticalLayout_frameSourceFooter.addLayout(self.horizontalLayout_source_info)

        self.horizontalLayout_source = QHBoxLayout()
        self.horizontalLayout_source.setSpacing(5)
        self.horizontalLayout_source.setObjectName(u"horizontalLayout_source")
        self.horizontalLayout_source.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.horizontalLayout_source.setContentsMargins(0, 0, 0, 0)
        self.btnReflow = QPushButton(self.frameSource)
        self.btnReflow.setObjectName(u"btnReflow")
        self.btnReflow.setMinimumSize(QSize(0, 28))
        icon3 = QIcon()
        icon3.addFile(u":/images/resource/icons8-refresh-48.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btnReflow.setIcon(icon3)
        self.btnReflow.setMaximumSize(QSize(30, 16777215))
        self.btnReflow.setFont(font3)
        self.btnReflow.setIconSize(QSize(18, 18))

        self.horizontalLayout_source.addWidget(self.btnReflow)

        self.btnNormCompat = QPushButton(self.frameSource)
        self.btnNormCompat.setObjectName(u"btnNormCompat")
        self.btnNormCompat.setMinimumSize(QSize(0, 28))
        self.btnNormCompat.setMaximumSize(QSize(30, 16777215))
        self.btnNormCompat.setFont(font3)
        self.btnNormCompat.setIconSize(QSize(16, 16))

        self.horizontalLayout_source.addWidget(self.btnNormCompat)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_source.addItem(self.horizontalSpacer_3)

        self.btnClearTbSource = QPushButton(self.frameSource)
        self.btnClearTbSource.setObjectName(u"btnClearTbSource")
        self.btnClearTbSource.setMinimumSize(QSize(0, 28))
        self.btnClearTbSource.setMaximumSize(QSize(30, 16777215))
        self.btnClearTbSource.setFont(font3)

        self.horizontalLayout_source.addWidget(self.btnClearTbSource)

        self.btnPaste = QPushButton(self.frameSource)
        self.btnPaste.setObjectName(u"btnPaste")
        self.btnPaste.setMinimumSize(QSize(0, 28))
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.btnPaste.sizePolicy().hasHeightForWidth())
        self.btnPaste.setSizePolicy(sizePolicy1)
        self.btnPaste.setFont(font5)

        self.horizontalLayout_source.addWidget(self.btnPaste)

        self.horizontalLayout_source.setStretch(2, 1)

        self.verticalLayout_frameSourceFooter.addLayout(self.horizontalLayout_source)


        self.verticalLayout_frame.addLayout(self.verticalLayout_frameSourceFooter)

        self.verticalLayout_frame.setStretch(0, 1)

        self.horizontalLayout_textBox.addWidget(self.frameSource)

        self.frameDestination = QFrame(self.tab_main)
        self.frameDestination.setObjectName(u"frameDestination")
        self.frameDestination.setFrameShape(QFrame.Shape.Box)
        self.frameDestination.setFrameShadow(QFrame.Shadow.Sunken)
        self.frameDestination.setLineWidth(2)
        self.verticalLayout_frameDestination = QVBoxLayout(self.frameDestination)
        self.verticalLayout_frameDestination.setSpacing(0)
        self.verticalLayout_frameDestination.setObjectName(u"verticalLayout_frameDestination")
        self.verticalLayout_frameDestination.setContentsMargins(0, 0, 0, 0)
        self.tbDestination = QPlainTextEdit(self.frameDestination)
        self.tbDestination.setObjectName(u"tbDestination")
        self.tbDestination.setAcceptDrops(False)
        self.tbDestination.setMidLineWidth(0)
        self.tbDestination.setUndoRedoEnabled(False)
        self.tbDestination.setReadOnly(True)
        self.tbDestination.setFont(font2)
        self.tbDestination.setFrameShape(QFrame.Shape.NoFrame)
        self.tbDestination.setLineWidth(0)
        self.tbDestination.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self.verticalLayout_frameDestination.addWidget(self.tbDestination)

        self.lineframeDestinationFooter = QFrame(self.frameDestination)
        self.lineframeDestinationFooter.setObjectName(u"lineframeDestinationFooter")
        self.lineframeDestinationFooter.setFrameShape(QFrame.Shape.HLine)
        self.lineframeDestinationFooter.setFrameShadow(QFrame.Shadow.Sunken)
        self.lineframeDestinationFooter.setLineWidth(2)

        self.verticalLayout_frameDestination.addWidget(self.lineframeDestinationFooter)

        self.verticalLayout_frameDestinationFooter = QVBoxLayout()
        self.verticalLayout_frameDestinationFooter.setSpacing(6)
        self.verticalLayout_frameDestinationFooter.setObjectName(u"verticalLayout_frameDestinationFooter")
        self.verticalLayout_frameDestinationFooter.setContentsMargins(8, 8, 8, 8)
        self.horizontalLayout_destination_info = QHBoxLayout()
        self.horizontalLayout_destination_info.setObjectName(u"horizontalLayout_destination_info")
        self.horizontalLayout_destination_info.setContentsMargins(0, 0, 0, 0)
        self.lblDestination = QLabel(self.frameDestination)
        self.lblDestination.setObjectName(u"lblDestination")
        self.lblDestination.setFont(font3)
        self.lblDestination.setFrameShape(QFrame.Shape.NoFrame)
        self.lblDestination.setFrameShadow(QFrame.Shadow.Plain)
        self.lblDestination.setLineWidth(1)
        self.lblDestination.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.lblDestination.setMargin(3)
        self.lblDestination.setIndent(-1)

        self.horizontalLayout_destination_info.addWidget(self.lblDestination)

        self.lblDestinationCode = QLabel(self.frameDestination)
        self.lblDestinationCode.setObjectName(u"lblDestinationCode")
        self.lblDestinationCode.setMinimumSize(QSize(0, 28))
        sizePolicy.setHeightForWidth(self.lblDestinationCode.sizePolicy().hasHeightForWidth())
        self.lblDestinationCode.setSizePolicy(sizePolicy)
        self.lblDestinationCode.setFont(font4)
        self.lblDestinationCode.setMargin(1)

        self.horizontalLayout_destination_info.addWidget(self.lblDestinationCode)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_destination_info.addItem(self.horizontalSpacer_5)

        self.horizontalLayout_destination_info.setStretch(2, 1)

        self.verticalLayout_frameDestinationFooter.addLayout(self.horizontalLayout_destination_info)

        self.horizontalLayout_deatination = QHBoxLayout()
        self.horizontalLayout_deatination.setSpacing(5)
        self.horizontalLayout_deatination.setObjectName(u"horizontalLayout_deatination")
        self.horizontalLayout_deatination.setContentsMargins(0, 0, 0, 0)
        self.btnDeTofu = QPushButton(self.frameDestination)
        self.btnDeTofu.setObjectName(u"btnDeTofu")
        self.btnDeTofu.setMinimumSize(QSize(0, 28))
        self.btnDeTofu.setMaximumSize(QSize(30, 16777215))
        self.btnDeTofu.setFont(font3)
        self.btnDeTofu.setIconSize(QSize(18, 18))

        self.horizontalLayout_deatination.addWidget(self.btnDeTofu)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_deatination.addItem(self.horizontalSpacer_4)

        self.btnClearTbDestination = QPushButton(self.frameDestination)
        self.btnClearTbDestination.setObjectName(u"btnClearTbDestination")
        self.btnClearTbDestination.setMinimumSize(QSize(0, 28))
        self.btnClearTbDestination.setMaximumSize(QSize(30, 16777215))
        self.btnClearTbDestination.setFont(font3)

        self.horizontalLayout_deatination.addWidget(self.btnClearTbDestination)

        self.btnCopy = QPushButton(self.frameDestination)
        self.btnCopy.setObjectName(u"btnCopy")
        self.btnCopy.setMinimumSize(QSize(0, 28))
        sizePolicy1.setHeightForWidth(self.btnCopy.sizePolicy().hasHeightForWidth())
        self.btnCopy.setSizePolicy(sizePolicy1)
        self.btnCopy.setFont(font5)

        self.horizontalLayout_deatination.addWidget(self.btnCopy)

        self.horizontalLayout_deatination.setStretch(1, 1)

        self.verticalLayout_frameDestinationFooter.addLayout(self.horizontalLayout_deatination)


        self.verticalLayout_frameDestination.addLayout(self.verticalLayout_frameDestinationFooter)

        self.verticalLayout_frameDestination.setStretch(0, 1)

        self.horizontalLayout_textBox.addWidget(self.frameDestination)

        self.horizontalLayout_textBox.setStretch(0, 1)
        self.horizontalLayout_textBox.setStretch(1, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_textBox)

        icon4 = QIcon()
        icon4.addFile(u":/images/resource/icons8-document-64.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.tabWidget.addTab(self.tab_main, icon4, "")
        self.tab_batch = QWidget()
        self.tab_batch.setObjectName(u"tab_batch")
        self.verticalLayout = QVBoxLayout(self.tab_batch)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_listbox = QHBoxLayout()
        self.horizontalLayout_listbox.setObjectName(u"horizontalLayout_listbox")
        self.listSource = DragListWidget(self.tab_batch)
        self.listSource.setObjectName(u"listSource")
        font6 = QFont()
        font6.setFamilies([u"Segoe UI"])
        font6.setPointSize(12)
        self.listSource.setFont(font6)
        self.listSource.setAcceptDrops(True)
        self.listSource.setFrameShape(QFrame.Shape.Box)
        self.listSource.setLineWidth(2)
        self.listSource.setDragEnabled(True)
        self.listSource.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.listSource.setAlternatingRowColors(True)
        self.listSource.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.listSource.setSortingEnabled(True)

        self.horizontalLayout_listbox.addWidget(self.listSource)

        self.tbPreview = QPlainTextEdit(self.tab_batch)
        self.tbPreview.setObjectName(u"tbPreview")
        self.tbPreview.setFont(font6)
        self.tbPreview.setAcceptDrops(True)
        self.tbPreview.setFrameShape(QFrame.Shape.Box)
        self.tbPreview.setLineWidth(2)
        self.tbPreview.setUndoRedoEnabled(False)
        self.tbPreview.setReadOnly(True)

        self.horizontalLayout_listbox.addWidget(self.tbPreview)


        self.verticalLayout.addLayout(self.horizontalLayout_listbox)

        self.horizontalLayout_listbox_action = QHBoxLayout()
        self.horizontalLayout_listbox_action.setObjectName(u"horizontalLayout_listbox_action")
        self.horizontalLayout_listbox_buttons = QHBoxLayout()
        self.horizontalLayout_listbox_buttons.setObjectName(u"horizontalLayout_listbox_buttons")
        self.btnAdd = QPushButton(self.tab_batch)
        self.btnAdd.setObjectName(u"btnAdd")
        font7 = QFont()
        font7.setPointSize(9)
        font7.setBold(True)
        self.btnAdd.setFont(font7)

        self.horizontalLayout_listbox_buttons.addWidget(self.btnAdd)

        self.btnRemove = QPushButton(self.tab_batch)
        self.btnRemove.setObjectName(u"btnRemove")
        self.btnRemove.setFont(font3)

        self.horizontalLayout_listbox_buttons.addWidget(self.btnRemove)

        self.btnClear = QPushButton(self.tab_batch)
        self.btnClear.setObjectName(u"btnClear")
        self.btnClear.setFont(font3)

        self.horizontalLayout_listbox_buttons.addWidget(self.btnClear)

        self.btnPreview = QPushButton(self.tab_batch)
        self.btnPreview.setObjectName(u"btnPreview")
        font8 = QFont()
        font8.setPointSize(9)
        font8.setBold(False)
        self.btnPreview.setFont(font8)
        icon5 = QIcon()
        icon5.addFile(u":/images/resource/visual.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btnPreview.setIcon(icon5)
        self.btnPreview.setIconSize(QSize(16, 16))

        self.horizontalLayout_listbox_buttons.addWidget(self.btnPreview)


        self.horizontalLayout_listbox_action.addLayout(self.horizontalLayout_listbox_buttons)

        self.horizontalLayout_preview = QHBoxLayout()
        self.horizontalLayout_preview.setObjectName(u"horizontalLayout_preview")
        self.label = QLabel(self.tab_batch)
        self.label.setObjectName(u"label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)
        font9 = QFont()
        font9.setPointSize(10)
        self.label.setFont(font9)
        self.label.setMargin(1)

        self.horizontalLayout_preview.addWidget(self.label)

        self.lineEditDir = QLineEdit(self.tab_batch)
        self.lineEditDir.setObjectName(u"lineEditDir")

        self.horizontalLayout_preview.addWidget(self.lineEditDir)

        self.btnOutDir = QPushButton(self.tab_batch)
        self.btnOutDir.setObjectName(u"btnOutDir")
        sizePolicy1.setHeightForWidth(self.btnOutDir.sizePolicy().hasHeightForWidth())
        self.btnOutDir.setSizePolicy(sizePolicy1)
        self.btnOutDir.setMaximumSize(QSize(30, 16777215))
        self.btnOutDir.setFont(font3)
        icon6 = QIcon()
        icon6.addFile(u":/images/resource/icons8-folder-64.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btnOutDir.setIcon(icon6)
        self.btnOutDir.setIconSize(QSize(18, 18))

        self.horizontalLayout_preview.addWidget(self.btnOutDir)

        self.btnPreviewClear = QPushButton(self.tab_batch)
        self.btnPreviewClear.setObjectName(u"btnPreviewClear")
        sizePolicy1.setHeightForWidth(self.btnPreviewClear.sizePolicy().hasHeightForWidth())
        self.btnPreviewClear.setSizePolicy(sizePolicy1)
        self.btnPreviewClear.setFont(font3)

        self.horizontalLayout_preview.addWidget(self.btnPreviewClear)


        self.horizontalLayout_listbox_action.addLayout(self.horizontalLayout_preview)

        self.horizontalLayout_listbox_action.setStretch(0, 1)
        self.horizontalLayout_listbox_action.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_listbox_action)

        icon7 = QIcon()
        icon7.addFile(u":/images/resource/icons8-documents-64.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.tabWidget.addTab(self.tab_batch, icon7, "")

        self.verticalLayout_3.addWidget(self.tabWidget)

        self.gridLayout_action_main = QGridLayout()
        self.gridLayout_action_main.setObjectName(u"gridLayout_action_main")
        self.gridLayout_action_main.setVerticalSpacing(6)
        self.gridLayout_action_main.setContentsMargins(10, 0, 10, 0)
        self.horizontalLayout_process = QHBoxLayout()
        self.horizontalLayout_process.setObjectName(u"horizontalLayout_process")
        self.btnProcess = QPushButton(self.centralwidget)
        self.btnProcess.setObjectName(u"btnProcess")
        icon8 = QIcon()
        icon8.addFile(u":/images/resource/icons8-start-48.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btnProcess.setIcon(icon8)
        sizePolicy1.setHeightForWidth(self.btnProcess.sizePolicy().hasHeightForWidth())
        self.btnProcess.setSizePolicy(sizePolicy1)
        self.btnProcess.setMinimumSize(QSize(100, 0))
        font10 = QFont()
        font10.setPointSize(14)
        font10.setBold(True)
        self.btnProcess.setFont(font10)
        self.btnProcess.setIconSize(QSize(28, 28))

        self.horizontalLayout_process.addWidget(self.btnProcess)


        self.gridLayout_action_main.addLayout(self.horizontalLayout_process, 0, 1, 1, 1)

        self.horizontalLayout_saveExit = QHBoxLayout()
        self.horizontalLayout_saveExit.setSpacing(10)
        self.horizontalLayout_saveExit.setObjectName(u"horizontalLayout_saveExit")
        self.horizontalLayout_saveExit.setContentsMargins(-1, -1, 3, -1)
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_saveExit.addItem(self.horizontalSpacer_2)

        self.cbSaveTarget = QComboBox(self.centralwidget)
        self.cbSaveTarget.addItem("")
        self.cbSaveTarget.addItem("")
        self.cbSaveTarget.setObjectName(u"cbSaveTarget")
        self.cbSaveTarget.setMinimumSize(QSize(100, 28))
        self.cbSaveTarget.setFont(font9)

        self.horizontalLayout_saveExit.addWidget(self.cbSaveTarget)

        self.btnSaveAs = QPushButton(self.centralwidget)
        self.btnSaveAs.setObjectName(u"btnSaveAs")
        sizePolicy1.setHeightForWidth(self.btnSaveAs.sizePolicy().hasHeightForWidth())
        self.btnSaveAs.setSizePolicy(sizePolicy1)
        self.btnSaveAs.setFont(font9)

        self.horizontalLayout_saveExit.addWidget(self.btnSaveAs)

        self.btnExit = QPushButton(self.centralwidget)
        self.btnExit.setObjectName(u"btnExit")
        sizePolicy1.setHeightForWidth(self.btnExit.sizePolicy().hasHeightForWidth())
        self.btnExit.setSizePolicy(sizePolicy1)
        self.btnExit.setFont(font3)

        self.horizontalLayout_saveExit.addWidget(self.btnExit)

        self.horizontalLayout_saveExit.setStretch(0, 1)

        self.gridLayout_action_main.addLayout(self.horizontalLayout_saveExit, 0, 2, 1, 1)

        self.horizontalLayout_openFile = QHBoxLayout()
        self.horizontalLayout_openFile.setSpacing(5)
        self.horizontalLayout_openFile.setObjectName(u"horizontalLayout_openFile")
        self.btnOpenFile = QPushButton(self.centralwidget)
        self.btnOpenFile.setObjectName(u"btnOpenFile")
        sizePolicy1.setHeightForWidth(self.btnOpenFile.sizePolicy().hasHeightForWidth())
        self.btnOpenFile.setSizePolicy(sizePolicy1)
        self.btnOpenFile.setFont(font9)

        self.horizontalLayout_openFile.addWidget(self.btnOpenFile)

        self.lblFilename = QLabel(self.centralwidget)
        self.lblFilename.setObjectName(u"lblFilename")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.lblFilename.sizePolicy().hasHeightForWidth())
        self.lblFilename.setSizePolicy(sizePolicy3)
        self.lblFilename.setFont(font9)
        self.lblFilename.setMargin(1)

        self.horizontalLayout_openFile.addWidget(self.lblFilename)

        self.horizontalSpacer_1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_openFile.addItem(self.horizontalSpacer_1)

        self.horizontalLayout_openFile.setStretch(1, 1)

        self.gridLayout_action_main.addLayout(self.horizontalLayout_openFile, 0, 0, 1, 1)

        self.gridLayout_action_main.setColumnStretch(0, 1)
        self.gridLayout_action_main.setColumnStretch(1, 1)
        self.gridLayout_action_main.setColumnStretch(2, 1)

        self.verticalLayout_3.addLayout(self.gridLayout_action_main)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1155, 26))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        self.menuSettings = QMenu(self.menubar)
        self.menuSettings.setObjectName(u"menuSettings")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        self.statusbar.setFont(font9)
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionAbout)
        self.menuSettings.addAction(self.actionConvertFilename)
        self.menuSettings.addSeparator()
        self.menuSettings.addAction(self.actionAddPdfPageHeader)
        self.menuSettings.addAction(self.actionCompactPdfText)
        self.menuSettings.addAction(self.actionAutoReflow)
        self.menuSettings.addSeparator()
        self.menuSettings.addAction(self.actionSelectEditorFont)
        self.menuSettings.addSeparator()
        self.menuSettings.addAction(self.actionUsePdfTextExtractWorker)

        self.retranslateUi(MainWindow)

        self.cbManual.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)
        self.cbSaveTarget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"OpenccPurepyGui", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.actionConvertFilename.setText(QCoreApplication.translate("MainWindow", u"Convert Filename", None))
#if QT_CONFIG(tooltip)
        self.actionConvertFilename.setToolTip(QCoreApplication.translate("MainWindow", u"Convert Filename in batch conversion", None))
#endif // QT_CONFIG(tooltip)
        self.actionAddPdfPageHeader.setText(QCoreApplication.translate("MainWindow", u"Add PDF Page Header", None))
        self.actionCompactPdfText.setText(QCoreApplication.translate("MainWindow", u"Compact PDF Text", None))
        self.actionUsePdfTextExtractWorker.setText(QCoreApplication.translate("MainWindow", u"Use  PDF Text Extract Worker", None))
        self.actionAutoReflow.setText(QCoreApplication.translate("MainWindow", u"Auto-Reflow PDF Text", None))
        self.actionSelectEditorFont.setText(QCoreApplication.translate("MainWindow", u"Select Editor Font ...", None))
        self.rbS2t.setText(QCoreApplication.translate("MainWindow", u"zh-Hans \uff08\u7b80\uff09 To zh-Hant \uff08\u7e41\uff09", None))
        self.rbT2s.setText(QCoreApplication.translate("MainWindow", u"zh-Hant \uff08\u7e41\uff09 To zh-Hans \uff08\u7b80\uff09", None))
        self.rbManual.setText(QCoreApplication.translate("MainWindow", u"Manual (\u81ea\u5b9a\u4e49) :", None))
        self.cbManual.setItemText(0, QCoreApplication.translate("MainWindow", u"s2t (\u7b80 -> \u7e41)", None))
        self.cbManual.setItemText(1, QCoreApplication.translate("MainWindow", u"s2tw (\u7b80 -> \u7e41/\u53f0)", None))
        self.cbManual.setItemText(2, QCoreApplication.translate("MainWindow", u"s2twp (\u7b80 -> \u7e41/\u53f0/\u60ef)", None))
        self.cbManual.setItemText(3, QCoreApplication.translate("MainWindow", u"s2hk (\u7b80 -> \u7e41/\u6e2f)", None))
        self.cbManual.setItemText(4, QCoreApplication.translate("MainWindow", u"s2hkp (\u7b80 -> \u7e41/\u6e2f/\u60ef)", None))
        self.cbManual.setItemText(5, QCoreApplication.translate("MainWindow", u"t2s (\u7e41 -> \u7b80)", None))
        self.cbManual.setItemText(6, QCoreApplication.translate("MainWindow", u"t2tw (\u7e41 -> \u7e41/\u53f0)", None))
        self.cbManual.setItemText(7, QCoreApplication.translate("MainWindow", u"t2twp (\u7e41 -> \u7e41/\u53f0/\u60ef)", None))
        self.cbManual.setItemText(8, QCoreApplication.translate("MainWindow", u"t2hk (\u7e41 -> \u7e41/\u6e2f)", None))
        self.cbManual.setItemText(9, QCoreApplication.translate("MainWindow", u"t2hkp (\u7e41 -> \u7e41/\u6e2f/\u60ef)", None))
        self.cbManual.setItemText(10, QCoreApplication.translate("MainWindow", u"tw2s (\u7e41/\u53f0 -> \u7b80)", None))
        self.cbManual.setItemText(11, QCoreApplication.translate("MainWindow", u"tw2sp (\u7e41/\u53f0 -> \u7b80/\u60ef)", None))
        self.cbManual.setItemText(12, QCoreApplication.translate("MainWindow", u"tw2t (\u7e41/\u53f0 -> \u7e41)", None))
        self.cbManual.setItemText(13, QCoreApplication.translate("MainWindow", u"tw2tp (\u7e41/\u53f0 -> \u7e41/\u60ef)", None))
        self.cbManual.setItemText(14, QCoreApplication.translate("MainWindow", u"hk2s (\u7e41/\u6e2f -> \u7b80)", None))
        self.cbManual.setItemText(15, QCoreApplication.translate("MainWindow", u"hk2sp (\u7e41/\u6e2f -> \u7b80/\u60ef)", None))
        self.cbManual.setItemText(16, QCoreApplication.translate("MainWindow", u"hk2t (\u7e41/\u6e2f -> \u7e41)", None))
        self.cbManual.setItemText(17, QCoreApplication.translate("MainWindow", u"hk2tp (\u7e41/\u6e2f -> \u7e41/\u60ef)", None))
        self.cbManual.setItemText(18, QCoreApplication.translate("MainWindow", u"jp2t (\u65e5/\u65b0 -> \u65e5/\u65e7)", None))
        self.cbManual.setItemText(19, QCoreApplication.translate("MainWindow", u"t2jp (\u65e5/\u65e7 -> \u65e5/\u65b0)", None))

        self.rbStd.setText(QCoreApplication.translate("MainWindow", u"General \uff08\u901a\u7528\u7b80\u7e41\uff09", None))
        self.rbZhTw.setText(QCoreApplication.translate("MainWindow", u"ZH/TW \uff08\u4e2d\u53f0\u7b80\u7e41\uff09", None))
        self.rbHK.setText(QCoreApplication.translate("MainWindow", u"Hong Kong \uff08\u4e2d\u6e2f\u7b80\u7e41\uff09", None))
        self.cbZhTw.setText(QCoreApplication.translate("MainWindow", u"Regional Terms \uff08\u5730\u533a\u7528\u8bed\uff09", None))
        self.cbPunct.setText(QCoreApplication.translate("MainWindow", u"Punctuations \uff08\u6807\u70b9\uff09", None))
#if QT_CONFIG(tooltip)
        self.tbSource.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.lblSource.setText(QCoreApplication.translate("MainWindow", u"Source", None))
        self.lblSourceCode.setText("")
        self.lblCharCount.setText("")
#if QT_CONFIG(tooltip)
        self.btnReflow.setToolTip(QCoreApplication.translate("MainWindow", u"Reflow PDF extracted CJK text", None))
#endif // QT_CONFIG(tooltip)
        self.btnReflow.setText("")
#if QT_CONFIG(tooltip)
        self.btnNormCompat.setToolTip(QCoreApplication.translate("MainWindow", u"Normalize CJK compatibility ideographs and Unicode compatibility characters", None))
#endif // QT_CONFIG(tooltip)
        self.btnNormCompat.setText(QCoreApplication.translate("MainWindow", u"\u2261", None))
#if QT_CONFIG(tooltip)
        self.btnClearTbSource.setToolTip(QCoreApplication.translate("MainWindow", u"Clear source box contents", None))
#endif // QT_CONFIG(tooltip)
        self.btnClearTbSource.setText(QCoreApplication.translate("MainWindow", u"AC", None))
        self.btnPaste.setText(QCoreApplication.translate("MainWindow", u"Paste", None))
        self.lblDestination.setText(QCoreApplication.translate("MainWindow", u"Destination", None))
        self.lblDestinationCode.setText("")
#if QT_CONFIG(tooltip)
        self.btnDeTofu.setToolTip(QCoreApplication.translate("MainWindow", u"Fallback unsupported CJK characters to displayable alternatives to avoid tofu (\u25a1)", None))
#endif // QT_CONFIG(tooltip)
        self.btnDeTofu.setText(QCoreApplication.translate("MainWindow", u"\u8c46", None))
#if QT_CONFIG(tooltip)
        self.btnClearTbDestination.setToolTip(QCoreApplication.translate("MainWindow", u"Clear destination contents", None))
#endif // QT_CONFIG(tooltip)
        self.btnClearTbDestination.setText(QCoreApplication.translate("MainWindow", u"AC", None))
        self.btnCopy.setText(QCoreApplication.translate("MainWindow", u"Copy", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_main), QCoreApplication.translate("MainWindow", u"Single Convert \uff08\u5355\u9879\uff09", None))
        self.btnAdd.setText(QCoreApplication.translate("MainWindow", u"\u2795", None))
        self.btnRemove.setText(QCoreApplication.translate("MainWindow", u"\u2796", None))
        self.btnClear.setText(QCoreApplication.translate("MainWindow", u"AC", None))
        self.btnPreview.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"Output:", None))
        self.lineEditDir.setText(QCoreApplication.translate("MainWindow", u"./output", None))
        self.btnOutDir.setText("")
        self.btnPreviewClear.setText(QCoreApplication.translate("MainWindow", u"AC", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_batch), QCoreApplication.translate("MainWindow", u"Batch Convert \uff08\u6279\u91cf\uff09", None))
        self.btnProcess.setText(QCoreApplication.translate("MainWindow", u"Process", None))
        self.cbSaveTarget.setItemText(0, QCoreApplication.translate("MainWindow", u"Source", None))
        self.cbSaveTarget.setItemText(1, QCoreApplication.translate("MainWindow", u"Destination", None))

        self.btnSaveAs.setText(QCoreApplication.translate("MainWindow", u"Save As", None))
        self.btnExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.btnOpenFile.setText(QCoreApplication.translate("MainWindow", u"Open File", None))
        self.lblFilename.setText("")
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
        self.menuSettings.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
    # retranslateUi

