# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
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
    QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidgetItem, QMainWindow, QMenu,
    QMenuBar, QPlainTextEdit, QPushButton, QRadioButton,
    QSizePolicy, QSpacerItem, QStatusBar, QTabWidget,
    QVBoxLayout, QWidget)

from custom_widgets import (DragListWidget, TextEditWidget)
import resource_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(990, 700)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionExit.setMenuRole(QAction.MenuRole.NoRole)
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.actionAbout.setMenuRole(QAction.MenuRole.NoRole)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_config = QHBoxLayout()
        self.horizontalLayout_config.setObjectName(u"horizontalLayout_config")
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
        self.cbManual.setObjectName(u"cbManual")
        self.cbManual.setMaximumSize(QSize(180, 16777215))
        font1 = QFont()
        font1.setPointSize(11)
        self.cbManual.setFont(font1)

        self.horizontalLayout_config.addWidget(self.cbManual)


        self.verticalLayout_3.addLayout(self.horizontalLayout_config)

        self.horizontalLayout_config_region = QHBoxLayout()
        self.horizontalLayout_config_region.setObjectName(u"horizontalLayout_config_region")
        self.horizontalLayout_region = QHBoxLayout()
        self.horizontalLayout_region.setObjectName(u"horizontalLayout_region")
        self.rbStd = QRadioButton(self.centralwidget)
        self.buttonGroup_region = QButtonGroup(MainWindow)
        self.buttonGroup_region.setObjectName(u"buttonGroup_region")
        self.buttonGroup_region.addButton(self.rbStd)
        self.rbStd.setObjectName(u"rbStd")
        font2 = QFont()
        font2.setPointSize(10)
        self.rbStd.setFont(font2)
        self.rbStd.setChecked(True)

        self.horizontalLayout_region.addWidget(self.rbStd)

        self.rbZhTw = QRadioButton(self.centralwidget)
        self.buttonGroup_region.addButton(self.rbZhTw)
        self.rbZhTw.setObjectName(u"rbZhTw")
        self.rbZhTw.setFont(font2)
        self.rbZhTw.setChecked(False)

        self.horizontalLayout_region.addWidget(self.rbZhTw)

        self.rbHK = QRadioButton(self.centralwidget)
        self.buttonGroup_region.addButton(self.rbHK)
        self.rbHK.setObjectName(u"rbHK")
        self.rbHK.setFont(font2)

        self.horizontalLayout_region.addWidget(self.rbHK)


        self.horizontalLayout_config_region.addLayout(self.horizontalLayout_region)

        self.horizontalLayout_idioms = QHBoxLayout()
        self.horizontalLayout_idioms.setObjectName(u"horizontalLayout_idioms")
        self.cbZhTw = QCheckBox(self.centralwidget)
        self.cbZhTw.setObjectName(u"cbZhTw")
        self.cbZhTw.setFont(font2)

        self.horizontalLayout_idioms.addWidget(self.cbZhTw)

        self.cbPunct = QCheckBox(self.centralwidget)
        self.cbPunct.setObjectName(u"cbPunct")
        self.cbPunct.setFont(font2)
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
        self.horizontalLayout_textBox.setSpacing(6)
        self.horizontalLayout_textBox.setObjectName(u"horizontalLayout_textBox")
        self.horizontalLayout_textBox.setContentsMargins(0, 0, 0, -1)
        self.tbSource = TextEditWidget(self.tab_main)
        self.tbSource.setObjectName(u"tbSource")
        font3 = QFont()
        font3.setFamilies([u"Segoe UI"])
        font3.setPointSize(12)
        font3.setBold(False)
        self.tbSource.setFont(font3)
        self.tbSource.setFrameShape(QFrame.Shape.Box)
        self.tbSource.setLineWidth(2)
        self.tbSource.setMidLineWidth(0)
        self.tbSource.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.horizontalLayout_textBox.addWidget(self.tbSource)

        self.tbDestination = QPlainTextEdit(self.tab_main)
        self.tbDestination.setObjectName(u"tbDestination")
        self.tbDestination.setFont(font3)
        self.tbDestination.setAcceptDrops(False)
        self.tbDestination.setFrameShape(QFrame.Shape.Box)
        self.tbDestination.setLineWidth(2)
        self.tbDestination.setMidLineWidth(0)
        self.tbDestination.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tbDestination.setReadOnly(True)

        self.horizontalLayout_textBox.addWidget(self.tbDestination)


        self.verticalLayout_2.addLayout(self.horizontalLayout_textBox)

        self.horizontalLayout_textbox_action = QHBoxLayout()
        self.horizontalLayout_textbox_action.setObjectName(u"horizontalLayout_textbox_action")
        self.horizontalLayout_source = QHBoxLayout()
        self.horizontalLayout_source.setObjectName(u"horizontalLayout_source")
        self.lblSource = QLabel(self.tab_main)
        self.lblSource.setObjectName(u"lblSource")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblSource.sizePolicy().hasHeightForWidth())
        self.lblSource.setSizePolicy(sizePolicy)
        self.lblSource.setMinimumSize(QSize(80, 0))
        self.lblSource.setMaximumSize(QSize(80, 16777215))
        self.lblSource.setFont(font2)
        self.lblSource.setFrameShape(QFrame.Shape.StyledPanel)
        self.lblSource.setLineWidth(1)
        self.lblSource.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_source.addWidget(self.lblSource)

        self.lblSourceCode = QLabel(self.tab_main)
        self.lblSourceCode.setObjectName(u"lblSourceCode")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lblSourceCode.sizePolicy().hasHeightForWidth())
        self.lblSourceCode.setSizePolicy(sizePolicy1)
        self.lblSourceCode.setFont(font2)
        self.lblSourceCode.setMargin(5)

        self.horizontalLayout_source.addWidget(self.lblSourceCode)

        self.lblCharCount = QLabel(self.tab_main)
        self.lblCharCount.setObjectName(u"lblCharCount")
        sizePolicy1.setHeightForWidth(self.lblCharCount.sizePolicy().hasHeightForWidth())
        self.lblCharCount.setSizePolicy(sizePolicy1)
        self.lblCharCount.setFont(font2)
        self.lblCharCount.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_source.addWidget(self.lblCharCount)

        self.btnDetect = QPushButton(self.tab_main)
        self.btnDetect.setObjectName(u"btnDetect")
        sizePolicy.setHeightForWidth(self.btnDetect.sizePolicy().hasHeightForWidth())
        self.btnDetect.setSizePolicy(sizePolicy)
        self.btnDetect.setMaximumSize(QSize(30, 16777215))
        font4 = QFont()
        font4.setPointSize(10)
        font4.setBold(True)
        self.btnDetect.setFont(font4)
        icon = QIcon()
        icon.addFile(u":/images/resource/icons8-refresh-48.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btnDetect.setIcon(icon)

        self.horizontalLayout_source.addWidget(self.btnDetect)

        self.btnClearTbSource = QPushButton(self.tab_main)
        self.btnClearTbSource.setObjectName(u"btnClearTbSource")
        self.btnClearTbSource.setMaximumSize(QSize(30, 16777215))
        self.btnClearTbSource.setFont(font4)

        self.horizontalLayout_source.addWidget(self.btnClearTbSource)

        self.btnPaste = QPushButton(self.tab_main)
        self.btnPaste.setObjectName(u"btnPaste")
        sizePolicy.setHeightForWidth(self.btnPaste.sizePolicy().hasHeightForWidth())
        self.btnPaste.setSizePolicy(sizePolicy)
        self.btnPaste.setFont(font2)

        self.horizontalLayout_source.addWidget(self.btnPaste)


        self.horizontalLayout_textbox_action.addLayout(self.horizontalLayout_source)

        self.horizontalLayout_deatination = QHBoxLayout()
        self.horizontalLayout_deatination.setObjectName(u"horizontalLayout_deatination")
        self.lblDestination = QLabel(self.tab_main)
        self.lblDestination.setObjectName(u"lblDestination")
        sizePolicy.setHeightForWidth(self.lblDestination.sizePolicy().hasHeightForWidth())
        self.lblDestination.setSizePolicy(sizePolicy)
        self.lblDestination.setMinimumSize(QSize(80, 0))
        self.lblDestination.setMaximumSize(QSize(80, 16777215))
        self.lblDestination.setFont(font2)
        self.lblDestination.setFrameShape(QFrame.Shape.StyledPanel)
        self.lblDestination.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_deatination.addWidget(self.lblDestination)

        self.lblDestinationCode = QLabel(self.tab_main)
        self.lblDestinationCode.setObjectName(u"lblDestinationCode")
        self.lblDestinationCode.setFont(font2)
        self.lblDestinationCode.setMargin(5)

        self.horizontalLayout_deatination.addWidget(self.lblDestinationCode)

        self.btnClearTbDestination = QPushButton(self.tab_main)
        self.btnClearTbDestination.setObjectName(u"btnClearTbDestination")
        self.btnClearTbDestination.setMaximumSize(QSize(30, 16777215))
        self.btnClearTbDestination.setFont(font4)

        self.horizontalLayout_deatination.addWidget(self.btnClearTbDestination)

        self.btnCopy = QPushButton(self.tab_main)
        self.btnCopy.setObjectName(u"btnCopy")
        sizePolicy.setHeightForWidth(self.btnCopy.sizePolicy().hasHeightForWidth())
        self.btnCopy.setSizePolicy(sizePolicy)
        self.btnCopy.setFont(font2)

        self.horizontalLayout_deatination.addWidget(self.btnCopy)


        self.horizontalLayout_textbox_action.addLayout(self.horizontalLayout_deatination)

        self.horizontalLayout_textbox_action.setStretch(0, 1)
        self.horizontalLayout_textbox_action.setStretch(1, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_textbox_action)

        icon1 = QIcon()
        icon1.addFile(u":/images/resource/icons8-document-64.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.tabWidget.addTab(self.tab_main, icon1, "")
        self.tab_batch = QWidget()
        self.tab_batch.setObjectName(u"tab_batch")
        self.verticalLayout = QVBoxLayout(self.tab_batch)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_listbox = QHBoxLayout()
        self.horizontalLayout_listbox.setObjectName(u"horizontalLayout_listbox")
        self.listSource = DragListWidget(self.tab_batch)
        self.listSource.setObjectName(u"listSource")
        font5 = QFont()
        font5.setFamilies([u"Segoe UI"])
        font5.setPointSize(12)
        self.listSource.setFont(font5)
        self.listSource.setAcceptDrops(True)
        self.listSource.setFrameShape(QFrame.Shape.Box)
        self.listSource.setLineWidth(2)
        self.listSource.setDragEnabled(True)
        self.listSource.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.listSource.setAlternatingRowColors(True)
        self.listSource.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.listSource.setSortingEnabled(True)

        self.horizontalLayout_listbox.addWidget(self.listSource)

        self.tbPreview = QPlainTextEdit(self.tab_batch)
        self.tbPreview.setObjectName(u"tbPreview")
        self.tbPreview.setFont(font5)
        self.tbPreview.setAcceptDrops(True)
        self.tbPreview.setFrameShape(QFrame.Shape.Box)
        self.tbPreview.setLineWidth(2)

        self.horizontalLayout_listbox.addWidget(self.tbPreview)


        self.verticalLayout.addLayout(self.horizontalLayout_listbox)

        self.horizontalLayout_listbox_action = QHBoxLayout()
        self.horizontalLayout_listbox_action.setObjectName(u"horizontalLayout_listbox_action")
        self.horizontalLayout_listbox_buttons = QHBoxLayout()
        self.horizontalLayout_listbox_buttons.setObjectName(u"horizontalLayout_listbox_buttons")
        self.btnAdd = QPushButton(self.tab_batch)
        self.btnAdd.setObjectName(u"btnAdd")
        self.btnAdd.setFont(font4)

        self.horizontalLayout_listbox_buttons.addWidget(self.btnAdd)

        self.btnRemove = QPushButton(self.tab_batch)
        self.btnRemove.setObjectName(u"btnRemove")
        self.btnRemove.setFont(font4)

        self.horizontalLayout_listbox_buttons.addWidget(self.btnRemove)

        self.btnClear = QPushButton(self.tab_batch)
        self.btnClear.setObjectName(u"btnClear")
        self.btnClear.setFont(font4)

        self.horizontalLayout_listbox_buttons.addWidget(self.btnClear)

        self.btnPreview = QPushButton(self.tab_batch)
        self.btnPreview.setObjectName(u"btnPreview")
        font6 = QFont()
        font6.setPointSize(9)
        font6.setBold(False)
        self.btnPreview.setFont(font6)
        icon2 = QIcon()
        icon2.addFile(u":/images/resource/icons8-preview-48.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btnPreview.setIcon(icon2)
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
        self.label.setFont(font2)
        self.label.setMargin(1)

        self.horizontalLayout_preview.addWidget(self.label)

        self.lineEditDir = QLineEdit(self.tab_batch)
        self.lineEditDir.setObjectName(u"lineEditDir")

        self.horizontalLayout_preview.addWidget(self.lineEditDir)

        self.btnOutDir = QPushButton(self.tab_batch)
        self.btnOutDir.setObjectName(u"btnOutDir")
        sizePolicy.setHeightForWidth(self.btnOutDir.sizePolicy().hasHeightForWidth())
        self.btnOutDir.setSizePolicy(sizePolicy)
        self.btnOutDir.setMaximumSize(QSize(30, 16777215))
        self.btnOutDir.setFont(font4)
        icon3 = QIcon()
        icon3.addFile(u":/images/resource/icons8-folder-64.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btnOutDir.setIcon(icon3)
        self.btnOutDir.setIconSize(QSize(18, 18))

        self.horizontalLayout_preview.addWidget(self.btnOutDir)

        self.btnPreviewClear = QPushButton(self.tab_batch)
        self.btnPreviewClear.setObjectName(u"btnPreviewClear")
        sizePolicy.setHeightForWidth(self.btnPreviewClear.sizePolicy().hasHeightForWidth())
        self.btnPreviewClear.setSizePolicy(sizePolicy)
        self.btnPreviewClear.setFont(font4)

        self.horizontalLayout_preview.addWidget(self.btnPreviewClear)


        self.horizontalLayout_listbox_action.addLayout(self.horizontalLayout_preview)

        self.horizontalLayout_listbox_action.setStretch(0, 1)
        self.horizontalLayout_listbox_action.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_listbox_action)

        icon4 = QIcon()
        icon4.addFile(u":/images/resource/icons8-documents-64.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.tabWidget.addTab(self.tab_batch, icon4, "")

        self.verticalLayout_3.addWidget(self.tabWidget)

        self.horizontalLayout_action_main = QHBoxLayout()
        self.horizontalLayout_action_main.setObjectName(u"horizontalLayout_action_main")
        self.horizontalLayout_action_main.setContentsMargins(10, -1, 10, 0)
        self.horizontalLayout_openFile = QHBoxLayout()
        self.horizontalLayout_openFile.setObjectName(u"horizontalLayout_openFile")
        self.btnOpenFile = QPushButton(self.centralwidget)
        self.btnOpenFile.setObjectName(u"btnOpenFile")
        sizePolicy.setHeightForWidth(self.btnOpenFile.sizePolicy().hasHeightForWidth())
        self.btnOpenFile.setSizePolicy(sizePolicy)
        self.btnOpenFile.setFont(font2)

        self.horizontalLayout_openFile.addWidget(self.btnOpenFile)

        self.lblFilename = QLabel(self.centralwidget)
        self.lblFilename.setObjectName(u"lblFilename")
        self.lblFilename.setFont(font2)
        self.lblFilename.setMargin(5)

        self.horizontalLayout_openFile.addWidget(self.lblFilename)


        self.horizontalLayout_action_main.addLayout(self.horizontalLayout_openFile)

        self.horizontalLayout_process = QHBoxLayout()
        self.horizontalLayout_process.setObjectName(u"horizontalLayout_process")
        self.btnProcess = QPushButton(self.centralwidget)
        self.btnProcess.setObjectName(u"btnProcess")
        sizePolicy.setHeightForWidth(self.btnProcess.sizePolicy().hasHeightForWidth())
        self.btnProcess.setSizePolicy(sizePolicy)
        self.btnProcess.setMinimumSize(QSize(110, 0))
        font7 = QFont()
        font7.setPointSize(12)
        font7.setBold(True)
        self.btnProcess.setFont(font7)
        icon5 = QIcon()
        icon5.addFile(u":/images/resource/icons8-start-48.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btnProcess.setIcon(icon5)
        self.btnProcess.setIconSize(QSize(24, 24))

        self.horizontalLayout_process.addWidget(self.btnProcess)


        self.horizontalLayout_action_main.addLayout(self.horizontalLayout_process)

        self.horizontalLayout_saveExit = QHBoxLayout()
        self.horizontalLayout_saveExit.setObjectName(u"horizontalLayout_saveExit")
        self.horizontalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_saveExit.addItem(self.horizontalSpacer)

        self.btnSaveAs = QPushButton(self.centralwidget)
        self.btnSaveAs.setObjectName(u"btnSaveAs")
        sizePolicy.setHeightForWidth(self.btnSaveAs.sizePolicy().hasHeightForWidth())
        self.btnSaveAs.setSizePolicy(sizePolicy)
        self.btnSaveAs.setFont(font2)

        self.horizontalLayout_saveExit.addWidget(self.btnSaveAs)

        self.btnExit = QPushButton(self.centralwidget)
        self.btnExit.setObjectName(u"btnExit")
        sizePolicy.setHeightForWidth(self.btnExit.sizePolicy().hasHeightForWidth())
        self.btnExit.setSizePolicy(sizePolicy)
        self.btnExit.setFont(font2)

        self.horizontalLayout_saveExit.addWidget(self.btnExit)


        self.horizontalLayout_action_main.addLayout(self.horizontalLayout_saveExit)

        self.horizontalLayout_action_main.setStretch(0, 1)
        self.horizontalLayout_action_main.setStretch(1, 1)
        self.horizontalLayout_action_main.setStretch(2, 1)

        self.verticalLayout_3.addLayout(self.horizontalLayout_action_main)

        self.verticalLayout_3.setStretch(2, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 990, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        self.statusbar.setFont(font2)
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionAbout)

        self.retranslateUi(MainWindow)

        self.cbManual.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Zho Converter PyRs", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.rbS2t.setText(QCoreApplication.translate("MainWindow", u"zh-Hans \uff08\u7b80\uff09 To zh-Hant \uff08\u7e41\uff09", None))
        self.rbT2s.setText(QCoreApplication.translate("MainWindow", u"zh-Hant \uff08\u7e41\uff09 To zh-Hans \uff08\u7b80\uff09", None))
        self.rbManual.setText(QCoreApplication.translate("MainWindow", u"Manual (\u81ea\u5b9a\u4e49) :", None))
        self.cbManual.setItemText(0, QCoreApplication.translate("MainWindow", u"s2t (\u7b80 -> \u7e41)", None))
        self.cbManual.setItemText(1, QCoreApplication.translate("MainWindow", u"s2tw (\u7b80 -> \u7e41/\u53f0)", None))
        self.cbManual.setItemText(2, QCoreApplication.translate("MainWindow", u"s2twp (\u7b80 -> \u7e41/\u53f0/\u60ef)", None))
        self.cbManual.setItemText(3, QCoreApplication.translate("MainWindow", u"s2hk (\u7b80 -> \u7e41/\u6e2f)", None))
        self.cbManual.setItemText(4, QCoreApplication.translate("MainWindow", u"t2s (\u7e41 -> \u7b80)", None))
        self.cbManual.setItemText(5, QCoreApplication.translate("MainWindow", u"t2tw (\u7e41 -> \u7e41/\u53f0)", None))
        self.cbManual.setItemText(6, QCoreApplication.translate("MainWindow", u"t2twp (\u7e41 -> \u7e41/\u53f0/\u60ef)", None))
        self.cbManual.setItemText(7, QCoreApplication.translate("MainWindow", u"t2hk (\u7e41 -> \u7e41/\u6e2f)", None))
        self.cbManual.setItemText(8, QCoreApplication.translate("MainWindow", u"tw2s (\u7e41/\u53f0 -> \u7b80)", None))
        self.cbManual.setItemText(9, QCoreApplication.translate("MainWindow", u"tw2sp (\u7e41/\u53f0 -> \u7b80/\u60ef)", None))
        self.cbManual.setItemText(10, QCoreApplication.translate("MainWindow", u"tw2t (\u7e41/\u53f0 -> \u7e41)", None))
        self.cbManual.setItemText(11, QCoreApplication.translate("MainWindow", u"tw2tp (\u7e41/\u53f0 -> \u7e41/\u60ef)", None))
        self.cbManual.setItemText(12, QCoreApplication.translate("MainWindow", u"hk2s (\u7e41/\u6e2f -> \u7b80)", None))
        self.cbManual.setItemText(13, QCoreApplication.translate("MainWindow", u"hk2t (\u7e41/\u6e2f -> \u7e41)", None))
        self.cbManual.setItemText(14, QCoreApplication.translate("MainWindow", u"jp2t (\u65e5/\u65b0 -> \u65e5/\u65e7)", None))
        self.cbManual.setItemText(15, QCoreApplication.translate("MainWindow", u"t2jp (\u65e5/\u65e7 -> \u65e5/\u65b0)", None))

        self.rbStd.setText(QCoreApplication.translate("MainWindow", u"General \uff08\u901a\u7528\u7b80\u7e41\uff09", None))
        self.rbZhTw.setText(QCoreApplication.translate("MainWindow", u"ZH/TW \uff08\u4e2d\u53f0\u7b80\u7e41\uff09", None))
        self.rbHK.setText(QCoreApplication.translate("MainWindow", u"Hong Kong \uff08\u4e2d\u6e2f\u7b80\u7e41\uff09", None))
        self.cbZhTw.setText(QCoreApplication.translate("MainWindow", u"ZH/TW Idioms \uff08\u60ef\u7528\u8bed\uff09", None))
        self.cbPunct.setText(QCoreApplication.translate("MainWindow", u"Punctuations \uff08\u6807\u70b9\uff09", None))
        self.lblSource.setText(QCoreApplication.translate("MainWindow", u"Source", None))
        self.lblSourceCode.setText("")
        self.lblCharCount.setText("")
#if QT_CONFIG(tooltip)
        self.btnDetect.setToolTip(QCoreApplication.translate("MainWindow", u"Refresh source info", None))
#endif // QT_CONFIG(tooltip)
        self.btnDetect.setText("")
#if QT_CONFIG(tooltip)
        self.btnClearTbSource.setToolTip(QCoreApplication.translate("MainWindow", u"Clear source box contents", None))
#endif // QT_CONFIG(tooltip)
        self.btnClearTbSource.setText(QCoreApplication.translate("MainWindow", u"AC", None))
        self.btnPaste.setText(QCoreApplication.translate("MainWindow", u"Paste", None))
        self.lblDestination.setText(QCoreApplication.translate("MainWindow", u"Destination", None))
        self.lblDestinationCode.setText("")
#if QT_CONFIG(tooltip)
        self.btnClearTbDestination.setToolTip(QCoreApplication.translate("MainWindow", u"Clear destination contents", None))
#endif // QT_CONFIG(tooltip)
        self.btnClearTbDestination.setText(QCoreApplication.translate("MainWindow", u"AC", None))
        self.btnCopy.setText(QCoreApplication.translate("MainWindow", u"Copy", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_main), QCoreApplication.translate("MainWindow", u"Single Convert \uff08\u5355\u4ef6\uff09", None))
        self.btnAdd.setText(QCoreApplication.translate("MainWindow", u"\uff0b", None))
        self.btnRemove.setText(QCoreApplication.translate("MainWindow", u"\u2014", None))
        self.btnClear.setText(QCoreApplication.translate("MainWindow", u"AC", None))
        self.btnPreview.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"Output:", None))
        self.lineEditDir.setText(QCoreApplication.translate("MainWindow", u"./output", None))
        self.btnOutDir.setText("")
        self.btnPreviewClear.setText(QCoreApplication.translate("MainWindow", u"AC", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_batch), QCoreApplication.translate("MainWindow", u"Batch Convert \uff08\u6279\u91cf\uff09", None))
        self.btnOpenFile.setText(QCoreApplication.translate("MainWindow", u"Open File", None))
        self.lblFilename.setText("")
        self.btnProcess.setText(QCoreApplication.translate("MainWindow", u"Process", None))
        self.btnSaveAs.setText(QCoreApplication.translate("MainWindow", u"Save As", None))
        self.btnExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
    # retranslateUi

