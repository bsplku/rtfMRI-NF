# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QDateEdit, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QMainWindow,
    QPushButton, QSizePolicy, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, ExperimentMainWindow):
        if not ExperimentMainWindow.objectName():
            ExperimentMainWindow.setObjectName(u"ExperimentMainWindow")
        ExperimentMainWindow.resize(800, 300)
        self.grid_main = QWidget(ExperimentMainWindow)
        self.grid_main.setObjectName(u"grid_main")
        self.gridLayout = QGridLayout(self.grid_main)
        self.gridLayout.setObjectName(u"gridLayout")
        self.group_Settings = QGroupBox(self.grid_main)
        self.group_Settings.setObjectName(u"group_Settings")
        self.gridLayout_3 = QGridLayout(self.group_Settings)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.group_MRIParm = QGroupBox(self.group_Settings)
        self.group_MRIParm.setObjectName(u"group_MRIParm")
        self.gridLayout_2 = QGridLayout(self.group_MRIParm)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_date = QLabel(self.group_MRIParm)
        self.label_date.setObjectName(u"label_date")
        self.label_date.setMinimumSize(QSize(71, 0))

        self.gridLayout_2.addWidget(self.label_date, 0, 0, 1, 1)

        self.box_date = QDateEdit(self.group_MRIParm)
        self.box_date.setObjectName(u"box_date")
        self.box_date.setDateTime(QDateTime(QDate(2022, 7, 27), QTime(0, 0, 0)))

        self.gridLayout_2.addWidget(self.box_date, 0, 1, 1, 1)

        self.label_subject = QLabel(self.group_MRIParm)
        self.label_subject.setObjectName(u"label_subject")
        self.label_subject.setMinimumSize(QSize(71, 0))

        self.gridLayout_2.addWidget(self.label_subject, 1, 0, 1, 1)

        self.box_subject = QLineEdit(self.group_MRIParm)
        self.box_subject.setObjectName(u"box_subject")

        self.gridLayout_2.addWidget(self.box_subject, 1, 1, 1, 1)


        self.gridLayout_3.addWidget(self.group_MRIParm, 0, 0, 1, 1)

        self.group_ReviewParm = QGroupBox(self.group_Settings)
        self.group_ReviewParm.setObjectName(u"group_ReviewParm")
        self.gridLayout_5 = QGridLayout(self.group_ReviewParm)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_nrt_volumes_title = QLabel(self.group_ReviewParm)
        self.label_nrt_volumes_title.setObjectName(u"label_nrt_volumes_title")
        self.label_nrt_volumes_title.setMinimumSize(QSize(71, 0))

        self.gridLayout_5.addWidget(self.label_nrt_volumes_title, 0, 0, 1, 1)

        self.label_nrt_volumes = QLabel(self.group_ReviewParm)
        self.label_nrt_volumes.setObjectName(u"label_nrt_volumes")
        self.label_nrt_volumes.setMinimumSize(QSize(71, 0))

        self.gridLayout_5.addWidget(self.label_nrt_volumes, 0, 1, 1, 1)

        self.label_rt_volumes_title = QLabel(self.group_ReviewParm)
        self.label_rt_volumes_title.setObjectName(u"label_rt_volumes_title")
        self.label_rt_volumes_title.setMinimumSize(QSize(71, 0))

        self.gridLayout_5.addWidget(self.label_rt_volumes_title, 1, 0, 1, 1)

        self.label_rt_volumes = QLabel(self.group_ReviewParm)
        self.label_rt_volumes.setObjectName(u"label_rt_volumes")
        self.label_rt_volumes.setMinimumSize(QSize(71, 0))

        self.gridLayout_5.addWidget(self.label_rt_volumes, 1, 1, 1, 1)


        self.gridLayout_3.addWidget(self.group_ReviewParm, 1, 0, 1, 1)

        self.group_Paradigm = QGroupBox(self.group_Settings)
        self.group_Paradigm.setObjectName(u"group_Paradigm")
        self.gridLayout_4 = QGridLayout(self.group_Paradigm)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.btn_rt = QPushButton(self.group_Paradigm)
        self.btn_rt.setObjectName(u"btn_rt")

        self.gridLayout_4.addWidget(self.btn_rt, 16, 1, 1, 1)

        self.label_series_3 = QLabel(self.group_Paradigm)
        self.label_series_3.setObjectName(u"label_series_3")

        self.gridLayout_4.addWidget(self.label_series_3, 15, 0, 1, 1)

        self.label_series_2 = QLabel(self.group_Paradigm)
        self.label_series_2.setObjectName(u"label_series_2")

        self.gridLayout_4.addWidget(self.label_series_2, 0, 0, 1, 1)

        self.btn_nrt = QPushButton(self.group_Paradigm)
        self.btn_nrt.setObjectName(u"btn_nrt")

        self.gridLayout_4.addWidget(self.btn_nrt, 16, 0, 1, 1)

        self.box_series_rt = QLineEdit(self.group_Paradigm)
        self.box_series_rt.setObjectName(u"box_series_rt")

        self.gridLayout_4.addWidget(self.box_series_rt, 1, 1, 1, 1)

        self.box_series_nrt = QLineEdit(self.group_Paradigm)
        self.box_series_nrt.setObjectName(u"box_series_nrt")

        self.gridLayout_4.addWidget(self.box_series_nrt, 1, 0, 1, 1)

        self.label_series_5 = QLabel(self.group_Paradigm)
        self.label_series_5.setObjectName(u"label_series_5")

        self.gridLayout_4.addWidget(self.label_series_5, 0, 1, 1, 1)

        self.label_series_8 = QLabel(self.group_Paradigm)
        self.label_series_8.setObjectName(u"label_series_8")

        self.gridLayout_4.addWidget(self.label_series_8, 2, 1, 1, 1)

        self.run_rt = QComboBox(self.group_Paradigm)
        self.run_rt.addItem("")
        self.run_rt.addItem("")
        self.run_rt.addItem("")
        self.run_rt.addItem("")
        self.run_rt.setObjectName(u"run_rt")

        self.gridLayout_4.addWidget(self.run_rt, 3, 1, 1, 1)

        self.label_series_6 = QLabel(self.group_Paradigm)
        self.label_series_6.setObjectName(u"label_series_6")

        self.gridLayout_4.addWidget(self.label_series_6, 15, 1, 1, 1)

        self.label_series = QLabel(self.group_Paradigm)
        self.label_series.setObjectName(u"label_series")

        self.gridLayout_4.addWidget(self.label_series, 2, 0, 1, 1)

        self.run_nrt = QComboBox(self.group_Paradigm)
        self.run_nrt.addItem("")
        self.run_nrt.addItem("")
        self.run_nrt.addItem("")
        self.run_nrt.addItem("")
        self.run_nrt.setObjectName(u"run_nrt")

        self.gridLayout_4.addWidget(self.run_nrt, 3, 0, 1, 1)

        self.label_title_sham = QLabel(self.group_Paradigm)
        self.label_title_sham.setObjectName(u"label_title_sham")

        self.gridLayout_4.addWidget(self.label_title_sham, 4, 1, 1, 1)

        self.label_sham_path = QLabel(self.group_Paradigm)
        self.label_sham_path.setObjectName(u"label_sham_path")

        self.gridLayout_4.addWidget(self.label_sham_path, 5, 1, 1, 1)

        self.btn_sham_path = QPushButton(self.group_Paradigm)
        self.btn_sham_path.setObjectName(u"btn_sham_path")

        self.gridLayout_4.addWidget(self.btn_sham_path, 6, 1, 1, 1)


        self.gridLayout_3.addWidget(self.group_Paradigm, 2, 0, 1, 2)


        self.gridLayout.addWidget(self.group_Settings, 0, 0, 1, 2)

        ExperimentMainWindow.setCentralWidget(self.grid_main)

        self.retranslateUi(ExperimentMainWindow)

        QMetaObject.connectSlotsByName(ExperimentMainWindow)
    # setupUi

    def retranslateUi(self, ExperimentMainWindow):
        ExperimentMainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Real-time fMRI NF toolbox for multimodal image and text", None))
        self.group_Settings.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.group_MRIParm.setTitle(QCoreApplication.translate("MainWindow", u"Experiment", None))
        self.label_date.setText(QCoreApplication.translate("MainWindow", u"Date", None))
        self.box_date.setDisplayFormat(QCoreApplication.translate("MainWindow", u"yyyyMMdd", None))
        self.label_subject.setText(QCoreApplication.translate("MainWindow", u"Subject", None))
        self.box_subject.setText(QCoreApplication.translate("MainWindow", u"KJE", None))
        self.group_ReviewParm.setTitle(QCoreApplication.translate("MainWindow", u"Review", None))
        self.label_nrt_volumes_title.setText(QCoreApplication.translate("MainWindow", u"# of NRT Task Volumes", None))
        self.label_nrt_volumes.setText(QCoreApplication.translate("MainWindow", u"NRT Task Volumes", None))
        self.label_rt_volumes_title.setText(QCoreApplication.translate("MainWindow", u"# of RT Task Volumes", None))
        self.label_rt_volumes.setText(QCoreApplication.translate("MainWindow", u"RT Task Volumes", None))
        self.group_Paradigm.setTitle(QCoreApplication.translate("MainWindow", u"Run", None))
        self.btn_rt.setText(QCoreApplication.translate("MainWindow", u"RT", None))
        self.label_series_3.setText(QCoreApplication.translate("MainWindow", u"Start button (nrt)", None))
        self.label_series_2.setText(QCoreApplication.translate("MainWindow", u"Series (nrt)", None))
        self.btn_nrt.setText(QCoreApplication.translate("MainWindow", u"NRT", None))
        self.box_series_rt.setText(QCoreApplication.translate("MainWindow", u"2", None))
        self.box_series_nrt.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.label_series_5.setText(QCoreApplication.translate("MainWindow", u"Series (rt)", None))
        self.label_series_8.setText(QCoreApplication.translate("MainWindow", u"Run (rt)", None))
        self.run_rt.setItemText(0, QCoreApplication.translate("MainWindow", u"1", None))
        self.run_rt.setItemText(1, QCoreApplication.translate("MainWindow", u"2", None))
        self.run_rt.setItemText(2, QCoreApplication.translate("MainWindow", u"3", None))
        self.run_rt.setItemText(3, QCoreApplication.translate("MainWindow", u"4", None))

        self.label_series_6.setText(QCoreApplication.translate("MainWindow", u"Start button (rt)", None))
        self.label_series.setText(QCoreApplication.translate("MainWindow", u"Run (nrt)", None))
        self.run_nrt.setItemText(0, QCoreApplication.translate("MainWindow", u"pre1", None))
        self.run_nrt.setItemText(1, QCoreApplication.translate("MainWindow", u"pre2", None))
        self.run_nrt.setItemText(2, QCoreApplication.translate("MainWindow", u"post1", None))
        self.run_nrt.setItemText(3, QCoreApplication.translate("MainWindow", u"post2", None))

        self.label_title_sham.setText(QCoreApplication.translate("MainWindow", u"Matching Subject Etime Path (rt)", None))
        self.label_sham_path.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.btn_sham_path.setText(QCoreApplication.translate("MainWindow", u"Set Etime Path", None))
    # retranslateUi

