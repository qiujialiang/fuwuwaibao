import os
from pathlib import Path
import sys
import warnings

import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utilis.eda import eda
from utilis.pred import predict
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication,QFileDialog
from PyQt6.QtGui import QStandardItem
from datetime import datetime

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_class = r"fuwuwaibao\models\class\model_class_final 0.85.pth"
model_class = torch.load(path_class, map_location=device)
path_seg = r"fuwuwaibao\models\seg\model_FPN_final 0.899.pth"
model_seg = torch.load(path_seg,map_location=device)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1087, 657)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 40, 221, 231))
        self.groupBox.setObjectName("groupBox")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(80, 40, 101, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(80, 90, 111, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(80, 140, 101, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(80, 190, 101, 21))
        self.label_7.setObjectName("label_7")
        self.toolButton_fig = QtWidgets.QToolButton(parent=self.groupBox)
        self.toolButton_fig.setGeometry(QtCore.QRect(10, 30, 61, 31))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource\\tupian.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_fig.setIcon(icon)
        self.toolButton_fig.setObjectName("toolButton_fig")
        self.toolButton_folder = QtWidgets.QToolButton(parent=self.groupBox)
        self.toolButton_folder.setGeometry(QtCore.QRect(10, 80, 61, 31))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource\\wenjianjia.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_folder.setIcon(icon1)
        self.toolButton_folder.setObjectName("toolButton_folder")
        self.toolButton_video = QtWidgets.QToolButton(parent=self.groupBox)
        self.toolButton_video.setGeometry(QtCore.QRect(10, 130, 61, 31))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource\\shipin.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_video.setIcon(icon2)
        self.toolButton_video.setObjectName("toolButton_video")
        self.toolButton_camera = QtWidgets.QToolButton(parent=self.groupBox)
        self.toolButton_camera.setGeometry(QtCore.QRect(10, 180, 61, 31))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource\\shexiang.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_camera.setIcon(icon3)
        self.toolButton_camera.setObjectName("toolButton_camera")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 280, 221, 61))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_run = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton_run.setGeometry(QtCore.QRect(10, 20, 93, 28))
        self.pushButton_run.setObjectName("pushButton_run")
        self.pushButton_save = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton_save.setGeometry(QtCore.QRect(110, 20, 93, 28))
        self.pushButton_save.setObjectName("pushButton_save")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 360, 221, 241))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(50, 20, 151, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_2.setGeometry(QtCore.QRect(50, 150, 161, 31))
        self.label_2.setObjectName("label_2")
        self.toolButton_5 = QtWidgets.QToolButton(parent=self.groupBox_3)
        self.toolButton_5.setGeometry(QtCore.QRect(10, 20, 41, 31))
        font = QtGui.QFont()
        font.setKerning(True)
        self.toolButton_5.setFont(font)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource\\zhixin.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_5.setIcon(icon4)
        self.toolButton_5.setObjectName("toolButton_5")
        self.toolButton_6 = QtWidgets.QToolButton(parent=self.groupBox_3)
        self.toolButton_6.setGeometry(QtCore.QRect(10, 150, 41, 31))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource\\fenlei.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_6.setIcon(icon5)
        self.toolButton_6.setObjectName("toolButton_6")
        self.label_8 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(10, 60, 201, 81))
        self.label_8.setText("")
        self.label_8.setScaledContents(False)
        self.label_8.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignTop)
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(10, 190, 201, 31))
        self.label_9.setText("")
        self.label_9.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.tableView = QtWidgets.QTableView(parent=self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(280, 410, 791, 192))
        self.tableView.setObjectName("tableView")
        self.tableView.horizontalHeader().setCascadingSectionResizes(False)
        self.reslabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.reslabel.setGeometry(QtCore.QRect(280, 50, 771, 331))
        self.reslabel.setText("")
        self.reslabel.setObjectName("reslabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1087, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.model=QtGui.QStandardItemModel(0,6)
        self.model.setHorizontalHeaderLabels(['名称','时间','结果','缺陷数目','用时','保存路径'])
        self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tableView.setModel(self.model)

        self.row=0

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "选择文件"))
        self.label_4.setText(_translate("MainWindow", "选择图片文件"))
        self.label_5.setText(_translate("MainWindow", "选择图片文件夹"))
        self.label_6.setText(_translate("MainWindow", "选择视频文件"))
        self.label_7.setText(_translate("MainWindow", "摄像头未打开"))
        self.toolButton_fig.setText(_translate("MainWindow", "..."))
        self.toolButton_folder.setText(_translate("MainWindow", "..."))
        self.toolButton_video.setText(_translate("MainWindow", "..."))
        self.toolButton_camera.setText(_translate("MainWindow", "..."))
        self.groupBox_2.setTitle(_translate("MainWindow", "运行"))
        self.pushButton_run.setText(_translate("MainWindow", "开始运行 >"))
        self.pushButton_save.setText(_translate("MainWindow", "导出数据 >"))
        self.groupBox_3.setTitle(_translate("MainWindow", "结果"))
        self.label.setText(_translate("MainWindow", "分类置信度："))
        self.label_2.setText(_translate("MainWindow", "类别："))
        self.toolButton_5.setText(_translate("MainWindow", "..."))
        self.toolButton_6.setText(_translate("MainWindow", "..."))

        
        self.toolButton_fig.clicked.connect(lambda:self.select_image(mode=0))
        self.toolButton_folder.clicked.connect(lambda:self.select_image(mode=1))
        self.toolButton_video.clicked.connect(lambda:self.select_image(mode=2))
        self.pushButton_run.clicked.connect(lambda:self.check(fig_path=self.fig_path,mode=self.mode))
    
    def check(self,fig_path=None,mode=None):
        if fig_path==None or mode==None:
            pass
        if mode==0:
            probability_label,df_segmentation,time=predict(path=Path(fig_path),mode=0,model_class=model_class,model_seg=model_seg)
            fig_data=eda(df=df_segmentation,path=fig_path)
            
            pixmap=QtGui.QPixmap()
            pixmap.loadFromData(fig_data)
            self.reslabel.setPixmap(pixmap)
            self.reslabel.setScaledContents(True)
            
            self.time=time
            self.res=str(df_segmentation['EncodedPixels'])
            self.num=len(df_segmentation[df_segmentation["EncodedPixels"]!=''])
            self.now=datetime.now()
            self.filltable()
            
            probability=str(probability_label.iloc[0,1])
            labels=str(probability_label.iloc[0,2])
            self.label_8.setText(probability)
            self.label_9.setText(labels)
            
        elif mode==1:
            pass
        elif mode==2:
            pass
    
    def filltable(self):
        name=self.name
        time=self.time
        res=self.res
        num=self.num
        now=self.now
        
        item1=QStandardItem(str(name))
        item2=QStandardItem(str(time))
        item3=QStandardItem(str(res))
        item4=QStandardItem(str(num))
        item5=QStandardItem(str(now))
        
        self.model.setItem(self.row,0,item1)
        self.model.setItem(self.row,1,item5)
        self.model.setItem(self.row,2,item3)
        self.model.setItem(self.row,3,item4)
        self.model.setItem(self.row,4,item2)
        
        self.row+=1
        
    def select_image(self,mode=None):
        if mode==0:
            self.mode=0
            file_path,_=QFileDialog.getOpenFileName(None,"选择图片文件","",'Images (*.png *.xpm *.jpg *.jpeg)')
            if file_path:
                self.fig_path=file_path
                self.name=os.path.basename(file_path)
                self.label_4.setText(self.name)
            else:
                print("文件路径错误")
        elif mode==1:
            self.mode=1
            file_path=QFileDialog.getExistingDirectory(None,"选择图片文件夹")
            if file_path:
                self.fig_path=file_path
                self.name=os.path.basename(file_path)
                self.label_5.setText(self.name)
            else:
                print("文件路径错误")
        elif mode==2:
            self.mode=2
            file_path,_=QFileDialog.getOpenFileName(None,"选择视频文件","","Videos (*.mp4 *.avi *.mkv)")
            if file_path:
                self.name=os.path.basename(file_path)
                self.label_6.setText(self.name)
            else:
                print("文件路径错误")
            
    
def main():
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()  # 创建QMainWindow实例
    ui = Ui_MainWindow()  # 创建Ui_MainWindow实例
    ui.setupUi(MainWindow)  # 将UI设置应用到QMainWindow实例
    MainWindow.show()  # 显示主窗口
    sys.exit(app.exec())

if __name__ == "__main__":
    main()