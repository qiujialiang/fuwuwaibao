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
import cv2

import mysql.connector
db_config={
    'host': 'localhost',
    'user': 'root',
    'password': '12345',
    'database': 'project'
}

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_class = r"models\class\model_class_final 0.85.pth"
model_class = torch.load(path_class, map_location=device)
path_seg = r"models\seg\model_FPN_final 0.899.pth"
model_seg = torch.load(path_seg,map_location=device)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(755, 578)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.reslabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.reslabel.setGeometry(QtCore.QRect(260, 30, 451, 231))
        self.reslabel.setText("")
        self.reslabel.setObjectName("reslabel")
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 30, 211, 191))
        self.groupBox.setObjectName("groupBox")
        self.toolButton_fig = QtWidgets.QToolButton(parent=self.groupBox)
        self.toolButton_fig.setGeometry(QtCore.QRect(20, 30, 47, 31))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource/tupian.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_fig.setIcon(icon)
        self.toolButton_fig.setObjectName("toolButton_fig")
        self.toolButton_folder = QtWidgets.QToolButton(parent=self.groupBox)
        self.toolButton_folder.setGeometry(QtCore.QRect(20, 70, 47, 31))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource/wenjianjia.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_folder.setIcon(icon1)
        self.toolButton_folder.setObjectName("toolButton_folder")
        self.toolButton_video = QtWidgets.QToolButton(parent=self.groupBox)
        self.toolButton_video.setGeometry(QtCore.QRect(20, 110, 47, 31))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource/shipin.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_video.setIcon(icon2)
        self.toolButton_video.setObjectName("toolButton_video")
        self.toolButton_camera = QtWidgets.QToolButton(parent=self.groupBox)
        self.toolButton_camera.setGeometry(QtCore.QRect(20, 150, 47, 31))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource/shexiang.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.toolButton_camera.setIcon(icon3)
        self.toolButton_camera.setObjectName("toolButton_camera")
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(80, 30, 111, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(80, 70, 111, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(80, 110, 111, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(80, 150, 111, 31))
        self.label_5.setObjectName("label_5")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 239, 211, 71))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_run = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton_run.setGeometry(QtCore.QRect(10, 20, 93, 28))
        self.pushButton_run.setObjectName("pushButton_run")
        self.pushButton_save = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton_save.setGeometry(QtCore.QRect(110, 20, 93, 28))
        self.pushButton_save.setObjectName("pushButton_save")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 330, 211, 181))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_6 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(20, 30, 31, 31))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource/zhixin.png"))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(50, 30, 131, 31))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(20, 120, 31, 31))
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap("c:\\Users\\qiujialiang\\Desktop\\fuwuwaibao\\project\\ui\\resource/fenlei.png"))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(50, 120, 131, 31))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(20, 70, 171, 41))
        self.label_10.setText("")
        self.label_10.setWordWrap(True)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_11.setGeometry(QtCore.QRect(20, 150, 171, 21))
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.tableView = QtWidgets.QTableView(parent=self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(260, 310, 451, 192))
        self.tableView.setObjectName("tableView")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 755, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.model=QtGui.QStandardItemModel(0,5)
        self.model.setHorizontalHeaderLabels(['名称','时间','缺陷数目','用时','保存路径'])
        self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tableView.setModel(self.model)
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        
        self.row=0

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "选择文件"))
        self.toolButton_fig.setText(_translate("MainWindow", "..."))
        self.toolButton_folder.setText(_translate("MainWindow", "..."))
        self.toolButton_video.setText(_translate("MainWindow", "..."))
        self.toolButton_camera.setText(_translate("MainWindow", "..."))
        self.label_2.setText(_translate("MainWindow", "选择图片文件"))
        self.label_3.setText(_translate("MainWindow", "选择图片文件夹"))
        self.label_4.setText(_translate("MainWindow", "选择视频文件"))
        self.label_5.setText(_translate("MainWindow", "摄像头未打开"))
        self.groupBox_2.setTitle(_translate("MainWindow", "运行"))
        self.pushButton_run.setText(_translate("MainWindow", "开始运行>"))
        self.pushButton_save.setText(_translate("MainWindow", "导出数据>"))
        self.groupBox_3.setTitle(_translate("MainWindow", "结果"))
        self.label_7.setText(_translate("MainWindow", "分类置信度："))
        self.label_9.setText(_translate("MainWindow", "类别："))

        
        self.toolButton_fig.clicked.connect(lambda:self.select_image(mode=0))
        self.toolButton_folder.clicked.connect(lambda:self.select_image(mode=1))
        self.toolButton_video.clicked.connect(lambda:self.select_image(mode=2))
        self.pushButton_run.clicked.connect(lambda:self.check(fig_path=self.fig_path,mode=self.mode))
        self.tableView.doubleClicked.connect(self.showItem)

    
    def process_video(self,video_path):
        cap=cv2.VideoCapture(video_path)
        
    def showItem(self,index):
        name=index.data()
        conn=mysql.connector.connect(**db_config)
        cursor=conn.cursor()
        query="SELECT * FROM res WHERE name = %s"
        cursor.execute(query,(name,))
        result=cursor.fetchone()
        if result:
            self.fig_data=result[2]
            self.labels=result[5]
            self.probability=result[7]
            
            pixmap=QtGui.QPixmap()
            pixmap.loadFromData(self.fig_data)
            self.reslabel.setPixmap(pixmap)
            self.reslabel.setScaledContents(True)
            self.label_11.setText(self.labels)
            self.label_10.setText(self.probability)
        else: 
            print("error")
        cursor.close()
        conn.close()
    def save(self):
        conn=mysql.connector.connect(**db_config)
        cursor=conn.cursor()
        insert_query = """
            INSERT INTO res (name, raw_fig, res_fig, datetime, time, class, num, dice)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """    
        data = (
                self.name,
                self.raw_data,
                self.fig_data,
                self.now,
                self.time,
                self.labels,
                self.num,
                self.probability  # 假设 dice 是一个字符串
            )
        cursor.execute(insert_query, data)
        conn.commit()
        cursor.close()
        conn.close()
    def check(self,fig_path,mode=None):
        if fig_path==None or mode==None:
            pass
        if mode==0:
            probability_label,df_segmentation,time=predict(path=Path(fig_path),mode=0,model_class=model_class,model_seg=model_seg)
            with open(fig_path,'rb') as f:
                self.raw_data=f.read()
                
            self.fig_data=eda(df=df_segmentation,path=fig_path)
            
            pixmap=QtGui.QPixmap()
            pixmap.loadFromData(self.fig_data)
            self.reslabel.setPixmap(pixmap)
            self.reslabel.setScaledContents(True)
            
            self.time=f"{time:.2f}s"
            self.res=str(df_segmentation['EncodedPixels'])
            self.num=len(df_segmentation[df_segmentation["EncodedPixels"]!=''])
            self.now=datetime.now()
            self.filltable()
            
            self.probability=', '.join(probability_label['probability_label'].astype(str))
            self.labels = ', '.join(df_segmentation['EncodedPixels']!=''.astype(str))
            self.label_10.setText(self.probability)
            self.label_11.setText(self.labels)
            
            self.save()
            
        elif mode==1:
            image_extensions = ('.png', '.xpm', '.jpg', '.jpeg')
            for file_name in os.listdir(fig_path):
                if file_name.lower().endswith(image_extensions):
                    self.name=file_name
                    image_path = os.path.join(fig_path, file_name)
                    self.check(fig_path=image_path,mode=0)
        elif mode==2:
            self.process_video(self,video_path=fig_path)
    
    def filltable(self):
        name=self.name
        time=self.time
        num=self.num
        now=self.now
        
        item1=QStandardItem(str(name))
        item2=QStandardItem(str(time))
        item4=QStandardItem(str(num))
        item5=QStandardItem(str(now))
        
        self.model.setItem(self.row,0,item1)
        self.model.setItem(self.row,2,item4)
        self.model.setItem(self.row,1,item5)
        self.model.setItem(self.row,3,item2)
        
        self.row+=1
        
    def select_image(self,mode=None):
        if mode==0:
            self.mode=0
            file_path,_=QFileDialog.getOpenFileName(None,"选择图片文件","",'Images (*.png *.xpm *.jpg *.jpeg)')
            if file_path:
                self.fig_path=file_path
                self.name=os.path.basename(file_path)
                self.label_2.setText(self.name)
            else:
                print("文件路径错误")
        elif mode==1:
            self.mode=1
            file_path=QFileDialog.getExistingDirectory(None,"选择图片文件夹")
            if file_path:
                self.fig_path=file_path
                self.name=os.path.basename(file_path)
                self.label_3.setText(self.name)      
            else:
                print("文件路径错误")
        elif mode==2:
            self.mode=2
            file_path,_=QFileDialog.getOpenFileName(None,"选择视频文件","","Videos (*.mp4 *.avi *.mkv)")
            if file_path:
                self.name=os.path.basename(file_path)
                self.label_4.setText(self.name)
            else:
                print("文件路径错误")

def clear_database():
    conn=mysql.connector.connect(**db_config)
    cursor=conn.cursor()
    cursor.execute("DELETE FROM res")
    conn.commit()
    cursor.close()
    conn.close()        
    
def main():
    clear_database()
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()  # 创建QMainWindow实例
    ui = Ui_MainWindow()  # 创建Ui_MainWindow实例
    ui.setupUi(MainWindow)  # 将UI设置应用到QMainWindow实例
    MainWindow.show()  # 显示主窗口
    sys.exit(app.exec())

if __name__ == "__main__":
    main()