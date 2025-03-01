import os
import torch
from pathlib import Path
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtWidgets import QFileDialog
from view.main_ui import Ui_MainWindow
from utilis.database import DatabaseHelper, DetectResult
from utilis.pred import predict
from utilis.eda import eda


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Initialize database
        conn = DatabaseHelper(
            host='192.168.0.190',
            user='cacc',
            password='20230612',
            database='steel_defect',
        )

        # Initialize models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_model_path = "models/class/model_class_final 0.85.pth"
        seg_model_path = "models/seg/model_FPN_final 0.899.pth"
        self.class_model = torch.load(class_model_path, map_location=device, weights_only=False)
        self.seg_model = torch.load(seg_model_path, map_location=device, weights_only=False)

        # Connect
        self.figure_btn.clicked.connect(self.select_image)
        self.figure_path = None
        self.folder_btn.clicked.connect(self.select_folder)
        self.run_btn.clicked.connect(self.detect)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "选择图片文件", "", 'Images (*.png *.jpg *.jpeg *.xpm)')
        if file_path:
            self.figure_path = file_path
            self.figure_label.setText(os.path.basename(file_path))
        else:
            print("文件路径错误！")

    def select_folder(self):
        file_path = QFileDialog.getExistingDirectory(None, "选择图片文件夹")
        if file_path:
            self.folder_path = file_path
            self.folder_label.setText(os.path.basename(file_path))
        else:
            print("文件夹路径错误！")

    def detect(self):
        predicted_label, segmentation, time_cost = predict(
            path=Path(self.figure_path),
            mode=0,
            model_class=self.class_model,
            model_seg=self.seg_model,
        )

        with open(self.figure_path, 'rb') as f:
            raw_fig = f.read()

        res_fig = eda(df=segmentation, path=self.figure_path)
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(res_fig)
        self.result_label.setPixmap(pixmap)
