import os
from queue import Queue
import cv2
import numpy as np
import torch
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog, QTableWidgetItem
from PyQt6.QtCore import QTimer
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from view.main_ui import Ui_MainWindow
from utilis.database import DatabaseHelper, DetectResult, DetectObj
from controller.figure_process_worker import FigureProcessWorker
from controller.video_process_worker import VideoProcessWorker


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_model_path = "models/class/model_class_final 0.85.pth"
    seg_model_path = "models/seg/model_FPN_final 0.899.pth"
    class_model = torch.load(class_model_path, map_location=device, weights_only=False)
    seg_model = torch.load(seg_model_path, map_location=device, weights_only=False)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Initialize database
        self.conn = DatabaseHelper(
            host='192.168.0.190',
            user='cacc',
            password='20230612',
            database='steel_defect',
        )

        # Maintain a queue to be detected and a list for saving results
        self.detect_queue = Queue()  # Figure queue to be detected
        self.result_list = []  # List to store detection result of figures
        self.chosen_video = None  # Chosen video to be detected

        self.result_video = []

        # Connect slot and btn
        self.figure_btn.clicked.connect(self.select_image)
        self.folder_btn.clicked.connect(self.select_folder)
        self.run_btn.clicked.connect(self.start_detection)
        self.save_btn.clicked.connect(self.save_result_to_db)
        self.video_browse_btn.clicked.connect(self.select_video)
        self.video_detect_btn.clicked.connect(self.start_video_detection)

        self.timer = QTimer(self)
        # Connect buttons to actions
        self.play_btn.clicked.connect(self.play_video)

    """
    Steel Defect Detection by images
    Choose single figure or the folder of figure and detect
    """
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "选择图片文件", "", 'Images (*.png *.jpg *.jpeg *.xpm)')
        if file_path:
            figure_name = os.path.basename(file_path)
            figure = open(file_path, "rb").read()
            detect_obj = DetectObj(figure_name, figure, file_path)
            self.enqueue_figure(detect_obj)
            self.figure_label.setText(figure_name)

        else:
            print("文件路径错误！")

    def select_folder(self):
        """Add all detect objects in specific folder into detect_queue"""
        file_path = QFileDialog.getExistingDirectory(None, "选择图片文件夹")
        if file_path:
            self.folder_label.setText(os.path.basename(file_path))
            for filename in os.listdir(file_path):
                if filename.endswith((".jpg", ".png", ".jpeg", ".xpm")):
                    full_path = os.path.join(file_path, filename)
                    figure = open(full_path, "rb").read()
                    detect_obj = DetectObj(filename, figure, full_path)
                    self.enqueue_figure(detect_obj)
        else:
            print("文件夹路径错误！")

    def start_detection(self):
        self.detection_worker = FigureProcessWorker(self.detect_queue, self.class_model, self.seg_model)
        self.detection_worker.result_signal.connect(self.handle_result)
        self.detection_worker.start()

    def handle_result(self, result):
        self.result_list.append(result)
        self.detect_list.takeItem(0)
        self.show_info(result)
        self.insert_result_to_table(result)


    def enqueue_figure(self, detect_obj):
        """Push object to detect into queue"""
        self.detect_queue.put(detect_obj)
        self.detect_list.addItem(detect_obj.name)

    def insert_result_to_table(self, result):
        row_position = self.result_table.rowCount()
        self.result_table.insertRow(row_position)

        self.result_table.setItem(row_position, 0, QTableWidgetItem(result.name))
        self.result_table.setItem(row_position, 1, QTableWidgetItem(result.label))
        self.result_table.setItem(row_position, 2, QTableWidgetItem(str(result.num)))
        self.result_table.setItem(row_position, 3, QTableWidgetItem(result.time))

        self.result_table.itemClicked.connect(self.show_item_info)

    def show_item_info(self, item):
        row = item.row()
        self.show_info(self.result_list[row])

    def save_result_to_db(self):
        """TODO: Check duplicate"""
        for result in self.result_list:
            self.conn.save_result(result)
        print("所有记录已插入数据库。")

    def show_info(self, result):
        """TODO: Show the information of specific record in result_table"""
        self.class_label.setText(result.label)
        self.dice_label.setText(result.dice)

        # Show result figure
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(result.res_fig)
        self.result_label.setPixmap(pixmap)

    """
    Steel Defect Detection by videos and camera
    Choose video and camera captured frame to detect
    """

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "选择视频文件", "", 'Video files (*.mp4 *.avi)')
        if file_path:
            video_name = os.path.basename(file_path)
            self.chosen_video = cv2.VideoCapture(file_path)
            if not self.chosen_video.isOpened():
                print("无法打开视频文件！")
                return

    def start_video_detection(self):
        """Start the video detection in a separate thread"""
        if self.chosen_video is not None:
            self.video_thread = VideoProcessWorker(self.chosen_video, self.class_model, self.seg_model)
            self.video_thread.video_path_signal.connect(self.play_video)
            self.video_thread.start()

    # def stop_video_detection(self):
    #     """Stop video detection and release resources"""
    #     if self.video_thread is not None:
    #         self.video_thread.stop()
    #         self.chosen_video.release()
    #         print("视频处理已停止")

    def play_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 24)

    def update_frame(self):
        # Read a frame from the video
        ret, frame = self.cap.read()

        if ret:
            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to QImage
            height, width, channels = frame_rgb.shape
            bytes_per_line = channels * width
            qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            # Convert QImage to QPixmap and set it to the QLabel
            pixmap = QPixmap.fromImage(qimage)
            self.video_display_label.setPixmap(pixmap)
        else:
            # If no more frames, stop the video
            self.cap.release()
            self.timer.stop()