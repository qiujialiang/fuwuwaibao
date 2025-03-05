import os
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import cv2
from utilis.pred import predict
from utilis.eda import eda
import torch

class VideoProcessWorker(QThread):
    video_signal = pyqtSignal(str)

    def __init__(self, video_capture, class_model, seg_model, output_path="output.mp4"):
        super().__init__()
        self.video_capture = video_capture
        self.class_model = class_model
        self.seg_model = seg_model
        self.running = True
        self.output_path = output_path

        self.width, self.height = 0, 0

    def run(self):
        frame_count = 0
        frames = []

        while self.running and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame_count += 1
            print(f"处理第 {frame_count} 帧...")

            _, encoded_frame = cv2.imencode('.jpg', frame)
            binary_frame = encoded_frame.tobytes()

            predicted_label, segmentation, time_cost = predict(
                name=f"frame_{frame_count}",
                image_data=binary_frame,
                model_class=self.class_model,
                model_seg=self.seg_model,
                mode=2
            )

            res_fig = eda(df=segmentation, data=binary_frame)

            frames.append(res_fig)

            torch.cuda.empty_cache()

        if frames:
            self.save_video(frames, self.output_path)

        # 发送信号，通知视频已保存
        self.video_signal.emit(self.output_path)

    def save_video(self, frames, output_path, fps=24):
        """将帧列表保存为视频"""

        # 如果没有帧数据，直接返回
        if not frames:
            return

        # 解码第一帧来获取图像尺寸
        nparr = np.frombuffer(frames[0], np.uint8)  # 将二进制数据转换为 NumPy 数组
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为图像
        height, width, _ = frame.shape  # 获取图像的高、宽和通道数
        shape = (width, height)  # 视频的尺寸是 (width, height)

        # 设置视频编码格式
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, shape)

        for frame_data in frames:
            nparr = np.frombuffer(frame_data, np.uint8)  # 将二进制数据转换为 NumPy 数组
            frame_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为图像
            video_writer.write(frame_img)  # 将解码后的图像写入视频

        video_writer.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.wait()  # 等待线程结束
