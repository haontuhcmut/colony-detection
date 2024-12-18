import sys
import cv2
import csv

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QApplication, QLabel, QPushButton, QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout
)

from PyQt6.QtCore import Qt, QTimer


class WebcamYOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Capture and Counting")
        self.setGeometry(100, 100, 1000, 640)


        #Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # UI component
        self.video_label = QLabel("Video feed will appear here", self)
        self.video_label.setFixedSize(512, 512)
        self.video_label.setStyleSheet("background-color: #cccccc;")

        self.capture_button = QPushButton("Capture", self)
        self.capture_button.setFixedSize(100, 100)

        self.id_sample = QLabel("SAMPLES ID", self)
        self.sample_id_input = QLineEdit(self)
        self.sample_id_input.setPlaceholderText("Enter Sample ID")

        self.select_dir_button_output = QPushButton("Output file name", self)

        self.result_label = QLabel("Counting results:", self)

        ############Layout##########
        main_layout = QVBoxLayout()
        id_layout = QHBoxLayout()

        main_layout.addWidget(self.select_dir_button_output, alignment=Qt.AlignmentFlag.AlignHCenter)
        id_layout.addWidget(self.id_sample)
        id_layout.addWidget(self.sample_id_input)
        main_layout.addLayout(id_layout)
        main_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.capture_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)
        self.setLayout(main_layout)


        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)


        #Connect Signals
        self.select_dir_button_output.clicked.connect(self.select_output_directory)


    def select_output_directory(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output File Name")
        if self.output_dir:
            print(f"Selected output file name: {self.output_dir}")
        else:
            print("No file name selected.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image))






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamYOLOApp()
    window.show()
    sys.exit(app.exec())
