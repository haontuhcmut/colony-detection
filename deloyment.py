import sys
import cv2
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from ultralytics import YOLO


class WebcamYOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Capture and Counting")
        self.setGeometry(100, 100, 400, 400)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # Load model
        self.model = YOLO('runs/detect/train8/weights/best.pt')

        # UI components
        self.video_label = QLabel(self)
        self.capture_button = QPushButton("Capture and Counting", self)
        self.result_label = QLabel("Counting Results:", self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        # Timer for updating video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Connect button to capture and predict function
        self.capture_button.clicked.connect(self.capture_and_predict)

    def update_frame(self):
        """Update the video feed."""
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV frame (BGR) to QImage (RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(image))

    def capture_and_predict(self):
        """Capture the current frame, save it, and perform YOLO prediction."""
        ret, frame = self.cap.read()
        if ret:
            # Save the frame temporarily
            temp_image_path = "captured_image.jpg"
            cv2.imwrite(temp_image_path, frame)

            # Perform YOLO prediction
            results = self.model(temp_image_path)

            # Annotate image with YOLO results
            annotated_image = results[0].plot()

            # Convert annotated image to QImage for display
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            image = QImage(annotated_image, annotated_image.shape[1], annotated_image.shape[0],
                           QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(image))

            # Extract prediction details
            result_text = []
            for box in results[0].boxes:  # Iterate over each bounding box
                cls_name = self.model.names[int(box.cls.item())]  # Class name
                confidence = box.conf.item()  # Convert confidence tensor to scalar
                result_text.append(f"{cls_name} ({confidence:.2f})")

            # Display results
            self.result_label.setText("Prediction Results:\n" + "\n".join(result_text))

    def closeEvent(self, event):
        """Release the webcam resource on close."""
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamYOLOApp()
    window.show()
    sys.exit(app.exec())
