import sys
import cv2
import csv
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QHBoxLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from ultralytics import YOLO


class WebcamYOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Capture and Counting")
        self.setGeometry(100, 100, 1000, 640)

        #Initialize webcam
        self.cap = cv2.VideoCapture(0)

        #Load model
        self.model = YOLO('runs/detect/train8/weights/best.pt')

        # UI components
        self.video_label = QLabel(self)
        self.video_label.setText("Video feed will appear here")
        self.video_label.setStyleSheet("background-color: #cccccc;")
        self.video_label.setFixedSize(480, 480)

        self.capture_button = QPushButton("Capture and Counting", self)

        self.id_sample = QLabel("ID_SAMPLES", self)
        self.id_sample_edit = QLineEdit(self)

        self.save = QPushButton("Save", self)

        self.result_label = QLabel("Counting Results:", self)

        # Layout
        main_layout = QVBoxLayout()

        # Add video feed
        main_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add capture button
        main_layout.addWidget(self.capture_button)

        # Add ID sample input
        id_layout = QHBoxLayout()
        id_layout.addWidget(self.id_sample)
        id_layout.addWidget(self.id_sample_edit)
        main_layout.addLayout(id_layout)

        # Add save
        main_layout.addWidget(self.save)

        # Add result label
        main_layout.addWidget(self.result_label)

        # Set main layout
        self.setLayout(main_layout)

        # Timer for updating video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

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

            # Save annotated image
            predicted_image_path = "predicted_image.jpg"
            cv2.imwrite(predicted_image_path, annotated_image)

            # Extract prediction details
            predictions = []
            for box in results[0].boxes:
                cls_name = self.model.names[int(box.cls.item())]  # Class name
                confidence = box.conf.item()  # Confidence score
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()  # Bounding box coordinates
                predictions.append([cls_name, confidence, x_min, y_min, x_max, y_max])

            # Save results to CSV
            csv_file_path = "prediction_results.csv"
            with open(csv_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Class", "Confidence", "X_min", "Y_min", "X_max", "Y_max"])
                writer.writerows(predictions)

            # Update the UI
            result_text = [f"{p[0]} ({p[1]:.2f})" for p in predictions]
            self.result_label.setText("Prediction Results:\n" + "\n".join(result_text))

            # Notify the user
            print(f"Predicted image saved to: {predicted_image_path}")
            print(f"Prediction results saved to: {csv_file_path}")

    def closeEvent(self, event):
        """Release the webcam resource on close."""
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamYOLOApp()
    window.show()
    sys.exit(app.exec())
