from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QFileDialog, QInputDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from face_recognition_system import FaceRecognitionSystem

import cv2

class AttendanceSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance System")
        self.setGeometry(100, 100, 800, 600)

        self.face_recognition_system = FaceRecognitionSystem()

        main_layout = QHBoxLayout()
        
        # Camera feed layout
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        camera_layout.addWidget(self.camera_label)

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        camera_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        camera_layout.addWidget(self.stop_button)

        self.add_face_button = QPushButton("Add Known Face")
        self.add_face_button.clicked.connect(self.add_known_face)
        camera_layout.addWidget(self.add_face_button)

        main_layout.addLayout(camera_layout)

        # Attendance list layout
        attendance_layout = QVBoxLayout()
        attendance_label = QLabel("Attendance List:")
        attendance_layout.addWidget(attendance_label)
        self.attendance_list = QLabel()
        attendance_layout.addWidget(self.attendance_list)

        main_layout.addLayout(attendance_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.face_recognition_system.start_camera()
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        self.face_recognition_system.stop_camera()

    def update_frame(self):
        frame, name = self.face_recognition_system.process_frame()
        if frame is not None:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_label.setPixmap(pixmap)

        if name:
            self.update_attendance_list(name)

    def update_attendance_list(self, name):
        current_text = self.attendance_list.text()
        new_text = f"{current_text}\n{name}" if current_text else name
        self.attendance_list.setText(new_text)

    def add_known_face(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if image_path:
            name, ok = QInputDialog.getText(self, "Enter Name", "Enter the name for this face:")
            if ok and name:
                self.face_recognition_system.add_known_face(image_path, name)