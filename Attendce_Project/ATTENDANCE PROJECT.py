import sys
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv
import json
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont, QFontDatabase
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Constants from config
PATH = config['PATH']
ATTENDANCE_FILE = config['ATTENDANCE_FILE']
UNKNOWN_FACE_LABEL = config['UNKNOWN_FACE_LABEL']
FACE_RECOGNITION_THRESHOLD = config['FACE_RECOGNITION_THRESHOLD']

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceTracker:
    def __init__(self, name, bbox):
        self.name = name
        self.bbox = bbox
        self.time_since_update = 0
        self.animation = None

    def update(self, bbox):
        self.bbox = bbox
        self.time_since_update = 0

def load_images():
    images = []
    class_names = []
    my_list = os.listdir(PATH)
    logging.info(f"Loading images: {my_list}")
    for cl in my_list:
        cur_img = cv2.imread(f'{PATH}/{cl}')
        if cur_img is not None:
            images.append(cur_img)
            class_names.append(os.path.splitext(cl)[0])
        else:
            logging.warning(f"Failed to load image: {cl}")
    logging.info(f"Loaded class names: {class_names}")
    return images, class_names

def find_encodings(images, class_names):
    encode_list = []
    valid_class_names = []
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                encode = face_encodings[0]
                encode_list.append(encode)
                valid_class_names.append(class_names[i])
            else:
                logging.warning(f"No face detected in image {i} ({class_names[i]}). Skipping...")
        except Exception as e:
            logging.error(f"Error processing image {i} ({class_names[i]}): {str(e)}")
    return encode_list, valid_class_names

def mark_attendance(name):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')
    
    try:
        with open(ATTENDANCE_FILE, 'a+', newline='') as f:
            f.seek(0)
            reader = csv.reader(f)
            rows = list(reader)
            
            if not rows:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])
                rows.append(['Name', 'Date', 'Time'])
            
            for row in rows:
                if len(row) >= 2 and row[0] == name and row[1] == date_string:
                    logging.info(f"{name} already marked for today.")
                    return False
            
            writer = csv.writer(f)
            writer.writerow([name, date_string, time_string])
            logging.info(f"Marked attendance for {name}")
            return True
    except IOError as e:
        logging.error(f"Error writing to attendance file: {str(e)}")
        return False

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern Face Recognition Attendance")
        self.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (video feed)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.header_label = QLabel("Face Recognition Attendance")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 24px; color: #00A0A0;")
        left_layout.addWidget(self.header_label)
        
        self.image_label = QLabel()
        left_layout.addWidget(self.image_label)
        
        main_layout.addWidget(left_panel, 2)
        
        # Right panel (sidebar)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.attendance_label = QLabel("Today's Attendance")
        self.attendance_label.setStyleSheet("font-size: 18px; color: #00A0A0;")
        right_layout.addWidget(self.attendance_label)
        
        self.attendance_list = QLabel()
        self.attendance_list.setStyleSheet("font-size: 14px;")
        right_layout.addWidget(self.attendance_list)
        
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel, 1)
        
        # Load known faces and encodings
        images, class_names = load_images()
        self.encode_list_known, self.valid_class_names = find_encodings(images, class_names)
        logging.info('Encoding Complete')
        logging.info(f"Successfully encoded {len(self.encode_list_known)} faces out of {len(class_names)} images")
        
        self.cap = cv2.VideoCapture(0)#change to 0 to use default camera
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms
        
        self.face_trackers = []
        
        # Update attendance list every 5 seconds
        self.attendance_timer = QTimer(self)
        self.attendance_timer.timeout.connect(self.update_attendance_list)
        self.attendance_timer.start(5000)
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            
            # Perform face recognition
            face_locations = face_recognition.face_locations(frame_rgb)
            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
            
            # Update trackers
            for tracker in self.face_trackers:
                tracker.time_since_update += 1
            
            updated_trackers = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.encode_list_known, face_encoding)
                face_distances = face_recognition.face_distance(self.encode_list_known, face_encoding)
                
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < FACE_RECOGNITION_THRESHOLD:
                    name = self.valid_class_names[best_match_index].upper()
                else:
                    name = UNKNOWN_FACE_LABEL
                
                top, right, bottom, left = face_location
                bbox = (left, top, right, bottom)
                
                # Find or create tracker
                tracker = next((t for t in self.face_trackers if t.name == name), None)
                if tracker:
                    tracker.update(bbox)
                else:
                    tracker = FaceTracker(name, bbox)
                    if name != UNKNOWN_FACE_LABEL:
                        mark_attendance(name)
                
                updated_trackers.append(tracker)
            
            # Remove old trackers
            self.face_trackers = [t for t in updated_trackers if t.time_since_update < 5]
            
            # Create QImage and QPainter
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw face boxes and names
            for tracker in self.face_trackers:
                left, top, right, bottom = tracker.bbox
                color = Qt.green if tracker.name != UNKNOWN_FACE_LABEL else Qt.red
                
                pen = QPen(color, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(left, top, right - left, bottom - top)
                
                painter.setFont(QFont("Arial", 12))
                painter.drawText(left + 6, top - 6, tracker.name)
            
            painter.end()
            
            self.image_label.setPixmap(pixmap)
    
    def update_attendance_list(self):
        try:
            with open(ATTENDANCE_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                today = datetime.now().strftime('%Y-%m-%d')
                attendance = [row[0] for row in reader if row[1] == today]
            
            attendance_text = "\n".join(attendance)
            self.attendance_list.setText(attendance_text)
        except Exception as e:
            logging.error(f"Error updating attendance list: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Load custom font
    QFontDatabase.addApplicationFont("path/to/your/modern/font.ttf")
    app.setFont(QFont("Your Modern Font Name", 10))
    
    window = FaceRecognitionApp()
    window.setGeometry(100, 100, 1200, 800)
    window.show()
    sys.exit(app.exec_())