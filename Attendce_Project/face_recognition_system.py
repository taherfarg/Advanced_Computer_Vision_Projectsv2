import cv2
import numpy as np
import face_recognition
import dlib
from datetime import datetime
import csv
import os

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        self.cap = None
        self.tracker = dlib.correlation_tracker()
        self.tracking = False
        self.current_name = None

    def load_known_faces(self):
        known_faces_dir = 'Images'
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
        
        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(known_faces_dir, filename)
                name = os.path.splitext(filename)[0]
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    self.known_face_encodings.append(encoding[0])
                    self.known_face_names.append(name)
        
        print(f"Loaded {len(self.known_face_names)} known faces.")

    def add_known_face(self, image_path, name):
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            self.known_face_encodings.append(encoding[0])
            self.known_face_names.append(name)
            print(f"Added {name} to known faces.")
        else:
            print(f"No face found in the image for {name}.")

    def start_camera(self):
        self.cap = cv2.VideoCapture(1)

    def stop_camera(self):
        if self.cap:
            self.cap.release()

    def process_frame(self):
        if not self.cap:
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if not self.tracking:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                name = "Unknown"
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                self.tracker.start_track(frame, dlib.rectangle(left, top, right, bottom))
                self.tracking = True
                self.current_name = name
                self.mark_attendance(name)
                break
        else:
            tracking_quality = self.tracker.update(frame)
            
            if tracking_quality >= 8.75:
                tracked_position = self.tracker.get_position()
                left = int(tracked_position.left())
                top = int(tracked_position.top())
                right = int(tracked_position.right())
                bottom = int(tracked_position.bottom())
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, self.current_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            else:
                self.tracking = False

        return frame, self.current_name

    def mark_attendance(self, name):
        with open('attendance.csv', 'a+', newline='') as file:
            writer = csv.writer(file)
            now = datetime.now()
            date_string = now.strftime('%Y-%m-%d')
            time_string = now.strftime('%H:%M:%S')
            writer.writerow([name, date_string, time_string])