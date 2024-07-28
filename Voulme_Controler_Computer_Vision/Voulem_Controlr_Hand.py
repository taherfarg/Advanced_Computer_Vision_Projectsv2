import cv2
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import colorsys

class VolumeController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.volume = self.setup_audio()
        self.volume_range = self.volume.GetVolumeRange()
        self.min_vol, self.max_vol = self.volume_range[0], self.volume_range[1]
        
        self.min_length = 30
        self.max_length = 300
        
        # UI colors
        self.bg_color = (25, 25, 25)
        self.accent_color = (0, 255, 255)  # Cyan
        self.text_color = (255, 255, 255)  # White
        
        # New features
        self.mute_toggle = False
        self.previous_vol = 0

    def setup_audio(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Apply a dark overlay to modernize the look
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.bg_color, -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, 
                                        landmark_drawing_spec=self.mp_draw.DrawingSpec(color=self.accent_color, thickness=2, circle_radius=4),
                                        connection_drawing_spec=self.mp_draw.DrawingSpec(color=self.text_color, thickness=2))
            
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            x1, y1 = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            x2, y2 = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            
            # Calculate the midpoint between thumb and index finger
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            length = np.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [self.min_length, self.max_length], [self.min_vol, self.max_vol])
            vol_percentage = np.interp(length, [self.min_length, self.max_length], [0, 100])
            
            if not self.mute_toggle:
                self.volume.SetMasterVolumeLevel(vol, None)
            
            self.draw_volume_control(frame, x1, y1, x2, y2, cx, cy, vol_percentage)
            
            # Check for mute gesture (thumb and pinky touching)
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            pinky_x, pinky_y = int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0])
            if np.hypot(x1 - pinky_x, y1 - pinky_y) < 20:
                self.toggle_mute()
        
        self.draw_mute_indicator(frame)
        return frame

    def draw_volume_control(self, frame, x1, y1, x2, y2, cx, cy, vol_percentage):
        # Draw line between thumb and index finger
        cv2.line(frame, (x1, y1), (x2, y2), self.accent_color, 3)
        
        # Draw circles at thumb and index fingertips
        cv2.circle(frame, (x1, y1), 10, self.accent_color, cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, self.accent_color, cv2.FILLED)
        
        # Calculate color based on volume percentage
        if vol_percentage == 0:
            color = (255, 255, 255)  # White for 0%
        elif vol_percentage == 100:
            color = (0, 0, 255)      # Red for 100%
        else:
            # Green with varying intensity for volumes in between
            hue = 120 / 360  # Green hue
            saturation = 1.0
            value = 0.5 + (vol_percentage / 200)  # Vary from 0.5 to 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            color = (int(b * 255), int(g * 255), int(r * 255))
        
        # Draw the control point between the fingers
        cv2.circle(frame, (cx, cy), 8, color, cv2.FILLED)
        cv2.circle(frame, (cx, cy), 12, self.accent_color, 2)
        
        bar_start_x, bar_end_x = 50, 85
        bar_top_y, bar_bottom_y = 150, 400
        vol_bar_height = int(np.interp(vol_percentage, [0, 100], [bar_bottom_y, bar_top_y]))
        
        # Draw background bar
        cv2.rectangle(frame, (bar_start_x, bar_top_y), (bar_end_x, bar_bottom_y), self.text_color, 3)
        # Draw filled volume bar
        cv2.rectangle(frame, (bar_start_x, vol_bar_height), (bar_end_x, bar_bottom_y), color, cv2.FILLED)
        
        # Add volume percentage text
        cv2.putText(frame, f'{int(vol_percentage)}%', (bar_start_x - 10, bar_bottom_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2)
        
        # Draw a line from the control point to the volume bar
        cv2.line(frame, (cx, cy), (bar_end_x, vol_bar_height), color, 2)

    def toggle_mute(self):
        self.mute_toggle = not self.mute_toggle
        if self.mute_toggle:
            self.previous_vol = self.volume.GetMasterVolumeLevelScalar()
            self.volume.SetMasterVolumeLevelScalar(0, None)
        else:
            self.volume.SetMasterVolumeLevelScalar(self.previous_vol, None)

    def draw_mute_indicator(self, frame):
        mute_text = "MUTED" if self.mute_toggle else "UNMUTED"
        mute_color = (0, 0, 255) if self.mute_toggle else (0, 255, 0)
        cv2.putText(frame, mute_text, (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, mute_color, 2)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    controller = VolumeController()
    prev_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        frame = controller.process_frame(frame)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, controller.text_color, 2)
        
        cv2.imshow("Volume Control", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()