import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math

# Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.6, maxHands=1)

# Colors
BLUE = (205, 0, 0)
GREEN = (0, 205, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class PlayerIcon():
    def __init__(self, posCenter, id, size=[90, 90]):
        self.posCenter = posCenter
        self.id = id
        self.size = size
        self.isDragging = False
        self.color = BLUE
        self.alpha = 0.5
        self.target_alpha = 0.5
        self.isSelected = False

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.isDragging = True
            self.posCenter = cursor[:2]
            self.target_alpha = 1.0
            self.isSelected = True
        else:
            self.isDragging = False
            self.target_alpha = 0.5
            self.isSelected = False
        
        self.alpha += (self.target_alpha - self.alpha) * 0.1

class FingerTracker:
    def __init__(self, smoothing=0.5):
        self.prev_positions = {}
        self.smoothing = smoothing

    def update(self, lmList):
        current_positions = {}
        for id in [8, 12]:
            if id < len(lmList):
                if id not in self.prev_positions:
                    self.prev_positions[id] = lmList[id][:2]
                else:
                    current = np.array(lmList[id][:2])
                    prev = np.array(self.prev_positions[id])
                    smoothed = prev * self.smoothing + current * (1 - self.smoothing)
                    self.prev_positions[id] = smoothed.astype(int)
                current_positions[id] = self.prev_positions[id]
        return current_positions

def draw_finger_info(img, finger_positions, l):
    if 8 in finger_positions and 12 in finger_positions:
        start = tuple(finger_positions[8])
        end = tuple(finger_positions[12])

        for point in [start, end]:
            cv2.circle(img, point, 15, WHITE, -1)
            cv2.circle(img, point, 12, BLACK, -1)
            cv2.circle(img, point, 9, WHITE, -1)
        
        cv2.line(img, start, end, WHITE, 4)
        cv2.line(img, start, end, BLACK, 2)
        
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(img, f"{l:.0f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 4)
        cv2.putText(img, f"{l:.0f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

def draw_football_pitch(img):
    h, w = img.shape[:2]
    
    cv2.rectangle(img, (50, 50), (w-50, h-50), WHITE, 2)
    cv2.line(img, (w//2, 50), (w//2, h-50), WHITE, 2)
    cv2.circle(img, (w//2, h//2), 70, WHITE, 2)
    cv2.rectangle(img, (50, h//2-100), (200, h//2+100), WHITE, 2)
    cv2.rectangle(img, (w-200, h//2-100), (w-50, h//2+100), WHITE, 2)
    cv2.rectangle(img, (50, h//2-50), (100, h//2+50), WHITE, 2)
    cv2.rectangle(img, (w-100, h//2-50), (w-50, h//2+50), WHITE, 2)

playerList = [PlayerIcon([np.random.randint(100, 1180), np.random.randint(100, 620)], f"P{i+1}") for i in range(11)]
finger_tracker = FingerTracker(smoothing=0.5)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    img = cv2.flip(img, 1)
    
    draw_football_pitch(img)
    
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        if len(lmList) >= 21:
            finger_positions = finger_tracker.update(lmList)
            
            if 8 in finger_positions and 12 in finger_positions:
                l = np.linalg.norm(np.array(finger_positions[8]) - np.array(finger_positions[12]))
                
                draw_finger_info(img, finger_positions, l)
                
                if l < 50:
                    for player in playerList:
                        player.update(finger_positions[8])

    playerList.sort(key=lambda x: x.posCenter[1])

    for player in playerList:
        cx, cy = player.posCenter[:2]
        w, h = player.size
        
        icon_size = w//2 if not player.isSelected else w//2 + 10
        
        overlay = img.copy()
        cv2.circle(overlay, (int(cx), int(cy)), icon_size, player.color, -1)
        cv2.addWeighted(overlay, player.alpha, img, 1 - player.alpha, 0, img)
        
        cv2.circle(img, (int(cx), int(cy)), icon_size, WHITE, 2)
        
        cv2.putText(img, player.id, (int(cx-35), int(cy+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        if player.isSelected:
            cv2.circle(img, (int(cx), int(cy-icon_size-15)), 15, GREEN, -1)

    cv2.rectangle(img, (10, 10), (210, 60), BLACK, -1)
    cv2.rectangle(img, (10, 10), (210, 60), WHITE, 2)
    cv2.putText(img, f"Hands: {len(hands)}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    cv2.imshow("Football Tactics Board", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
