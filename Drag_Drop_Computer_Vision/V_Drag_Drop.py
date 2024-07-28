import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math

# Setup
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Colors
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class DragRect():
    def __init__(self, posCenter, size=[100, 100]):
        self.posCenter = posCenter
        self.size = size
        self.isDragging = False
        self.color = np.random.randint(0, 255, 3).tolist()
        self.alpha = 0.5
        self.target_alpha = 0.5

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.isDragging = True
            self.posCenter = cursor[:2]
            self.target_alpha = 1.0
        else:
            self.isDragging = False
            self.target_alpha = 0.5
        
        # Smooth alpha transition
        self.alpha += (self.target_alpha - self.alpha) * 0.1

class FingerTracker:
    def __init__(self, smoothing=0.5):
        self.prev_positions = {}
        self.smoothing = smoothing

    def update(self, lmList):
        current_positions = {}
        for id in [8, 12]:  # Index and middle fingertips
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

        # Draw circles on fingertips with gradient
        for point in [start, end]:
            cv2.circle(img, point, 15, WHITE, -1)
            cv2.circle(img, point, 12, BLACK, -1)
            cv2.circle(img, point, 9, WHITE, -1)
        
        # Draw line between fingertips
        cv2.line(img, start, end, WHITE, 4)
        cv2.line(img, start, end, BLACK, 2)
        
        # Display pinch distance
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(img, f"{l:.0f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 4)
        cv2.putText(img, f"{l:.0f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

rectList = [DragRect([np.random.randint(100, 1180), np.random.randint(100, 620)]) for _ in range(5)]
finger_tracker = FingerTracker(smoothing=0.5)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=False)

    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            if len(lmList) >= 21:
                finger_positions = finger_tracker.update(lmList)
                
                if 8 in finger_positions and 12 in finger_positions:
                    l = np.linalg.norm(np.array(finger_positions[8]) - np.array(finger_positions[12]))
                    
                    draw_finger_info(img, finger_positions, l)
                    
                    if l < 50:
                        for rect in rectList:
                            rect.update(finger_positions[8])

    # Draw Rectangles
    overlay = img.copy()
    for rect in rectList:
        cx, cy = rect.posCenter[:2]
        w, h = rect.size
        cv2.rectangle(overlay, (int(cx - w // 2), int(cy - h // 2)),
                      (int(cx + w // 2), int(cy + h // 2)), rect.color, -1)
    
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for rect in rectList:
        cx, cy = rect.posCenter[:2]
        w, h = rect.size
        cv2.rectangle(img, (int(cx - w // 2), int(cy - h // 2)),
                      (int(cx + w // 2), int(cy + h // 2)), WHITE, 2)

    # Display number of detected hands with a modern UI element
    cv2.rectangle(img, (10, 10), (210, 60), BLACK, -1)
    cv2.rectangle(img, (10, 10), (210, 60), WHITE, 2)
    cv2.putText(img, f"Hands: {len(hands)}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    cv2.imshow("Modern Hand Interaction", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()