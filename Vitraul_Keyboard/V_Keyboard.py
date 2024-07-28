import cv2
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Controller
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize the keyboard controller
keyboard = Controller()

# Define the keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
        ["SPACE", "BACKSPACE", "CLEAR"]]

# Button class
class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

# Create button list
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if key in ["SPACE", "BACKSPACE", "CLEAR"]:
            buttonList.append(Button([j * 390 + 25, 400], key, size=[380, 85]))
        else:
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

# Variables for smooth typing
finalText = ""
lastClickTime = 0
clickDelay = 0.3  # Delay between clicks in seconds
lastKey = ""
keyCount = 0
doubleClickThreshold = 0.5  # Time threshold for double click in seconds

# Function to draw all buttons
def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(imgNew, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    
    out = img.copy()
    alpha = 0.3
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=False)
    
    img = drawAll(img, buttonList)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                l, _, _ = detector.findDistance((lmList[8][0], lmList[8][1]), (lmList[12][0], lmList[12][1]), img)

                if l < 30 and time.time() - lastClickTime > clickDelay:
                    if button.text == "SPACE":
                        finalText += " "
                        keyboard.press(' ')
                    elif button.text == "BACKSPACE":
                        finalText = finalText[:-1]
                        keyboard.press('\b')
                    elif button.text == "CLEAR":
                        finalText = ""
                    else:
                        if time.time() - lastClickTime < doubleClickThreshold and button.text == lastKey:
                            keyCount += 1
                            finalText = finalText[:-1]  # Remove the last character
                        else:
                            keyCount = 0
                        finalText += button.text
                        keyboard.press(button.text)
                    
                    lastClickTime = time.time()
                    lastKey = button.text

    # Display the text
    cv2.rectangle(img, (50, 550), (1230, 650), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Calculate and display FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Virtual Keyboard", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()