import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import math

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the hand detector with improved parameters
detector = HandDetector(detectionCon=0.7, maxHands=2)

# Define the calculator layout with more operations
keys = [["(", ")", "^", "√", "!"],
        ["7", "8", "9", "+", "sin"],
        ["4", "5", "6", "-", "cos"],
        ["1", "2", "3", "*", "tan"],
        ["C", "0", ".", "/", "="],
        ["DEL", "π", "e", "log", "exp"]]

# Define color scheme
BUTTON_COLOR = (45, 45, 45)
HIGHLIGHT_COLOR = (75, 75, 75)
TEXT_COLOR = (255, 255, 255)
RESULT_COLOR = (0, 255, 0)
ERROR_COLOR = (0, 0, 255)

# Button class with hover state
class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text
        self.is_hovering = False

# Create button list
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

# Variables for smooth operation
equation = ""
result = ""
lastClickTime = 0
clickDelay = 0.3  # Delay between clicks in seconds

# Function to draw all buttons
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        color = HIGHLIGHT_COLOR if button.is_hovering else BUTTON_COLOR
        cv2.rectangle(img, button.pos, (x + w, y + h), color, cv2.FILLED)
        cv2.rectangle(img, button.pos, (x + w, y + h), HIGHLIGHT_COLOR, 2)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)
    return img

# Improved calculate function with more operations
def calculate(equation):
    try:
        # Replace symbols with their Python equivalents
        equation = equation.replace('^', '**')
        equation = equation.replace('π', 'math.pi')
        equation = equation.replace('e', 'math.e')
        
        # Handle square root
        while '√' in equation:
            sqrt_index = equation.index('√')
            closing_paren = equation.find(')', sqrt_index)
            if closing_paren == -1:
                return "Error"
            sqrt_expr = equation[sqrt_index+1:closing_paren+1]
            equation = equation[:sqrt_index] + f"math.sqrt{sqrt_expr}" + equation[closing_paren+1:]
        
        # Handle factorial
        while '!' in equation:
            fact_index = equation.index('!')
            num_end = fact_index
            num_start = num_end - 1
            while num_start >= 0 and (equation[num_start].isdigit() or equation[num_start] == '.'):
                num_start -= 1
            num_start += 1
            num = int(equation[num_start:num_end])
            equation = equation[:num_start] + str(math.factorial(num)) + equation[fact_index+1:]
        
        # Handle trigonometric functions
        for func in ['sin', 'cos', 'tan']:
            equation = equation.replace(func, f'math.{func}')
        
        # Handle logarithm and exponential
        equation = equation.replace('log', 'math.log10')
        equation = equation.replace('exp', 'math.exp')
        
        return str(eval(equation))
    except Exception as e:
        return f"Error: {str(e)}"

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # Detect hands
    hands, img = detector.findHands(img)
    
    # Reset hover states
    for button in buttonList:
        button.is_hovering = False
    
    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    button.is_hovering = True
                    l, _, _ = detector.findDistance((lmList[8][0], lmList[8][1]), (lmList[12][0], lmList[12][1]), img)

                    if l < 30 and time.time() - lastClickTime > clickDelay:
                        if button.text == "C":
                            equation = ""
                            result = ""
                        elif button.text == "=":
                            result = calculate(equation)
                        elif button.text == "DEL":
                            equation = equation[:-1]
                        else:
                            equation += button.text
                        
                        lastClickTime = time.time()
    
    img = drawAll(img, buttonList)

    # Display the equation and result
    cv2.rectangle(img, (50, 550), (1230, 650), BUTTON_COLOR, cv2.FILLED)
    cv2.putText(img, equation, (60, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
    color = ERROR_COLOR if result.startswith("Error") else RESULT_COLOR
    cv2.putText(img, result, (60, 640),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Calculate and display FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, RESULT_COLOR, 2)

    cv2.imshow("Advanced Virtual Calculator", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()