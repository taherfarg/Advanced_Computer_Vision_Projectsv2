import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import pickle

# Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Colors
BLUE = (255, 150, 0)
GREEN = (0, 255, 100)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
PITCH_LINES = (200, 255, 200)


class PlayerIcon:
    def __init__(self, pos_center, id, size=(80, 80), team="A"):
        self.pos_center = pos_center
        self.id = id
        self.size = size
        self.is_dragging = False
        self.color = RED if team == "A" else BLUE
        self.alpha = 0.7
        self.target_alpha = 0.7
        self.is_selected = False
        self.animation_progress = 0
        self.team = team
        self.connection = None

    def update(self, cursor):
        cx, cy = self.pos_center
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.is_dragging = True
            self.pos_center = cursor[:2]
            self.target_alpha = 1.0
            self.is_selected = True
            self.animation_progress = min(1, self.animation_progress + 0.1)
        else:
            self.is_dragging = False
            self.target_alpha = 0.7
            self.is_selected = False
            self.animation_progress = max(0, self.animation_progress - 0.1)

        self.alpha += (self.target_alpha - self.alpha) * 0.1


class Ball:
    def __init__(self, pos_center):
        self.pos_center = pos_center
        self.is_dragging = False
        self.size = 30

    def update(self, cursor):
        cx, cy = self.pos_center
        if cx - self.size < cursor[0] < cx + self.size and cy - self.size < cursor[1] < cy + self.size:
            self.is_dragging = True
            self.pos_center = cursor[:2]
        else:
            self.is_dragging = False


def draw_football_pitch(img):
    h, w = img.shape[:2]
    cv2.line(img, (w // 2, 0), (w // 2, h), PITCH_LINES, 2)
    cv2.circle(img, (w // 2, h // 2), 70, PITCH_LINES, 2)
    cv2.rectangle(img, (0, h // 2 - 100), (150, h // 2 + 100), PITCH_LINES, 2)
    cv2.rectangle(img, (w - 150, h // 2 - 100), (w, h // 2 + 100), PITCH_LINES, 2)
    cv2.rectangle(img, (0, h // 2 - 50), (50, h // 2 + 50), PITCH_LINES, 2)
    cv2.rectangle(img, (w - 50, h // 2 - 50), (w, h // 2 + 50), PITCH_LINES, 2)


def create_button(img, text, position, size, color, text_color):
    x, y = position
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), WHITE, 2)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    return x, y, x + w, y + h


def save_layout(player_list, ball):
    with open('layout.pkl', 'wb') as f:
        pickle.dump((player_list, ball), f)


def load_layout():
    try:
        with open('layout.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


player_list = [PlayerIcon([np.random.randint(100, 1180), np.random.randint(100, 620)], f"A{i + 1}", team="A") for i in range(6)] + \
              [PlayerIcon([np.random.randint(100, 1180), np.random.randint(100, 620)], f"B{i + 1}", team="B") for i in range(6)]
ball = Ball([640, 360])

mode = "Move"
last_mode_change = time.time()
connection_start = None

layout = load_layout()
if layout:
    player_list, ball = layout

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    img = cv2.flip(img, 1)
    draw_football_pitch(img)

    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        lm_list = hand["lmList"]
        if len(lm_list) >= 21:
            cursor = lm_list[8][:2]

            if mode == "Move":
                for player in player_list:
                    player.update(cursor)
                ball.update(cursor)
            elif mode == "Connect":
                for player in player_list:
                    if player.is_selected:
                        if connection_start is None:
                            connection_start = player
                        else:
                            connection_start.connection = player
                            connection_start = None

    player_list.sort(key=lambda x: x.pos_center[1])

    # Draw connections
    for player in player_list:
        if player.connection:
            cv2.line(img, tuple(map(int, player.pos_center)), tuple(map(int, player.connection.pos_center)), YELLOW, 2)

    # Draw Player Icons
    for player in player_list:
        cx, cy = player.pos_center
        w, h = player.size
        icon_size = int(w // 2 + player.animation_progress * 10)

        overlay = img.copy()
        cv2.circle(overlay, (int(cx), int(cy)), icon_size, player.color, -1)
        cv2.addWeighted(overlay, player.alpha, img, 1 - player.alpha, 0, img)

        cv2.circle(img, (int(cx), int(cy)), icon_size, WHITE, 2)

        text_size = cv2.getTextSize(player.id, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_w, text_h = text_size
        cv2.rectangle(img, (int(cx - text_w // 2 - 5), int(cy - text_h // 2 - 5)),
                      (int(cx + text_w // 2 + 5), int(cy + text_h // 2 + 5)), BLACK, -1)
        cv2.putText(img, player.id, (int(cx - text_w // 2), int(cy + text_h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        if player.is_selected:
            selection_progress = player.animation_progress
            selection_radius = int(15 + selection_progress * 5)
            cv2.circle(img, (int(cx), int(cy - icon_size - 20)), selection_radius, GREEN, -1)
            cv2.circle(img, (int(cx), int(cy - icon_size - 20)), selection_radius - 2, WHITE, 1)

    # Draw Ball
    cv2.circle(img, tuple(map(int, ball.pos_center)), ball.size, WHITE, -1)
    cv2.circle(img, tuple(map(int, ball.pos_center)), ball.size - 2, BLACK, 2)

    # Display modern UI element for hand detection
    cv2.rectangle(img, (10, 10), (210, 60), (*BLACK, 150), -1)
    cv2.rectangle(img, (10, 10), (210, 60), WHITE, 2)
    cv2.putText(img, f"Hands: {len(hands)}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    # Add a title to the board
    cv2.putText(img, "Modern Football Tactics", (img.shape[1] // 2 - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)

    # Add mode buttons
    move_button = create_button(img, "Move", (10, img.shape[0] - 60), (100, 50), GREEN if mode == "Move" else BLACK, WHITE)
    connect_button = create_button(img, "Connect", (120, img.shape[0] - 60), (100, 50), GREEN if mode == "Connect" else BLACK, WHITE)
    reset_button = create_button(img, "Reset", (230, img.shape[0] - 60), (100, 50), BLACK, WHITE)
    save_button = create_button(img, "Save", (340, img.shape[0] - 60), (100, 50), BLACK, WHITE)
    load_button = create_button(img, "Load", (450, img.shape[0] - 60), (100, 50), BLACK, WHITE)

    # Handle button clicks
    if hands:
        cursor = hands[0]["lmList"][8][:2]
        if move_button[0] < cursor[0] < move_button[2] and move_button[1] < cursor[1] < move_button[3]:
            mode = "Move"
        elif connect_button[0] < cursor[0] < connect_button[2] and connect_button[1] < cursor[1] < connect_button[3]:
            mode = "Connect"
        elif reset_button[0] < cursor[0] < reset_button[2] and reset_button[1] < cursor[1] < reset_button[3]:
            player_list = [PlayerIcon([np.random.randint(100, 1180), np.random.randint(100, 620)], f"A{i + 1}", team="A") for i in range(6)] + \
                          [PlayerIcon([np.random.randint(100, 1180), np.random.randint(100, 620)], f"B{i + 1}", team="B") for i in range(6)]
            ball = Ball([640, 360])
        elif save_button[0] < cursor[0] < save_button[2] and save_button[1] < cursor[1] < save_button[3]:
            save_layout(player_list, ball)
        elif load_button[0] < cursor[0] < load_button[2] and load_button[1] < cursor[1] < load_button[3]:
            layout = load_layout()
            if layout:
                player_list, ball = layout

    cv2.imshow("Modern Football Tactics Board", img)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('m'):
        if time.time() - last_mode_change > 0.5:
            mode = "Move"
            last_mode_change = time.time()
    elif key & 0xFF == ord('c'):
        if time.time() - last_mode_change > 0.5:
            mode = "Connect"
            last_mode_change = time.time()

cap.release()
cv2.destroyAllWindows()
