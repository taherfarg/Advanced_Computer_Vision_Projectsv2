import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import json
from datetime import date

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set the camera window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize variables for counting reps
right_countr = 0
left_countr = 0
start_time = time.time()
display_text = False
text_display_time = 0

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Exercise parameters
exercise_state = "warm_up"
warm_up_time = 30  # 30 seconds warm-up
cool_down_time = 30  # 30 seconds cool-down
exercise_time = 300  # 5 minutes exercise
difficulty = "medium"  # Can be "easy", "medium", or "hard"
exercise_type = "bicep_curl"  # Can be "bicep_curl", "tricep_extension", "shoulder_press"

# Calorie counter
calories_per_curl = 0.32  # Approximate calories burned per curl
total_calories = 0

# Set curl threshold based on difficulty
if difficulty == "easy":
    curl_threshold = 45
elif difficulty == "hard":
    curl_threshold = 15
else:  # medium
    curl_threshold = 30

# Initialize stage variables
right_stage = "down"
left_stage = "down"

# Initialize variables for audio feedback
right_audio_milestone = 5
left_audio_milestone = 5

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Process the frame with MediaPipe Pose
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = pose.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Initialize angles
    r_angle = 0
    l_angle = 0

    # Extract landmarks and calculate angles
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Right arm landmarks
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Draw lines and points for right arm
        cv2.line(frame, (int(r_shoulder[0] * frame.shape[1]), int(r_shoulder[1] * frame.shape[0])),
                 (int(r_elbow[0] * frame.shape[1]), int(r_elbow[1] * frame.shape[0])), (0, 0, 255), 2)
        cv2.line(frame, (int(r_elbow[0] * frame.shape[1]), int(r_elbow[1] * frame.shape[0])),
                 (int(r_wrist[0] * frame.shape[1]), int(r_wrist[1] * frame.shape[0])), (0, 0, 255), 2)
        cv2.circle(frame, (int(r_shoulder[0] * frame.shape[1]), int(r_shoulder[1] * frame.shape[0])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(r_elbow[0] * frame.shape[1]), int(r_elbow[1] * frame.shape[0])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(r_wrist[0] * frame.shape[1]), int(r_wrist[1] * frame.shape[0])), 5, (0, 0, 255), -1)

        # Left arm landmarks
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Draw lines and points for left arm
        cv2.line(frame, (int(l_shoulder[0] * frame.shape[1]), int(l_shoulder[1] * frame.shape[0])),
                 (int(l_elbow[0] * frame.shape[1]), int(l_elbow[1] * frame.shape[0])), (0, 0, 255), 2)
        cv2.line(frame, (int(l_elbow[0] * frame.shape[1]), int(l_elbow[1] * frame.shape[0])),
                 (int(l_wrist[0] * frame.shape[1]), int(l_wrist[1] * frame.shape[0])), (0, 0, 255), 2)
        cv2.circle(frame, (int(l_shoulder[0] * frame.shape[1]), int(l_shoulder[1] * frame.shape[0])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(l_elbow[0] * frame.shape[1]), int(l_elbow[1] * frame.shape[0])), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(l_wrist[0] * frame.shape[1]), int(l_wrist[1] * frame.shape[0])), 5, (0, 0, 255), -1)

        # Calculate angles
        try:
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        except:
            print("Failed to calculate angles")
            continue

        # Curl counter logic
        if exercise_type == "bicep_curl":
            # Right arm
            if r_angle > 160:
                right_stage = "down"
            if r_angle < curl_threshold and right_stage == 'down':
                right_stage = "up"
                right_countr += 1
                print("Right count:", right_countr)
                
                # Audio feedback for right arm
                if right_countr == right_audio_milestone:
                    engine.say(f"Great job! You've completed {right_countr} right arm curls")
                    engine.runAndWait()
                    right_audio_milestone += 5
            
            # Left arm
            if l_angle > 160:
                left_stage = "down"
            if l_angle < curl_threshold and left_stage == 'down':
                left_stage = "up"
                left_countr += 1
                print("Left count:", left_countr)
                
                # Audio feedback for left arm
                if left_countr == left_audio_milestone:
                    engine.say(f"Excellent! You've completed {left_countr} left arm curls")
                    engine.runAndWait()
                    left_audio_milestone += 5

        elif exercise_type == "tricep_extension":
            # Right arm
            if r_angle < 45:
                right_stage = "down"
            if r_angle > 160 and right_stage == 'down':
                right_stage = "up"
                right_countr += 1
                print("Right count:", right_countr)
                
                # Audio feedback for right arm
                if right_countr == right_audio_milestone:
                    engine.say(f"Great job! You've completed {right_countr} right and left tricep extensions")
                    engine.runAndWait()
                    right_audio_milestone += 12
            
            # Left arm
            if l_angle < 45:
                left_stage = "down"
            if l_angle > 160 and left_stage == 'down':
                left_stage = "up"
                left_countr += 1
                print("Left count:", left_countr)

        elif exercise_type == "shoulder_press":
            # Implement logic for shoulder press
            # You'll need to use different landmarks and angle calculations
            pass

        # Update calorie count
        total_calories = (right_countr + left_countr) * calories_per_curl

    else:
        # Display "No arms detected" when no pose is detected
        cv2.putText(frame, "No arms detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw UI elements
    cv2.rectangle(frame, (0, 0), (300, 240), (52, 73, 94), -1)
    cv2.putText(frame, 'REPS', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Right: {right_countr}  Left: {left_countr}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"State: {exercise_state.capitalize()}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Calories: {total_calories:.2f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Visual feedback for curl form
    if r_angle > 160:
        cv2.putText(frame, "Right: Ready", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    elif r_angle < curl_threshold:
        cv2.putText(frame, "Right: Good!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Right: Keep going", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    if l_angle > 160:
        cv2.putText(frame, "Left: Ready", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    elif l_angle < curl_threshold:
        cv2.putText(frame, "Left: Good!", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Left: Keep going", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    # Exercise state management
    current_time = time.time()
    if exercise_state == "warm_up" and current_time - start_time > warm_up_time:
        exercise_state = "exercise"
        start_time = current_time
    elif exercise_state == "exercise" and current_time - start_time > exercise_time:
        exercise_state = "cool_down"
        start_time = current_time
    elif exercise_state == "cool_down" and current_time - start_time > cool_down_time:
        break  # End the session

    # Display the frame
    cv2.imshow('Exercise Assistant', frame)
    
    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Save progress
progress = {
    "date": str(date.today()),
    "right_curls": right_countr,
    "left_curls": left_countr,
    "calories": total_calories
}

# Load existing progress data
try:
    with open("progress.json", "r") as f:
        progress_data = json.load(f)
except FileNotFoundError:
    progress_data = []

# Append new progress and save
progress_data.append(progress)
with open("progress.json", "w") as f:
    json.dump(progress_data, f)

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()