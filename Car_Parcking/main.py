import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not
import time
from collections import deque
from datetime import datetime

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

def create_stylish_dashboard(frame, available_spots, total_spots, fps, time_elapsed, hourly_data):
    dashboard_width = 700
    dashboard = np.zeros((frame.shape[0], dashboard_width, 3), dtype=np.uint8)
    
    # Background
    dashboard[:] = (25, 25, 25)  # Dark background
    
    # Font settings
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale_title = 1.5
    font_scale_large = 1.2
    font_scale_medium = 0.8
    font_scale_small = 0.6
    thickness_large = 2
    thickness_medium = 1
    thickness_small = 1
    line_type = cv2.LINE_AA
    
    # Title
    cv2.putText(dashboard, "Smart Parking System", (20, 60), font, font_scale_title, (255, 255, 255), thickness_large, line_type)
    cv2.line(dashboard, (20, 80), (680, 80), (100, 100, 100), 2)
    
    # Available spots
    cv2.rectangle(dashboard, (20, 100), (340, 240), (45, 45, 45), -1)
    cv2.putText(dashboard, "Available Spots", (30, 140), font, font_scale_medium, (200, 200, 200), thickness_medium, line_type)
    cv2.putText(dashboard, f"{available_spots}", (30, 210), font, font_scale_large * 2, (0, 255, 0), thickness_large, line_type)
    
    # Occupancy rate
    cv2.rectangle(dashboard, (360, 100), (680, 240), (45, 45, 45), -1)
    occupancy_rate = (total_spots - available_spots) / total_spots * 100
    cv2.putText(dashboard, "Occupancy Rate", (370, 140), font, font_scale_medium, (200, 200, 200), thickness_medium, line_type)
    cv2.putText(dashboard, f"{occupancy_rate:.1f}%", (370, 210), font, font_scale_large * 2, (0, 140, 255), thickness_large, line_type)
    
    # Progress bar
    bar_width = 640
    bar_height = 40
    filled_width = int(bar_width * occupancy_rate / 100)
    cv2.rectangle(dashboard, (30, 260), (30 + bar_width, 260 + bar_height), (50, 50, 50), -1)
    cv2.rectangle(dashboard, (30, 260), (30 + filled_width, 260 + bar_height), (0, 140, 255), -1)
    
    # Additional stats
    cv2.putText(dashboard, f"Total Spots: {total_spots}", (30, 340), font, font_scale_medium, (200, 200, 200), thickness_small, line_type)
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (30, 380), font, font_scale_medium, (200, 200, 200), thickness_small, line_type)
    
    # Time and Date
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    cv2.putText(dashboard, f"Time: {current_time}", (370, 340), font, font_scale_medium, (200, 200, 200), thickness_small, line_type)
    cv2.putText(dashboard, f"Date: {current_date}", (370, 380), font, font_scale_medium, (200, 200, 200), thickness_small, line_type)
    
    # Hourly Occupancy Graph
    cv2.putText(dashboard, "Hourly Occupancy", (30, 440), font, font_scale_medium, (255, 255, 255), thickness_medium, line_type)
    graph_height = 200
    graph_width = 640
    cv2.rectangle(dashboard, (30, 460), (30 + graph_width, 460 + graph_height), (50, 50, 50), 1)
    
    if hourly_data:
        max_occupancy = max(max(hourly_data), 1)  # Avoid division by zero
        bar_width = graph_width // 24
        for i, occupancy in enumerate(hourly_data):
            bar_height = int((occupancy / max_occupancy) * graph_height)
            cv2.rectangle(dashboard, 
                          (30 + i * bar_width, 460 + graph_height - bar_height),
                          (30 + (i+1) * bar_width - 2, 460 + graph_height),
                          (0, 255, 255), -1)
        
        # Add x-axis labels (hours)
        for i in range(0, 25, 4):
            x = 30 + (i * bar_width)
            cv2.putText(dashboard, f"{i:02d}", (x, 460 + graph_height + 25), font, font_scale_small, (200, 200, 200), thickness_small, line_type)
    
    # System status
    status_color = (0, 255, 0)  # Green for normal operation
    cv2.putText(dashboard, "System Status: Normal", (30, frame.shape[0] - 30), font, font_scale_medium, status_color, thickness_medium, line_type)
    
    return dashboard

def main():
    mask = 'mask_1920_1080.png'
    video_path = 'parking_1920_1080_loop.mp4'

    mask = cv2.imread(mask, 0)
    cap = cv2.VideoCapture(video_path)

    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)
    spots_status = [None for _ in spots]
    diffs = [None for _ in spots]

    previous_frame = None
    frame_nmr = 0
    ret = True
    step = 30
    
    fps = 0
    start_time = time.time()
    frame_count = 0

    hourly_data = deque([0] * 24, maxlen=24)  # Initialize with zeros
    last_hour = -1

    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()
        
        if frame_nmr % step == 0 and previous_frame is not None:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        
        if frame_nmr % step == 0:
            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            for spot_indx in arr_:
                spot = spots[spot_indx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_indx] = spot_status
        
        if frame_nmr % step == 0:
            previous_frame = frame.copy()
        
        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spots[spot_indx]
            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Green for empty
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)  # Red for occupied
        
        available_spots = sum(spots_status)
        total_spots = len(spots_status)
        
        time_elapsed = time.time() - start_time
        
        # Update hourly data
        current_hour = int(time_elapsed / 3600) % 24
        if current_hour != last_hour:
            hourly_data[current_hour] = total_spots - available_spots
            last_hour = current_hour
        
        dashboard = create_stylish_dashboard(frame, available_spots, total_spots, fps, time_elapsed, hourly_data)
        
        # Resize the frame to match the dashboard height
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * dashboard.shape[0] / frame.shape[0]), dashboard.shape[0]))
        
        # Combine frame and dashboard
        combined_frame = np.hstack((frame_resized, dashboard))
        
        cv2.namedWindow('Smart Parking Monitoring', cv2.WINDOW_NORMAL)
        cv2.imshow('Smart Parking Monitoring', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_nmr += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
