import cv2
import numpy as np
import time
import threading
from queue import Queue

# Global variables
frame_queue = Queue(maxsize=10)  # Queue for holding frames

def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera at index {camera_index} cannot be opened.")
    return cap

def capture_frames(camera_index):
    cap = initialize_camera(camera_index)
    while True:
        ret, captured_frame = cap.read()
        if ret:
            if not frame_queue.full():  # Only add if the queue is not full
                frame_queue.put(captured_frame)  # Put the captured frame in the queue
        else:
            print("Failed to capture frame. Reinitializing camera...")
            cap.release()
            cap = initialize_camera(camera_index)

def detect_blue_object(frame):
    # Apply bilateral filter to smooth the image
    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 150])  # Adjust if needed
    upper_blue = np.array([130, 255, 255])  # Adjust if needed

    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []

    object_count = 1  # Reset object count for each frame
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Adjust threshold if needed
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detected_area = frame[y:y + h, x:x + w]
        avg_color = cv2.mean(detected_area)[:3]
        if avg_color[0] < avg_color[1] and avg_color[0] < avg_color[2]:
            continue
        center_x = x + w // 2
        center_y = y + h // 2

        # Check for overlapping bounding boxes
        if not any((abs(center_x - coord[0]) < w and abs(center_y - coord[1]) < h) for coord in coordinates):
            coordinates.append((center_x, center_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            object_count += 1

    # Calculate midpoint for detected objects
    if len(coordinates) == 2:
        mid_x = sum(coord[0] for coord in coordinates) // 2
        mid_y = sum(coord[1] for coord in coordinates) // 2
        print(f"Object 1: ({coordinates[0][0]}, {coordinates[0][1]})")
        print(f"Object 2: ({coordinates[1][0]}, {coordinates[1][1]})")
        print(f"Midpoint: ({mid_x}, {mid_y})")

    elif len(coordinates) == 1:
        print("One object is detected.")
    else:
        print("No objects detected.")

    return frame

# Function to draw the screen midpoint
def screen_midpoint(frame):
    height, width, _ = frame.shape
    screen_mid_x = width // 2
    screen_mid_y = height // 2
    cv2.circle(frame, (screen_mid_x, screen_mid_y), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"screen_mid_point:({screen_mid_x},{screen_mid_y})", (screen_mid_x + 10, screen_mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def process_frames():
    frame_count = 0
    start_time = time.time()

    while True:
        if not frame_queue.empty():  # Only process if there's a frame in the queue
            frame = frame_queue.get()  # Get the frame from the queue
            processed_frame = detect_blue_object(frame)
            processed_frame = screen_midpoint(processed_frame)

            # Count frames
            frame_count += 1
            elapsed_time = time.time() - start_time

            # Calculate FPS and reset every second
            if elapsed_time >= 1.0: 
                print(f"Frames in one second: {frame_count}")
                cv2.putText(processed_frame, f"FPS: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                frame_count = 0  # Reset frame count
                start_time = time.time()  # Reset the start time

            cv2.imshow('Detected Blue Objects', processed_frame)

            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Start the capture and processing threads
camera_index = 1
capture_thread = threading.Thread(target=capture_frames, args=(camera_index,))
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

# Wait for threads to finish (this won't happen in normal execution since the while loops are infinite)
capture_thread.join()
process_thread.join()

# Release resources when done
cv2.destroyAllWindows()
