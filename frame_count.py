"""import cv2
import time
import matplotlib.pyplot as plt
# Open the default camera (0 is usually the default camera index)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize frame counting
frame_count = 0
start_time = time.time()
while True:
    # Read a frame from the camera feed
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Increment the frame count
    frame_count += 1

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    # Once one second has passed, calculate the FPS and reset
    if elapsed_time >= 1.0:
        print(f"Frames in one second: {frame_count}")
        frame_count = 0
        start_time = time.time()  # Reset the start time

    # Display the frame (optional)
    cv2.imshow("frames",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
"""
import cv2
import numpy as np
import threading
import time

# Global variables
frame = None
lock = threading.Lock()

def read_camera(camera_index=1):
    global frame
    cap = cv2.VideoCapture(camera_index)
    
    while True:
        ret, new_frame = cap.read()
        if not ret:
            break
        
        # Lock the frame for thread safety
        with lock:
            frame = new_frame.copy()

    cap.release()

def detect_blue_object():
    global frame
    lower_blue = np.array([90, 150, 150])  # Adjust lower range as needed
    upper_blue = np.array([130, 255, 255])  # Adjust upper range as needed

    frame_count = 0
    start_time = time.time()
    
    while True:
        # Lock the frame for thread safety
        with lock:
            if frame is None:
                continue
            
            current_frame = frame.copy()  # Copy the current frame for processing

        # Process the frame to detect blue objects
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_count = 1
        coordinates = []

        for contour in contours:
            if cv2.contourArea(contour) < 1500:  # Increased threshold
                continue
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Check for overlapping bounding boxes
            if not any((abs(center_x - coord[0]) < w and abs(center_y - coord[1]) < h) for coord in coordinates):
                coordinates.append((center_x, center_y))
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(current_frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                object_count += 1

        # Calculate and display the FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            print(f"FPS: {frame_count}")  # Print FPS to console
            frame_count = 0  # Reset frame count
            start_time = time.time()  # Reset the start time
        cv2.putText(current_frame, f"FPS: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Detected Blue Objects', current_frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Create and start threads
camera_thread = threading.Thread(target=read_camera)
detection_thread = threading.Thread(target=detect_blue_object)

camera_thread.start()
detection_thread.start()

camera_thread.join()
detection_thread.join()
