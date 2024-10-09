import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(1)  # Default 0 for in-built camera, 1 for external camera
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()

stop_flag = False  # Global flag to indicate when to stop the loop

# Define a function to detect blue objects in the frame
def detect_blue_object(frame):
    # Apply bilateral filter to smooth the image
    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 150])  # Adjust if needed
    upper_blue = np.array([140, 255, 255])  # Adjust if needed

    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []

    object_count = 1  # Reset object count for each frame
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Adjust threshold if needed
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
    if coordinates:
        mid_x = sum(coord[0] for coord in coordinates) // len(coordinates)
        mid_y = sum(coord[1] for coord in coordinates) // len(coordinates)
        cv2.circle(frame, (mid_x, mid_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Midpoint: ({mid_x},{mid_y})", (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

# Function to draw the screen midpoint
def screen_midpoint(frame):
    height, width, channel = frame.shape
    screen_mid_x = width // 2
    screen_mid_y = height // 2
    cv2.circle(frame, (screen_mid_x, screen_mid_y), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"screen_mid_point:({screen_mid_x},{screen_mid_y})", (screen_mid_x + 10, screen_mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# Initialize frame counting
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret or stop_flag:
        break  # Exit the loop if the stop flag is set

    frame = detect_blue_object(frame)
    frame = screen_midpoint(frame)

    # Count frames
    frame_count += 1
    elapsed_time = time.time() - start_time

    # Calculate FPS and reset every second
    if elapsed_time >= 1.0:
        print(f"Frames in one second: {frame_count}")
        # Display FPS on the frame
        #cv2.putText(frame, f"FPS: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        frame_count = 0  # Reset frame count
        start_time = time.time()  # Reset the start time

    # Convert BGR to RGB for displaying in Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Clear the previous plot and display the new frame
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis('off')  # Hide axes
    plt.pause(0.001)  # Pause to update the plot

# Release the camera and close the OpenCV window
cap.release()
plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot (if any)
