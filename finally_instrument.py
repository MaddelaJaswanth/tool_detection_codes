import cv2
import numpy as np
import time

def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera at index {camera_index} cannot be opened.")
    return cap

cap = initialize_camera(1)  # Try to open the external camera at index 1
stop_flag = False  # Global flag to indicate when to stop the loop

# Define a function to detect blue objects in the frame
def detect_blue_object(frame):
    # Apply bilateral filter to smooth the image
    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a wider HSV range to capture both regular and light blue
    lower_blue = np.array([90, 100, 100])  # Lower bound for blue
    upper_blue = np.array([130, 255, 255])  # Upper bound for blue

    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    detection_radius = 50  # Minimum distance to consider as separate objects

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Adjust threshold if needed
            continue
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check if this center is far enough from previously detected objects
        is_far_enough = True
        for coord in coordinates:
            if abs(center_x - coord[0]) < detection_radius and abs(center_y - coord[1]) < detection_radius:
                is_far_enough = False
                break

        if is_far_enough:
            coordinates.append((center_x, center_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Detected ({center_x},{center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Print coordinates of detected objects if any
    if coordinates:
        print(f"Detected Object: {coordinates[0]}")  # Print the first detected object's coordinates

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
window_name = 'Full_camera_display'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    try:
        if not cap.isOpened():  # Check if the camera is still open
            print("Camera disconnected. Attempting to reconnect...")
            cap = initialize_camera(1)  # Reinitialize camera
            time.sleep(2)  # Wait before retrying

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Attempting to reconnect...")
            cap.release()  # Release the current capture
            cap = initialize_camera(1)  # Try to open the camera again
            continue  # Skip to the next iteration

        frame = detect_blue_object(frame)
        frame = screen_midpoint(frame)
        frame_resized = cv2.resize(frame, (1920, 1080))

        # Count frames
        frame_count += 1
        elapsed_time = time.time() - start_time

        # Calculate FPS and reset every second
        if elapsed_time >= 1.0: 
            print(f"Frames in one second: {frame_count}")
            cv2.putText(frame_resized, f"FPS: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frame_count = 0  # Reset frame count
            start_time = time.time()  # Reset the start time

        cv2.imshow(window_name, frame_resized)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error as e:
        print(f"OpenCV error: {e}")  # Handle OpenCV errors
    except Exception as e:
        print(f"An error occurred: {e}")  # Handle any other exceptions

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
