import cv2
import numpy as np
import time

def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera at index {camera_index} cannot be opened.")
    return cap

def adjust_brightness_contrast(frame):
    # Convert the image to grayscale to calculate brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    # Determine brightness adjustment
    if avg_brightness < 100:  # Low brightness
        brightness = 40
        contrast = 20
    elif avg_brightness > 180:  # High brightness
        brightness = -30
        contrast = 30
    else:  # Normal brightness
        brightness = 0
        contrast = 0
    #print(f"Average Brightness: {avg_brightness:.2f}, Adjusting -> Brightness: {brightness}, Contrast: {contrast}")
    # Convert to float to prevent clipping
    img = frame.astype(np.float32)

    # Adjust brightness
    img += brightness

    # Adjust contrast
    img = img * (contrast / 127 + 1) - contrast

    # Clip to the range [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def detect_blue_object(frame):
    # Adjust brightness and contrast
    frame = adjust_brightness_contrast(frame)

    # Apply bilateral filter to smooth the image
    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for blue color
    lower_blue = np.array([100, 150, 150])  # Adjust these values based on conditions
    upper_blue = np.array([130, 255, 255])  # Adjust these values based on conditions

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
        print(f"Object 1: ({0}, {0})")
    return frame

def screen_midpoint(frame):
    height, width, channel = frame.shape
    screen_mid_x = width // 2
    screen_mid_y = height // 2
    cv2.circle(frame, (screen_mid_x, screen_mid_y), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"screen_mid_point:({screen_mid_x},{screen_mid_y})", (screen_mid_x + 10, screen_mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

cap = initialize_camera(1)  # Try to open the external camera at index 1
stop_flag = False  # Global flag to indicate when to stop the loop

# Initialize frame counting
frame_count = 0
start_time = time.time()

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

        # Count frames
        frame_count += 1
        elapsed_time = time.time() - start_time

        # Calculate FPS and reset every second
        if elapsed_time >= 1.0: 
            #print(f"Frames in one second: {frame_count}")
            cv2.putText(frame, f"FPS: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frame_count = 0  # Reset frame count
            start_time = time.time()  # Reset the start time

        cv2.imshow('Detected Blue Objects', frame)

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