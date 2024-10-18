"""import cv2
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
    lower_blue = np.array([90, 100, 120])  # Lower bound for blue
    upper_blue = np.array([130, 255, 255])  # Upper bound for blue
    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    detection_radius=80
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
        is_detected_object=True
        for coord in coordinates:
            if abs(center_x - coord[0]) < detection_radius and abs(center_y - coord[1]) < detection_radius:
                is_detected_object = False
                break
        if is_detected_object:
            coordinates.append((center_x, center_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Detected{object_count} ({center_x},{center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            object_count+=1
            
        # Check for overlapping bounding boxes
        #if not any((abs(center_x - coord[0]) < w and abs(center_y - coord[1]) < h) for coord in coordinates):
            #coordinates.append((center_x, center_y))
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #object_count += 1
        #if coordinates:
            #print(f"Detected Object: {coordinates[0]}")  # Print the first detected object's coordinates
        if len(coordinates)==2:
            mid_x = sum(coord[0] for coord in coordinates) // 2
            mid_y = sum(coord[1] for coord in coordinates) // 2
            print(f"Object 1: ({coordinates[0][0]}, {coordinates[0][1]})")
            print(f"Object 2: ({coordinates[1][0]}, {coordinates[1][1]})")
            print(f"Midpoint: ({mid_x}, {mid_y})")

    # Print coordinates of detected objects one by one
    #for idx, coord in enumerate(coordinates, start=1):
        #print(f"Object {idx}: ({coord[0]}, {coord[1]})")

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
window_name='Full_camera_display'
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
        #frame = screen_midpoint(frame)
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
"""



import cv2
import numpy as np
import time

def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera at index {camera_index} cannot be opened.")
    return cap

cap = initialize_camera(1)  # Try to open the external camera at index 1

# Function to detect blue objects in the frame and merge nearby contours
def detect_blue_object(frame):
    # Apply bilateral filter to smooth the image
    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a wider HSV range to capture both regular and light blue
    lower_blue = np.array([100, 150, 120])  # Lower bound for blue
    upper_blue = np.array([130, 255, 255])  # Upper bound for blue

    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Morphological transformations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Merge contours by creating one bounding box for nearby contours
    if contours:
        # Create an empty array to store all points from all contours
        all_contours = np.vstack([contour for contour in contours])
        
        # Create a single bounding rectangle around all merged contours
        x, y, w, h = cv2.boundingRect(all_contours)
        
        # Draw a single bounding box around the merged contours
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected ({x+w//2},{y+h//2})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"Detected Object: Center at ({x+w//2}, {y+h//2})")

    return frame

# Function to draw the screen midpoint
def screen_midpoint(frame):
    height, width, _ = frame.shape
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






















































"""


import cv2
import numpy as np

# Initialize the camera (0 for built-in camera, 1 for external camera)
cap = cv2.VideoCapture(1)  # Change to 0 if you're using the built-in camera

# Set camera resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def nothing(x):
    pass

# Create a window for adjustments
cv2.namedWindow('Adjustments')

# Create trackbars for lower and upper HSV values
cv2.createTrackbar('Lower Hue', 'Adjustments', 100, 180, nothing)
cv2.createTrackbar('Upper Hue', 'Adjustments', 140, 180, nothing)
cv2.createTrackbar('Lower Saturation', 'Adjustments', 150, 255, nothing)
cv2.createTrackbar('Upper Saturation', 'Adjustments', 255, 255, nothing)
cv2.createTrackbar('Lower Value', 'Adjustments', 50, 255, nothing)
cv2.createTrackbar('Upper Value', 'Adjustments', 255, 255, nothing)

# Function to detect blue objects in the frame
def detect_blue_object(frame, lower_blue, upper_blue):
    frame = cv2.bilateralFilter(frame, 9, 75, 75)  # Smooth the image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # Create a mask for blue color
    mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Apply Gaussian blur
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Morphological opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Morphological closing

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    object_count = 1

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Adjust area threshold as needed
            continue
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        if not any((abs(center_x - coord[0]) < w and abs(center_y - coord[1]) < h) for coord in coordinates):
            coordinates.append((center_x, center_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around detected object
            cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Label the object
            object_count += 1

    if coordinates:
        mid_x = sum(coord[0] for coord in coordinates) // len(coordinates)
        mid_y = sum(coord[1] for coord in coordinates) // len(coordinates)
        cv2.circle(frame, (mid_x, mid_y), 5, (255, 0, 0), -1)  # Draw midpoint
        cv2.putText(frame, f"Midpoint: ({mid_x},{mid_y})", (mid_x + 10, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

# Create a window for displaying the feed
window_name = 'Camera Feed'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow window resizing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get values from trackbars
    lower_hue = cv2.getTrackbarPos('Lower Hue', 'Adjustments')
    upper_hue = cv2.getTrackbarPos('Upper Hue', 'Adjustments')
    lower_saturation = cv2.getTrackbarPos('Lower Saturation', 'Adjustments')
    upper_saturation = cv2.getTrackbarPos('Upper Saturation', 'Adjustments')
    lower_value = cv2.getTrackbarPos('Lower Value', 'Adjustments')
    upper_value = cv2.getTrackbarPos('Upper Value', 'Adjustments')

    lower_blue = np.array([lower_hue, lower_saturation, lower_value])
    upper_blue = np.array([upper_hue, upper_saturation, upper_value])

    # Detect blue objects in the current frame
    frame = detect_blue_object(frame, lower_blue, upper_blue)

    # Display the result with the camera feed
    cv2.imshow(window_name, frame)

    # Show mask for debugging
    #cv2.imshow('Mask', cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_blue, upper_blue))

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

"""
