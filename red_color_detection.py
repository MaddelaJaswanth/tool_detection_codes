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
def detect_red_object(frame):
    # Apply bilateral filter to smooth the image
    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red color detection
    lower_red1 = np.array([0, 120, 150])    # Lower range of red
    upper_red1 = np.array([10, 255, 255])   # Upper range of red
    lower_red2 = np.array([170, 120, 150])  # Second lower range of red
    upper_red2 = np.array([180, 255, 255])  # Second upper range of red

    # Create masks for red color and combine them
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Process the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_count = 1  # Reset object count for each frame
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Filter out small contours
            continue
        
        # Get the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the center of the detected object
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw a circle at the center of the detected object
        radius = int(0.5 * (w + h) / 2)  # Radius based on the bounding box
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)

        # Label the object with its coordinates
        cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        object_count += 1  # Increment object count

    # Debug: Show the mask for verification
    cv2.imshow('Red Mask', mask)  # Display the mask to verify the detection
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
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
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

        frame = detect_red_object(frame)
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
