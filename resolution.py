import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)  # Default 0 for in-built camera, 1 for external camera
plt.ion()  # Interactive mode on  
fig, ax = plt.subplots()

def detect_blue_object(frame):
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define blue color range
    lower_blue = np.array([100, 150, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Create a mask to detect blue objects
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coordinates = []  # To store the coordinates of detected objects
    object_count = 1  # Initialize object count
    
    # Loop over detected contours
    for contour in contours:
        if cv2.contourArea(contour) < 200:  # Skip small objects
            continue
        
        # Get bounding box for the detected object
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Add detected object coordinates to the list
        coordinates.append((center_x, center_y))
        
        # Draw rectangle around the object and label it with coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        object_count += 1

    
    # If at least two objects are detected, calculate the midpoint
    if len(coordinates) >= 2:
        mid_x = (coordinates[0][0] + coordinates[1][0]) // 2
        mid_y = (coordinates[0][1] + coordinates[1][1]) // 2
        #print("midpoint of x:",mid_x)
        #print("midpoint of y:",mid_y)
        # Draw a circle at the midpoint and display the coordinates
        cv2.circle(frame, (mid_x, mid_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Midpoint: ({mid_x}, {mid_y})", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame to detect blue objects and calculate midpoints
    frame = detect_blue_object(frame)
    
    # Clear the axis and display the processed frame
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    plt.pause(0.001)

# Release the video capture and turn off interactive plotting
cap.release()
plt.ioff()
plt.show()








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
    frame = cv2.bilateralFilter(frame, 15, 75, 75)  # Increased parameters
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 100])  # Lower values for brightness
    upper_blue = np.array([140, 255, 255])  # Keep upper values high
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Elliptical kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    object_count = 1  # Reset object count for each frame

    for contour in contours:
        if cv2.contourArea(contour) < 1200:  # Adjust threshold if needed
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detected_area = frame[y:y + h, x:x + w]
        avg_color = cv2.mean(detected_area)[:3]
        if avg_color[0] < avg_color[1] and avg_color[0] < avg_color[2]:
            continue
        center_x = x + w // 2
        center_y = y + h // 2

        if not any((abs(center_x - coord[0]) < w and abs(center_y - coord[1]) < h) for coord in coordinates):
            coordinates.append((center_x, center_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            object_count += 1

    if coordinates:
        mid_x = sum(coord[0] for coord in coordinates) // len(coordinates)
        mid_y = sum(coord[1] for coord in coordinates) // len(coordinates)
        cv2.circle(frame, (mid_x, mid_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Midpoint: ({mid_x},{mid_y})", (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

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
            print(f"Frames in one second: {frame_count}")
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
