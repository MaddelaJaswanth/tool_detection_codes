import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(1)
plt.ion()
fig, ax = plt.subplots()

def detect_blue_objects(frame):
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define blue color range
    lower_blue = np.array([100, 150, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply Gaussian blur to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Morphological transformations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store object coordinates
    coordinates = []
    object_count = 1
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # Ignore small contours
        
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Append coordinates with the object number
        coordinates.append(f"Object {object_count} coordinates: ({center_x}, {center_y})")
        object_count += 1
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Obj {object_count-1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame, coordinates

# FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, coordinates = detect_blue_objects(frame)
    
    # Display coordinates for each detected object
    for coord in coordinates:
        print(coord)
    
    # Clear plot for updating
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    plt.pause(0.001)

cap.release()
plt.ioff()
plt.show()
