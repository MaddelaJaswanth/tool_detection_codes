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
