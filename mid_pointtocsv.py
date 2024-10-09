import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# Open CSV file in append mode outside the loop
csv_file = open('midpoints.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Midpoint X', 'Midpoint Y'])  # Optional: write headers if needed

cap = cv2.VideoCapture(1)  # Default 0 for in-built camera, 1 for external camera
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()

def detect_blue_object(frame):
    # Apply bilateral filter to smooth the image
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    lower_blue = np.array([100, 150, 100])  # Adjust if needed
    upper_blue = np.array([130, 255, 255])  # Adjust if needed
    
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
        if cv2.contourArea(contour) < 200:  # Adjust threshold if needed
            continue        
        x, y, w, h = cv2.boundingRect(contour)               
        center_x = x + w // 2
        center_y = y + h // 2

        # Check for overlapping bounding boxes
        if not any((abs(center_x - coord[0]) < w and abs(center_y - coord[1]) < h) for coord in coordinates):
            coordinates.append((center_x, center_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            object_count += 1

    # Check if exactly two objects are detected
    if len(coordinates) == 2:
        print("Two objects detected!")
        
        # Calculate midpoint between the two detected objects
        midpoint_x = (coordinates[0][0] + coordinates[1][0]) // 2
        midpoint_y = (coordinates[0][1] + coordinates[1][1]) // 2
        
        # Print the midpoints to the console
        print(f"Midpoint: X = {midpoint_x}, Y = {midpoint_y}")
        cv2.circle(frame, (midpoint_x, midpoint_y), 5, (255,0, 0), -1)
        cv2.putText(frame, f" midpoint: ({midpoint_x},{midpoint_y})", (midpoint_x, midpoint_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # Write the midpoints of the two objects to the CSV file
        csv_writer.writerow([midpoint_x, midpoint_y])
        print(f"Midpoints written to CSV: X = {midpoint_x}, Y = {midpoint_y}")

    return frame
def screen_midpoint(frame):
    height, width, channel = frame.shape
    screen_mid_x = width // 2
    screen_mid_y = height // 2
    cv2.circle(frame, (screen_mid_x, screen_mid_y), 5, (0, 0, 255), -1)
    cv2.putText(frame, f"screen_mid_point:({screen_mid_x},{screen_mid_y})", (screen_mid_x + 10, screen_mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = detect_blue_object(frame)
    frame = screen_midpoint(frame)

    # Convert BGR to RGB for displaying in Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Clear the previous plot and display the new frame
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis('off')  # Hide axes
    plt.pause(0.001)  # Pause to update the plot

cap.release()
#cv2.destroyAllWindows()  # Add to close OpenCV windows
plt.ioff()
plt.show()

# Close the CSV file when the script ends
csv_file.close()
