import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the camera
cap = cv2.VideoCapture(1)

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# Function to choose the color range based on user input
def get_color_range(color_choice):
    if color_choice == 'blue':
        lower = np.array([0, 0, 100])
        upper = np.array([70, 70, 255])
    elif color_choice == 'green':
        lower = np.array([0, 100, 0])
        upper = np.array([70, 255, 70])
    elif color_choice == 'red':
        lower = np.array([150, 0, 0])
        upper = np.array([255, 70, 70])
    else:
        raise ValueError("Invalid color choice! Choose between 'blue', 'green', or 'red'.")
    return lower, upper

# Get user input for color selection
color_choice = input("Choose the color to detect (blue/green/red): ").strip().lower()

# Get the respective color range
lower_color, upper_color = get_color_range(color_choice)

def detect_color_objects(frame, lower_color, upper_color):
    """
    Detect objects of the selected color in the frame.
    Returns the frame with bounding boxes and the coordinates of the detected objects.
    """
    # Convert to RGB (because OpenCV by default uses BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect selected color objects
    mask = cv2.inRange(rgb_frame, lower_color, upper_color)
    
    # Remove noise from the mask using morphological transformations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask (these are the detected objects)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coordinates = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # Skip small contours that are likely noise
        
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw a green bounding box around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        coordinates.append((center_x, center_y))
        
        # Print the coordinates of the detected object
        print(f"Object detected at: ({center_x}, {center_y})")
    
    return frame, coordinates

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect the selected color objects in the frame
    frame, coordinates = detect_color_objects(frame, lower_color, upper_color)
    
    # Clear previous plot and show the updated frame
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)
    
    # If any key is pressed, break the loop
    if plt.waitforbuttonpress():
        break

# Release the camera and close the plot
cap.release()
plt.ioff()
plt.show()
