"""import cv2
import numpy as np
# Initialize the camera (0 for built-in camera, 1 for external camera)
cap = cv2.VideoCapture(1)  # Change to 0 if you're using the built-in camera

# Set camera resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
def adjust_brightness_contrast(frame):
    # Convert the image to grayscale to calculate brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    print("brightness is:",avg_brightness)
    # Determine brightness adjustment
    if avg_brightness > 100 and avg_brightness <180:  # Low brightness
        brightness = 20
        contrast = 40
    elif avg_brightness > 180:  # High brightness
        brightness = -50
        contrast = 50
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

# Function to detect blue objects in the frame
def detect_blue_object(frame):
    frame = adjust_brightness_contrast(frame)
    frame = cv2.bilateralFilter(frame, 9, 75, 75)  # Smooth the image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert the image to HSV color space
    #lower_blue = np.array([100, 150, 150])  # Adjust as necessary
    lower_blue = np.array([110, 150, 180])  # Lower brightness and saturation
    upper_blue = np.array([140, 255, 255])  # Adjust as necessary
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            object_count += 1

    if coordinates:
        mid_x = sum(coord[0] for coord in coordinates) // len(coordinates)
        mid_y = sum(coord[1] for coord in coordinates) // len(coordinates)
        cv2.circle(frame, (mid_x, mid_y), 5, (255, 0, 0), -1)
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

    # Detect blue objects in the current frame
    frame = detect_blue_object(frame)

    # Get the current window size
    #win_width = cv2.getWindowImageRect(window_name)[2]
    #win_height = cv2.getWindowImageRect(window_name)[3]

    # Resize the frame to match the window size
    frame_resized = cv2.resize(frame, (1920, 1080))

    # Calculate the midpoint of the resized window
    win_mid_x = 1920 // 2
    win_mid_y = 1080 // 2

    # Draw the midpoint of the resized window
    cv2.circle(frame_resized, (win_mid_x, win_mid_y), 5, (0, 255, 255), -1)
    cv2.putText(frame_resized, f"Window Midpoint: ({win_mid_x},{win_mid_y})", (win_mid_x + 10, win_mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the result with the camera feed
    cv2.imshow(window_name, frame_resized)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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

def adjust_brightness_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    print("Brightness is:", avg_brightness)
    
    # Dynamically adjust brightness and contrast
    if avg_brightness > 180:  # High brightness
        brightness = -50  # Reduce brightness more for very bright conditions
        contrast = 50
    elif avg_brightness < 100:  # Low brightness
        brightness = 20
        contrast = 40
    else:  # Normal brightness
        brightness = 0
        contrast = 0

    img = frame.astype(np.float32)
    img += brightness
    img = img * (contrast / 127 + 1) - contrast
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def apply_histogram_equalization(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])  # Equalize the Y (luminance) channel
    ycrcb = cv2.merge(channels)
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return frame

def detect_blue_object(frame):
    # Adjust brightness, contrast, and apply histogram equalization
    frame = adjust_brightness_contrast(frame)
    frame = apply_histogram_equalization(frame)
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate average brightness to adjust the HSV range dynamically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # Dynamically adjust the blue range based on brightness
    if avg_brightness > 180:  # High brightness
        lower_blue = np.array([110, 150, 180])
        upper_blue = np.array([130, 255, 220])
    elif avg_brightness < 100:  # Low brightness
        lower_blue = np.array([110, 120, 100])
        upper_blue = np.array([140, 255, 255])
    else:  # Normal brightness
        lower_blue = np.array([110, 150, 160])
        upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    object_count = 1

    # Define distance threshold to merge nearby contours
    distance_threshold = 50  # Adjust this value based on object size

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small areas
            continue
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Only consider objects that are far enough apart (no overlapping detection)
        if not any(np.linalg.norm(np.array([center_x, center_y]) - np.array(coord)) < distance_threshold for coord in coordinates):
            coordinates.append((center_x, center_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Obj {object_count} ({center_x},{center_y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            object_count += 1

    if coordinates:
        mid_x = sum(coord[0] for coord in coordinates) // len(coordinates)
        mid_y = sum(coord[1] for coord in coordinates) // len(coordinates)
        cv2.circle(frame, (mid_x, mid_y), 5, (255, 0, 0), -1)
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

    # Detect blue objects in the current frame
    frame = detect_blue_object(frame)

    # Resize the frame to 1920x1080
    frame_resized = cv2.resize(frame, (1920, 1080))

    # Calculate and draw the midpoint of the resized window
    win_mid_x = 1920 // 2
    win_mid_y = 1080 // 2
    cv2.circle(frame_resized, (win_mid_x, win_mid_y), 5, (0, 255, 255), -1)
    cv2.putText(frame_resized, f"Window Midpoint: ({win_mid_x},{win_mid_y})", (win_mid_x + 10, win_mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the result with the camera feed
    cv2.imshow(window_name, frame_resized)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
