import cv2
import matplotlib.pyplot as plt
def find_built_in_camera():
    # Test each camera index to find the built-in camera
    for index in range(5):  # Check up to 5 camera indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Try to read a frame to ensure it's working
            ret, frame = cap.read()
            if ret:
                # Assuming the built-in camera might have a different resolution, you can check it
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Check for typical built-in camera resolution (example: 640x480)
                if width == 640 and height == 480:
                    cap.release()
                    return index  # Return the index of the built-in camera
            cap.release()
    return None

def find_external_camera():
    # Test each camera index to find an external camera
    for index in range(5):  # Check up to 5 camera indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Read a frame to ensure it's working
            ret, frame = cap.read()
            if ret:
                # You can add checks to differentiate external cameras
                # For simplicity, we'll assume that external cameras might have higher resolutions
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Assuming external camera has a higher resolution (example: 1920x1080)
                if width == 1920 and height == 1080:
                    cap.release()
                    return index  # Return the index of the external camera
            cap.release()
    return None

# Find the camera indices
built_in_camera_index = find_built_in_camera()
external_camera_index = find_external_camera()

print(f"Built-in camera index: {built_in_camera_index}")
print(f"External camera index: {external_camera_index}")

# Now open the cameras as needed
if built_in_camera_index is not None:
    cap_builtin = cv2.VideoCapture(built_in_camera_index)
    print(f"Using built-in camera at index {built_in_camera_index}")

if external_camera_index is not None:
    cap_external = cv2.VideoCapture(external_camera_index)
    print(f"Using external camera at index {external_camera_index}")

# Capture and display from both cameras (optional)
while True:
    if built_in_camera_index is not None:
        ret_builtin, frame_builtin = cap_builtin.read()
        if ret_builtin:
            plt.imshow(frame_builtin)

    if external_camera_index is not None:
        ret_external, frame_external = cap_external.read()
        if ret_external:
            plt.imshow(frame_external)

# Release and close everything
if built_in_camera_index is not None:
    cap_builtin.release()

if external_camera_index is not None:
    cap_external.release()
