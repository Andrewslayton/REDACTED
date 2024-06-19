import cv2

# List available video capture devices
def list_devices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

# Print available devices
devices = list_devices()
print("Available video capture devices:", devices)

# Test each device to find the OBS Virtual Camera
for index in devices:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open video device {index}.")
    else:
        print(f"Successfully opened video device {index}.")
        ret, frame = cap.read()
        if ret:
            window_name = f'Device {index}'
            cv2.imshow(window_name, frame)
            print(f"Displaying feed from device {index}. Press any key to close window.")
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyWindow(window_name)
        cap.release()
