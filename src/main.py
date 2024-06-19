import os
import sys
import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser
from threading import Thread, Event
import pyvirtualcam
from pyvirtualcam import PixelFormat

if getattr(sys, 'frozen', False):
    dat_file = os.path.join(sys._MEIPASS, 'shape_predictor_68_face_landmarks.dat')
else:
    dat_file = 'shape_predictor_68_face_landmarks.dat'

# Initialize the dlib detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dat_file)

stop_event = Event()
filter_var = None
current_color = None
distortion_strength = None
area_scale = None

def apply_black_bar(frame, color, scale_factor):
    # Ensure the frame is an 8-bit image
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Convert to grayscale if not already
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # Color image in BGR format
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(frame.shape) == 2:  # Already grayscale
        gray = frame
    else:
        raise ValueError("Unsupported frame format")

    # Ensure the image is a C-contiguous array
    if not gray.flags['C_CONTIGUOUS']:
        gray = np.ascontiguousarray(gray)

    try:
        faces = detector(gray)
    except RuntimeError as e:
        print(f"Error during face detection: {e}")
        return frame

    for face in faces:
        shape = predictor(gray, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x1 = int(x - w * (scale_factor - 1) / 2)
        y1 = int(y - h * (scale_factor - 1) / 2)
        x2 = int(x + w * (1 + (scale_factor - 1) / 2))
        y2 = int(y + h * (1 + (scale_factor - 1) / 2))

        cv2.rectangle(frame, (x1, y1 + (y2 - y1) // 3), (x2, y1 + 2 * (y2 - y1) // 3), color, -1)

    return frame

def start_camera():
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        raise RuntimeError('Could not open video source')
    pref_width = 1280
    pref_height = 720
    pref_fps_in = 30
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
    vc.set(cv2.CAP_PROP_FPS, pref_fps_in)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = 20

    with pyvirtualcam.Camera(width, height, fps_out, fmt=PixelFormat.BGR, device="Unity Video Capture") as cam:
        while not stop_event.is_set():
            ret, frame = vc.read()
            if not ret:
                raise RuntimeError('Error fetching frame')

            # Debugging frame type and shape before processing
            print(f"Original frame type: {frame.dtype}, shape: {frame.shape}")
            
            filter_choice = filter_var.get()
            bar_color = tuple(map(int, current_color))
            strength = distortion_strength.get()
            scale_factor = area_scale.get()

            if filter_choice == "eyeBar":
                frame = apply_black_bar(frame, bar_color, scale_factor)
            # Add other filter functions here if necessary

            cam.send(frame)
            cam.sleep_until_next_frame()

def start_camera_thread():
    global stop_event
    stop_event.clear()
    camera_thread = Thread(target=start_camera)
    camera_thread.start()

def stop_camera():
    stop_event.set()

def update_color():
    global current_color
    bar_color = colorchooser.askcolor(title="Choose bar color")[0]
    current_color = tuple(map(int, bar_color)) if bar_color else (0, 0, 0)

root = tk.Tk()
root.title("Select Filter")

ttk.Label(root, text="Choose a filter to apply:").pack(pady=10)
filter_var = tk.StringVar(value="eyeBar")
current_color = (0, 0, 0)

ttk.Radiobutton(root, text="Eye Level Bar", variable=filter_var, value="eyeBar").pack(anchor=tk.W)
# Add other filter options here

ttk.Label(root, text="Filter Scale: (Size)").pack(pady=10)
area_scale = tk.Scale(root, from_=1, to=3, orient=tk.HORIZONTAL, resolution=0.1)
area_scale.set(1)
area_scale.pack(pady=10)

ttk.Label(root, text="Distortion Strength:").pack(pady=10)
distortion_strength = tk.Scale(root, from_=1, to=99, orient=tk.HORIZONTAL,)
distortion_strength.set(99)
distortion_strength.pack(pady=10)

ttk.Button(root, text="Change Bar Color", command=update_color).pack(pady=10)
ttk.Button(root, text="Start Camera", command=start_camera_thread).pack(pady=10)
ttk.Button(root, text="Stop Camera", command=stop_camera).pack(pady=10)

stop_event = Event()
root.mainloop()
