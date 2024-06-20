import os
import sys
import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser
from threading import Thread, Event
import pyvirtualcam
import matplotlib.pyplot as plt
from pyvirtualcam import PixelFormat

if getattr(sys, 'frozen', False):
    dat_file = os.path.join(sys._MEIPASS, 'shape_predictor_68_face_landmarks.dat')
else:
    dat_file = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dat_file)

stop_event = Event()
filter_var = None
current_color = None
distortion_strength = None
area_scale = None

def apply_black_bar(frame, color, scale_factor): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces) == 0:
        frame = cv2.GaussianBlur(frame, (99, 99), 30)
    for face in faces:
        shape = predictor(gray, face)
        x1=shape.part(36).x
        y1=shape.part(36).y
        x2=shape.part(45).x
        y2=shape.part(45).y

        # Calculate rotation angle
        left_eye = (shape.part(36).x, shape.part(36).y)
        right_eye = (shape.part(45).x, shape.part(45).y)
        angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        width = abs(x2 - x1) * scale_factor
        height = 50 * scale_factor
        size = (width, height)
        box = cv2.boxPoints(((center[0], center[1]), (size[0], size[1]), angle))
        box = np.intp(box)
        cv2.drawContours(frame, [box], 0, color, -1)
        
    return frame
def apply_pixel_distortion(frame, strength, scale_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    strength = max(1, strength)
    if strength % 2 == 0:
        strength += 1
    if len(faces) == 0:
        frame = cv2.GaussianBlur(frame, (99, 99), 30)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x1 = int(x - w * (scale_factor - 1) / 2)
        y1 = int(y - h * (scale_factor - 1) / 2)
        x2 = int(x + w * (1 + (scale_factor - 1) / 2))
        y2 = int(y + h * (1 + (scale_factor - 1) / 2))

        face_region = frame[y1:y2, x1:x2]
        distorted_region = cv2.GaussianBlur(face_region, (strength, strength), 30)
        frame[y1:y2, x1:x2] = distorted_region

    return frame

def apply_median_blur(frame, strength, scale_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    ksize = max(1, strength)
    if ksize % 2 == 0:
        ksize += 1
    if len(faces) == 0:
        frame = cv2.GaussianBlur(frame, (99, 99), 30)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x1 = int(x - w * (scale_factor - 1) / 2)
        y1 = int(y - h * (scale_factor - 1) / 2)
        x2 = int(x + w * (1 + (scale_factor - 1) / 2))
        y2 = int(y + h * (1 + (scale_factor - 1) / 2))

        face_region = frame[y1:y2, x1:x2]
        blurred_region = cv2.medianBlur(face_region, ksize)
        frame[y1:y2, x1:x2] = blurred_region

    return frame

def apply_box_filter(frame, strength, scale_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    ksize = (strength, strength)
    if len(faces) == 0:
        frame = cv2.GaussianBlur(frame, (99, 99), 30)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x1 = int(x - w * (scale_factor - 1) / 2)
        y1 = int(y - h * (scale_factor - 1) / 2)
        x2 = int(x + w * (1 + (scale_factor - 1) / 2))
        y2 = int(y + h * (1 + (scale_factor - 1) / 2))

        face_region = frame[y1:y2, x1:x2]
        filtered_region = cv2.boxFilter(face_region, -1, ksize)
        frame[y1:y2, x1:x2] = filtered_region

    return frame

def apply_laplacian(frame, scale_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        frame = cv2.GaussianBlur(frame, (99, 99), 30)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x1 = int(x - w * (scale_factor - 1) / 2)
        y1 = int(y - h * (scale_factor - 1) / 2)
        x2 = int(x + w * (1 + (scale_factor - 1) / 2))
        y2 = int(y + h * (1 + (scale_factor - 1) / 2))

        face_region = frame[y1:y2, x1:x2]
        laplacian_region = cv2.Laplacian(face_region, cv2.CV_64F)
        frame[y1:y2, x1:x2] = cv2.convertScaleAbs(laplacian_region)

    return frame

def apply_filter(frame):
    filter_choice = filter_var.get()
    bar_color = tuple(map(int, current_color))
    strength = distortion_strength.get()
    scale_factor = area_scale.get()
    if filter_choice == "eyeBar":
        frame = apply_black_bar(frame, bar_color, scale_factor)
    elif filter_choice == "distortion":
        frame = apply_pixel_distortion(frame, strength, scale_factor)
    elif filter_choice == "median":
        frame = apply_median_blur(frame, strength, scale_factor)
    elif filter_choice == "box":
        frame = apply_box_filter(frame, strength, scale_factor)
    elif filter_choice == "laplacian":
        frame = apply_laplacian(frame, scale_factor)

    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

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
            frame = apply_filter(frame)
            cam.send(frame)
            cam.sleep_until_next_frame()

def main():
    global filter_var, current_color, distortion_strength, area_scale
    root = tk.Tk()
    root.title("Select Filter")

    ttk.Label(root, text="Choose a filter to apply:").pack(pady=10)
    filter_var = tk.StringVar(value="eyeBar")
    current_color = (0, 0, 0)

    ttk.Radiobutton(root, text="Eye Level Bar", variable=filter_var, value="eyeBar").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="Pixel Distortion", variable=filter_var, value="distortion").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="Median Blur", variable=filter_var, value="median").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="Box Filter", variable=filter_var, value="box").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="Laplacian Edge Detection", variable=filter_var, value="laplacian").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="None", variable=filter_var, value="none").pack(anchor=tk.W)

    ttk.Label(root, text="Filter Scale: (Size)").pack(pady=10)
    area_scale = tk.Scale(root, from_=1, to=3, orient=tk.HORIZONTAL, resolution=0.1)
    area_scale.set(1)
    area_scale.pack(pady=10)

    ttk.Label(root, text="Distortion Strength:").pack(pady=10)
    distortion_strength = tk.Scale(root, from_=1, to=99, orient=tk.HORIZONTAL)
    distortion_strength.set(99)
    distortion_strength.pack(pady=10)

    ttk.Button(root, text="Change Bar Color", command=update_color).pack(pady=10)
    ttk.Button(root, text="Start Camera", command=start_camera_thread).pack(pady=10)
    ttk.Button(root, text="Stop Camera", command=stop_camera).pack(pady=10)

    root.mainloop()

def update_color():
    global current_color
    bar_color = colorchooser.askcolor(title="Choose bar color")[0]
    current_color = tuple(map(int, bar_color)) if bar_color else (0, 0, 0)

def start_camera_thread():
    global stop_event
    stop_event.clear()
    camera_thread = Thread(target=start_camera)
    camera_thread.start()

def stop_camera():
    stop_event.set()

if __name__ == "__main__":
    main()
