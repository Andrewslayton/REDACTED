import argparse
import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser

def apply_black_bar(frame, color):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y + h // 3), (x + w, y + 2 * h // 3), color, -1)

    return frame

def apply_pixel_distortion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        distorted_region = cv2.GaussianBlur(face_region, (99, 99), 30)
        frame[y:y+h, x:x+w] = distorted_region

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
    fps_in = vc.get(cv2.CAP_PROP_FPS)
    fps_out = 20

    with pyvirtualcam.Camera(width, height, fps_out, fmt=PixelFormat.BGR) as cam:
        while True:
            ret, frame = vc.read()
            if not ret:
                raise RuntimeError('Error fetching frame')

            filter_choice = filter_var.get()
            bar_color = tuple(map(int, current_color))

            if filter_choice == "blackbar":
                frame = apply_black_bar(frame, bar_color)
            elif filter_choice == "distortion":
                frame = apply_pixel_distortion(frame)
            cam.send(frame)
            cam.sleep_until_next_frame()

def update_color():
    global current_color
    bar_color = colorchooser.askcolor(title="Choose bar color")[0]
    current_color = tuple(map(int, bar_color)) if bar_color else (0, 0, 0)

root = tk.Tk()
root.title("Select Filter")

ttk.Label(root, text="Choose a filter to apply:").pack(pady=10)
filter_var = tk.StringVar(value="blackbar")
current_color = (0, 0, 0)

ttk.Radiobutton(root, text="Black Bar", variable=filter_var, value="blackbar").pack(anchor=tk.W)
ttk.Radiobutton(root, text="Pixel Distortion", variable=filter_var, value="distortion").pack(anchor=tk.W)
ttk.Radiobutton(root, text="None", variable=filter_var, value="none").pack(anchor=tk.W)
ttk.Button(root, text="Change Bar Color", command=update_color).pack(pady=10)
ttk.Button(root, text="Start Camera", command=start_camera).pack(pady=20)

root.mainloop()
