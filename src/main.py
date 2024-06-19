
import tkinter as tk
from tkinter import colorchooser, ttk
from threading import Thread, Event

import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat

from src.lib_install import main as lib_install
stop_event = Event()
filter_var = None
current_color = None
distortion_strength = None
area_scale = None
def apply_black_bar(frame, color, scale_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        frame = cv2.GaussianBlur(frame, (99, 99), 30)

    for (x, y, w, h) in faces:
        x1 = int(x - w * (scale_factor - 1) / 2)
        y1 = int(y - h * (scale_factor - 1) / 2)
        x2 = int(x + w * (1 + (scale_factor - 1) / 2))
        y2 = int(y + h * (1 + (scale_factor - 1) / 2))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        cv2.rectangle(frame, (x1, y1 + (y2 - y1) // 3), (x2, y1 + 2 * (y2 - y1) // 3), color, -1)

    return frame

def apply_pixel_distortion(frame, strength, scale_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    strength = max(1, strength)
    if strength % 2 == 0:
        strength += 1

    if len(faces) == 0:
        frame = cv2.GaussianBlur(frame, (99, 99), 30)
    else:
        for (x, y, w, h) in faces:
            x1 = int(x - w * (scale_factor - 1) / 2)
            y1 = int(y - h * (scale_factor - 1) / 2)
            x2 = int(x + w * (1 + (scale_factor - 1) / 2))
            y2 = int(y + h * (1 + (scale_factor - 1) / 2))

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_region = frame[y1:y2, x1:x2]
            distorted_region = cv2.GaussianBlur(face_region, (strength, strength), 30)
            frame[y1:y2, x1:x2] = distorted_region

    return frame


def apply_median_blur(frame, strength, scale_factor):
    ksize = max(1, strength)
    if ksize % 2 == 0:
        ksize += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        frame = cv2.medianBlur(frame, ksize)
    else:
        for (x, y, w, h) in faces:
            x1 = int(x - w * (scale_factor - 1) / 2)
            y1 = int(y - h * (scale_factor - 1) / 2)
            x2 = int(x + w * (1 + (scale_factor - 1) / 2))
            y2 = int(y + h * (1 + (scale_factor - 1) / 2))

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_region = frame[y1:y2, x1:x2]
            blurred_region = cv2.medianBlur(face_region, ksize)
            frame[y1:y2, x1:x2] = blurred_region

    return frame


def apply_box_filter(frame, strength, scale_factor):
    ksize = (strength, strength)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        frame = cv2.boxFilter(frame, -1, ksize)
    else:
        for (x, y, w, h) in faces:
            x1 = int(x - w * (scale_factor - 1) / 2)
            y1 = int(y - h * (scale_factor - 1) / 2)
            x2 = int(x + w * (1 + (scale_factor - 1) / 2))
            y2 = int(y + h * (1 + (scale_factor - 1) / 2))

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_region = frame[y1:y2, x1:x2]
            filtered_region = cv2.boxFilter(face_region, -1, ksize)
            frame[y1:y2, x1:x2] = filtered_region

    return frame


def apply_laplacian(frame, scale_factor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        frame = cv2.convertScaleAbs(laplacian)
    else:
        for (x, y, w, h) in faces:
            x1 = int(x - w * (scale_factor - 1) / 2)
            y1 = int(y - h * (scale_factor - 1) / 2)
            x2 = int(x + w * (1 + (scale_factor - 1) / 2))
            y2 = int(y + h * (1 + (scale_factor - 1) / 2))

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_region = frame[y1:y2, x1:x2]
            laplacian_region = cv2.Laplacian(face_region, cv2.CV_64F)
            frame[y1:y2, x1:x2] = cv2.convertScaleAbs(laplacian_region)

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

def main():
    global stop_event
    root = tk.Tk()
    root.title("Select Filter")

    ttk.Label(root, text="Choose a filter to apply:").pack(pady=10)
    global filter_var
    filter_var = tk.StringVar(value="eyeBar")
    global current_color
    current_color = (0, 0, 0)

    ttk.Radiobutton(root, text="Eye Level Bar", variable=filter_var, value="eyeBar").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="Pixel Distortion", variable=filter_var, value="distortion").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="Median Blur", variable=filter_var, value="median").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="Box Filter", variable=filter_var, value="box").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="Laplacian Edge Detection", variable=filter_var, value="laplacian").pack(anchor=tk.W)
    ttk.Radiobutton(root, text="None", variable=filter_var, value="none").pack(anchor=tk.W)

    ttk.Label(root, text="Filter Scale: (Size)").pack(pady=10)
    global area_scale
    area_scale = tk.Scale(root, from_=1, to=3, orient=tk.HORIZONTAL, resolution=0.1)
    area_scale.set(1)
    area_scale.pack(pady=10)

    ttk.Label(root, text="Distortion Strength:").pack(pady=10)
    global distortion_strength
    distortion_strength = tk.Scale(root, from_=1, to=99, orient=tk.HORIZONTAL,)
    distortion_strength.set(99)
    distortion_strength.pack(pady=10)

    ttk.Button(root, text="Change Bar Color", command=update_color).pack(pady=10)
    ttk.Button(root, text="Start Camera", command=start_camera_thread).pack(pady=10)
    ttk.Button(root, text="Stop Camera", command=stop_camera).pack(pady=10)

    stop_event = Event()
    root.mainloop()


if __name__ == "__main__":
    lib_install()
    main()
