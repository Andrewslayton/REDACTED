import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import tkinter as tk
from tkinter import colorchooser


bar_color = (0, 0, 0)  
apply_distortion = False

def apply_black_bar(frame, color):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y + h // 3), (x + w, y + 2 * h // 3), color, -1) #subject to change 
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
    global bar_color, apply_distortion
    cap = cv2.VideoCapture(0)
    with pyvirtualcam.Camera(width=640, height=480, fps=20, fmt=PixelFormat.BGR, device='OBS Virtual Camera') as cam:
        print(f'Using virtual camera: {cam.device}')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if apply_distortion:
                frame = apply_pixel_distortion(frame)
            else:
                frame = apply_black_bar(frame, bar_color)
            cam.send(frame)
            cam.sleep_until_next_frame()
    cap.release()

def choose_color():
    global bar_color
    color = colorchooser.askcolor()[0]
    if color:
        bar_color = (int(color[2]), int(color[1]), int(color[0]))  

def toggle_distortion():
    global apply_distortion
    apply_distortion = not apply_distortion

def create_gui():
    root = tk.Tk()
    root.title("BECOME UNSEEN")
    color_button = tk.Button(root, text="Choose Bar Color", command=choose_color)
    color_button.pack()
    distortion_button = tk.Button(root, text="Toggle Distortion", command=toggle_distortion)
    distortion_button.pack()
    start_button = tk.Button(root, text="Start Camera", command=start_camera)
    start_button.pack()

    root.mainloop()

if __name__ == '__main__':
    create_gui()
