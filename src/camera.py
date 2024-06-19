import logging
import cv2
from pyvirtualcam import PixelFormat, Camera
from src.consts import APP_NAME

DEFAULT_CAMERA_NAME = "Unity Video Capture"
CAMERA_NAME = f"{APP_NAME} Video Capture".replace(" ", "_")

camera_names = [CAMERA_NAME, DEFAULT_CAMERA_NAME]


class VirtualCameraMirror:
    def __init__(
        self,
        height: int,
        width: int,
        fps: int,
        fmt=PixelFormat.BGR,
        input_device: int = 0,
    ):
        self.height = height
        self.width = width
        self.fps = fps
        self.fmt = fmt

        self.vcap = cv2.VideoCapture(input_device)
        if not self.vcap.isOpened():
            raise RuntimeError("Could not open video source")

        self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.vcap.set(cv2.CAP_PROP_FPS, self.fps)

        self.vcam: Camera | None = None

    def __enter__(self):
        width = int(self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.vcap.get(cv2.CAP_PROP_FPS))

        camera_exception = None

        for name in camera_names:
            try:
                logging.info(f"Trying to open camera with name {name}")
                self.vcam = Camera(width, height, fps, fmt=self.fmt, device=name)
                break
            except RuntimeError as e:
                camera_exception = e
                logging.warning(
                    f"Could not open camera with name {name}, looking for another camera..."
                )
        if self.vcam is None:
            raise RuntimeError(
                "Could not open any virtual camera"
            ) from camera_exception
        return (self.vcap, self.vcam)

    def __exit__(self, exc_type, exc_value, traceback):
        self.vcap.release()
        if self.vcam is None:
            return
        self.vcam.close()
