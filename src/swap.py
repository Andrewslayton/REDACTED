import cv2
import dlib
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

# Initialize dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))

    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []
    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if all(rect[0] <= pt[0] < rect[0] + rect[2] and rect[1] <= pt[1] < rect[1] + rect[3] for pt in pts):
            indices = []
            for pt in pts:
                for idx, point in enumerate(points):
                    if np.allclose(pt, point, atol=1):
                        indices.append(idx)
                        break
            if len(indices) == 3:
                delaunay_triangles.append(indices)
    return delaunay_triangles

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def overlay_face(background_frame, face_frame):
    gray_background = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
    faces_background = detector(gray_background)
    faces_face = detector(gray_face)
    if len(faces_background) == 0 or len(faces_face) == 0:
        return background_frame

    shape_background = predictor(gray_background, faces_background[0])
    shape_face = predictor(gray_face, faces_face[0])

    points_background = np.array([[p.x, p.y] for p in shape_background.parts()], np.int32)
    points_face = np.array([[p.x, p.y] for p in shape_face.parts()], np.int32)

    rect_background = cv2.boundingRect(points_background)
    rect_face = cv2.boundingRect(points_face)

    print(f"rect_face: {rect_face}")
    print(f"rect_background: {rect_background}")
    center = (rect_background[0] + rect_background[2] // 2, rect_background[1] + rect_background[3] // 2)
    print(f"center: {center}")

    triangles = get_delaunay_triangles(rect_background, points_background)

    for tri_indices in triangles:
        t1 = [points_face[tri_indices[0]], points_face[tri_indices[1]], points_face[tri_indices[2]]]
        t2 = [points_background[tri_indices[0]], points_background[tri_indices[1]], points_background[tri_indices[2]]]

        rect1 = cv2.boundingRect(np.array(t1))
        rect2 = cv2.boundingRect(np.array(t2))

        t1_rect = []
        t2_rect = []
        t2_rect_int = []

        for i in range(3):
            t1_rect.append(((t1[i][0] - rect1[0]), (t1[i][1] - rect1[1])))
            t2_rect.append(((t2[i][0] - rect2[0]), (t2[i][1] - rect2[1])))
            t2_rect_int.append(((t2[i][0] - rect2[0]), (t2[i][1] - rect2[1])))

        mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

        img1_rect = face_frame[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]
        size = (rect2[2], rect2[3])
        img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

        background_frame[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = background_frame[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * (1 - mask) + img2_rect * mask

    mask = np.zeros_like(gray_background)
    cv2.fillConvexPoly(mask, cv2.convexHull(points_background), 255)
    r = cv2.boundingRect(cv2.convexHull(points_background))
    center = (r[0] + r[2] // 2, r[1] + r[3] // 2)
    seamless_clone = cv2.seamlessClone(background_frame, background_frame, mask, center, cv2.NORMAL_CLONE)
    return seamless_clone

def main():
    cap = cv2.VideoCapture(3)
    cap2 = cv2.VideoCapture(0)
    if not cap.isOpened() or not cap2.isOpened():
        raise RuntimeError('Could not open video source')

    pref_width = 1280
    pref_height = 720
    pref_fps_in = 30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
    cap.set(cv2.CAP_PROP_FPS, pref_fps_in)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20

    with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR, device="Unity Video Capture") as cam:
        while True:
            ret, face_frame = cap.read()
            ret2, background_frame = cap2.read()
            if not ret or not ret2:
                break

            face_frame = cv2.resize(face_frame, (width, height))
            background_frame = cv2.resize(background_frame, (width, height))

            frame = overlay_face(background_frame, face_frame)
            cam.send(frame)
            cam.sleep_until_next_frame()

if __name__ == "__main__":
    main()
