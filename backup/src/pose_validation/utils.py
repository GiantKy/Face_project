import math

from src.landmark_detection.utils import get_landmark_point
from src.landmark_detection.constants import LEFT_EYE, RIGHT_EYE, NOSE


class PoseSmoother:
    def __init__(self, size=5):
        self.buffer_x = []
        self.buffer_y = []
        self.size = size

    def smooth(self, x, y):

        self.buffer_x.append(x)
        self.buffer_y.append(y)

        if len(self.buffer_x) > self.size:
            self.buffer_x.pop(0)
            self.buffer_y.pop(0)

        avg_x = sum(self.buffer_x) / len(self.buffer_x)
        avg_y = sum(self.buffer_y) / len(self.buffer_y)

        return avg_x, avg_y


smoother = PoseSmoother()


def get_face_pose(landmarks):

    left_eye = []
    right_eye = []
    nose = []

    for idx in LEFT_EYE:
        p = get_landmark_point(landmarks, idx)
        if p:
            left_eye.append(p)

    for idx in RIGHT_EYE:
        p = get_landmark_point(landmarks, idx)
        if p:
            right_eye.append(p)

    for idx in NOSE:
        p = get_landmark_point(landmarks, idx)
        if p:
            nose.append(p)

    if not left_eye or not right_eye or not nose:
        return None

    left_eye = left_eye[0]
    right_eye = right_eye[0]
    nose = nose[0]

    # =========================
    # FACE CENTER (ổn định nhất)
    # =========================
    cx = (left_eye[0] + right_eye[0]) / 2
    cy = (left_eye[1] + right_eye[1]) / 2

    # =========================
    # VECTOR NOSE → CENTER
    # =========================
    dx = nose[0] - cx
    dy = nose[1] - cy

    # =========================
    # NORMALIZE
    # =========================
    face_width = abs(right_eye[0] - left_eye[0])

    if face_width == 0:
        return None

    norm_x = dx / face_width
    norm_y = dy / face_width

    # =========================
    # FIX CAMERA BIAS (QUAN TRỌNG)
    # =========================
    norm_x, norm_y = smoother.smooth(norm_x, norm_y)

    return {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "nose": nose,
        "norm_x": norm_x,
        "norm_y": norm_y
    }