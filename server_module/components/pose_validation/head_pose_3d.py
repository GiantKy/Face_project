import cv2
import numpy as np


class HeadPoseEstimator:

    def __init__(self, w=640, h=480):

        self.w = w
        self.h = h

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        focal = w
        center = (w / 2, h / 2)

        self.camera_matrix = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

    def estimate(self, landmarks, get_point):

        try:
            nose = get_point(landmarks, 1)
            chin = get_point(landmarks, 152)
            le = get_point(landmarks, 33)
            re = get_point(landmarks, 263)
            lm = get_point(landmarks, 61)
            rm = get_point(landmarks, 291)

            if not all([nose, chin, le, re, lm, rm]):
                return None

            image_points = np.array([
                nose, chin, le, re, lm, rm
            ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )

            if not success:
                return None

            rmat, _ = cv2.Rodrigues(rvec)

            sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)

            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(rmat[2, 1], rmat[2, 2])
                y = np.arctan2(-rmat[2, 0], sy)
                z = np.arctan2(rmat[1, 0], rmat[0, 0])
            else:
                x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                y = np.arctan2(-rmat[2, 0], sy)
                z = 0

            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)

            # 🔥 FIX NGƯỢC TRÁI PHẢI
            yaw = -yaw

            # 🔥 FIX PITCH WRAP (-180 ↔ 180)
            if pitch > 90:
                pitch -= 180
            elif pitch < -90:
                pitch += 180

            return {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            }

        except:
            return None