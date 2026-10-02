import cv2
import os
import mediapipe as mp

from .config import (
    MAX_NUM_FACES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

# =========================
# MEDIAPIPE FACE LANDMARKER (new API for mediapipe >= 0.10.14)
# =========================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the face_landmarker.task model
# Go up: landmark_detection -> src -> Face-Project (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cand_landmark_new = os.path.join(BASE_DIR, "models", "landmarks", "mediapipe_face_landmarker_official.task")
cand_landmark_old = os.path.join(BASE_DIR, "models", "face_landmarker.task")
MODEL_PATH = cand_landmark_new if os.path.exists(cand_landmark_new) else cand_landmark_old


import math

def _get_default_oval_params_local(w: int, h: int):
    cx = w // 2
    cy = int(h * 0.505)
    ay = int(h * 0.38)
    ax = int(ay * 0.65)
    return (cx, cy), (ax, ay)

def _is_landmarks_in_oval_local(landmarks, w, h, center, axes, tolerance=1.0):
    if not landmarks or not center or not axes:
        return False
    cx, cy = center
    ax, ay = axes
    if ax <= 0 or ay <= 0:
        return False
    xs = [lm.x * w if hasattr(lm, "x") else lm[0] for lm in landmarks]
    ys = [lm.y * h if hasattr(lm, "y") else lm[1] for lm in landmarks]
    fcx = (min(xs) + max(xs)) / 2.0
    fcy = (min(ys) + max(ys)) / 2.0
    norm_x = (float(fcx) - cx) / float(ax * tolerance)
    norm_y = (float(fcy) - cy) / float(ay * tolerance)
    return (norm_x ** 2 + norm_y ** 2) <= 1.0

def _face_distance_to_oval_center_local(flm, w, h, center):
    xs = [lm.x * w if hasattr(lm, "x") else lm[0] for lm in flm]
    ys = [lm.y * h if hasattr(lm, "y") else lm[1] for lm in flm]
    fcx = (min(xs) + max(xs)) / 2.0
    fcy = (min(ys) + max(ys)) / 2.0
    return math.hypot(fcx - center[0], fcy - center[1])


class LandmarkDetector:

    def __init__(self):

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=MAX_NUM_FACES,
            min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame, oval_center=None, oval_axes=None, filter_oval=True, oval_tolerance=1.0):
        """
        Trích xuất landmarks khuôn mặt chính.
        Nếu filter_oval=True: chỉ nhận khuôn mặt nằm trong khung oval
        và lược bỏ hoàn toàn các khuôn mặt ở ngoài oval (trả về [] nếu không có mặt trong oval).
        """
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect(mp_image)

        landmarks = []
        if result.face_landmarks:
            h, w, _ = frame.shape
            all_faces = result.face_landmarks
            if filter_oval:
                if oval_center is None or oval_axes is None:
                    oval_center, oval_axes = _get_default_oval_params_local(w, h)
                faces_in_oval = [
                    flm for flm in all_faces
                    if _is_landmarks_in_oval_local(flm, w, h, oval_center, oval_axes, tolerance=oval_tolerance)
                ]
                if faces_in_oval:
                    chosen_face = min(
                        faces_in_oval,
                        key=lambda flm: _face_distance_to_oval_center_local(flm, w, h, oval_center)
                    )
                else:
                    return []
            else:
                chosen_face = all_faces[0]

            for lm in chosen_face:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))

        return landmarks

    def detect_with_count(self, frame, oval_center=None, oval_axes=None, filter_oval=True, oval_tolerance=1.0):
        """
        Trả về tuple: (landmarks_mặt_chính, số_lượng_khuôn_mặt).
        Nếu filter_oval=True:
        - Chỉ nhận khuôn mặt nằm trong khung oval và lược bỏ hoàn toàn các mặt ngoài oval.
        - Nếu có 1 mặt trong oval và các mặt khác ngoài oval -> num_faces = 1, landmarks = mặt trong oval.
        - Nếu có >= 2 mặt cùng trong oval -> num_faces = số mặt trong oval.
        - Nếu không có mặt nào trong oval -> num_faces = 0, landmarks = [].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return [], 0

        h, w = frame.shape[:2]
        all_faces = result.face_landmarks

        if filter_oval:
            if oval_center is None or oval_axes is None:
                oval_center, oval_axes = _get_default_oval_params_local(w, h)

            faces_in_oval = [
                flm for flm in all_faces
                if _is_landmarks_in_oval_local(flm, w, h, oval_center, oval_axes, tolerance=oval_tolerance)
            ]

            if faces_in_oval:
                # Central Face Anchor Prioritization
                if len(faces_in_oval) > 1 and oval_axes[0] > 0 and oval_axes[1] > 0:
                    dists = [_face_distance_to_oval_center_local(flm, w, h, oval_center) for flm in faces_in_oval]
                    min_dist = min(dists)
                    if min_dist <= 0.45 * oval_axes[0]:
                        anchored = [flm for flm, d in zip(faces_in_oval, dists) if d <= 0.75 * oval_axes[0]]
                        if anchored:
                            faces_in_oval = anchored

                num_faces = len(faces_in_oval)
                chosen_face = min(
                    faces_in_oval,
                    key=lambda flm: _face_distance_to_oval_center_local(flm, w, h, oval_center)
                )
            else:
                return [], 0
        else:
            num_faces = len(all_faces)
            chosen_face = all_faces[0]

        landmarks = []
        for lm in chosen_face:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        return landmarks, num_faces

    def detect_raw_3d(self, frame, oval_center=None, oval_axes=None, filter_oval=True, oval_tolerance=1.0):
        """
        Trả về tuple: (landmarks_3d_mặt_chính, số_lượng_khuôn_mặt).
        Nếu filter_oval=True: chỉ nhận mặt trong oval và lược bỏ hoàn toàn các mặt ngoài oval.
        Nếu không có mặt nào trong oval: trả về (None, 0).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None, 0

        h, w = frame.shape[:2]
        all_faces = result.face_landmarks

        if filter_oval:
            if oval_center is None or oval_axes is None:
                oval_center, oval_axes = _get_default_oval_params_local(w, h)

            faces_in_oval = [
                flm for flm in all_faces
                if _is_landmarks_in_oval_local(flm, w, h, oval_center, oval_axes, tolerance=oval_tolerance)
            ]

            if faces_in_oval:
                if len(faces_in_oval) > 1 and oval_axes[0] > 0 and oval_axes[1] > 0:
                    dists = [_face_distance_to_oval_center_local(flm, w, h, oval_center) for flm in faces_in_oval]
                    min_dist = min(dists)
                    if min_dist <= 0.45 * oval_axes[0]:
                        anchored = [flm for flm, d in zip(faces_in_oval, dists) if d <= 0.75 * oval_axes[0]]
                        if anchored:
                            faces_in_oval = anchored

                num_faces = len(faces_in_oval)
                chosen_face = min(
                    faces_in_oval,
                    key=lambda flm: _face_distance_to_oval_center_local(flm, w, h, oval_center)
                )
            else:
                return None, 0
        else:
            num_faces = len(all_faces)
            chosen_face = all_faces[0]

        return chosen_face, num_faces
