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
# Go up: landmark_detection -> src -> backup -> Face-Project (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")


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

    def detect(self, frame):

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

            for lm in result.face_landmarks[0]:

                x = int(lm.x * w)
                y = int(lm.y * h)

                landmarks.append((x, y))

        return landmarks
