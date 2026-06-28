import cv2
import mediapipe as mp

from .config import (
    MAX_NUM_FACES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)


class LandmarkDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        landmarks = []

        if results.multi_face_landmarks:

            h, w, _ = frame.shape

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            for lm in face_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                landmarks.append((x, y))

        return landmarks