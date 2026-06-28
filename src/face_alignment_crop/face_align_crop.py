import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


class FaceAligner:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        h, w, _ = image.shape

        landmarks = []

        for lm in results.multi_face_landmarks[0].landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)

            landmarks.append((x, y))

        return landmarks

    def align_face(self, image, landmarks):
        left_eye = np.mean([
            landmarks[33],
            landmarks[133]
        ], axis=0).astype(int)

        right_eye = np.mean([
            landmarks[362],
            landmarks[263]
        ], axis=0).astype(int)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        angle = np.degrees(np.arctan2(dy, dx))

        h, w = image.shape[:2]

        matrix = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            angle,
            1.0
        )

        aligned = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC
        )

        return aligned

    def crop_face(
        self,
        image,
        landmarks,
        padding=20,
        output_size=(224, 224)
    ):
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]

        x1 = max(min(xs) - padding, 0)
        y1 = max(min(ys) - padding, 0)

        x2 = min(max(xs) + padding, image.shape[1])
        y2 = min(max(ys) + padding, image.shape[0])

        face = image[y1:y2, x1:x2]

        face = cv2.resize(face, output_size)

        return face