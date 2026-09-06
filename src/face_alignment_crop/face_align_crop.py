import cv2
import os
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")


class FaceAligner:
    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def get_landmarks(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        h, w, _ = image.shape
        landmarks = []
        for lm in result.face_landmarks[0]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))
        return landmarks

    def align_face(self, image, landmarks=None, bbox=None, **kwargs):
        """
        Can chinh xoay 2 mat nam ngang quanh chinh tam 2 mat (Eyes Center).
        Tranh xoay quanh tam anh (w//2, h//2) lam lech vi tri khuon mat.
        """
        if landmarks is None:
            landmarks = self.get_landmarks(image)

        if landmarks is None or len(landmarks) < 363:
            return image.copy()

        left_eye = np.mean([landmarks[33], landmarks[133]], axis=0).astype(int)
        right_eye = np.mean([landmarks[362], landmarks[263]], axis=0).astype(int)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        h, w = image.shape[:2]

        # Xoay quanh chinh trung diem 2 mat
        eyes_cx = int((left_eye[0] + right_eye[0]) / 2.0)
        eyes_cy = int((left_eye[1] + right_eye[1]) / 2.0)

        matrix = cv2.getRotationMatrix2D((eyes_cx, eyes_cy), angle, 1.0)
        aligned = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC)
        return aligned

    def crop_face(
        self,
        image,
        landmarks=None,
        bbox=None,
        padding=20,
        output_size=(224, 224),
        mode="bbox",
        **kwargs
    ):
        """
        Cat khuon mat tu anh:
        - Mac dinh uu tien mode='bbox': Cat thang dung 100% theo Bounding Box cua YOLO tren anh goc,
          khong xoay cheo, giu nguyen tu the tu nhien cua nguoi dung va khong bi vien den.
        - Fallback landmarks neu khong co bbox.
        """
        h, w = image.shape[:2]

        if bbox is not None and mode == "bbox":
            bx1, by1, bx2, by2 = map(int, bbox)
            x1 = max(bx1 - padding, 0)
            y1 = max(by1 - padding, 0)
            x2 = min(bx2 + padding, w)
            y2 = min(by2 + padding, h)
        elif landmarks is not None and len(landmarks) > 0:
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            x1 = max(min(xs) - padding, 0)
            y1 = max(min(ys) - padding, 0)
            x2 = min(max(xs) + padding, w)
            y2 = min(max(ys) + padding, h)
        elif bbox is not None:
            bx1, by1, bx2, by2 = map(int, bbox)
            x1 = max(bx1 - padding, 0)
            y1 = max(by1 - padding, 0)
            x2 = min(bx2 + padding, w)
            y2 = min(by2 + padding, h)
        else:
            landmarks = self.get_landmarks(image)
            if landmarks is not None and len(landmarks) > 0:
                xs = [p[0] for p in landmarks]
                ys = [p[1] for p in landmarks]
                x1 = max(min(xs) - padding, 0)
                y1 = max(min(ys) - padding, 0)
                x2 = min(max(xs) + padding, w)
                y2 = min(max(ys) + padding, h)
            else:
                return cv2.resize(image, output_size)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return cv2.resize(image, output_size)

        return cv2.resize(face, output_size)
