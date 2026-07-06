import os
from ultralytics import YOLO

# Tự động tìm thư mục gốc Face-Project
# __file__       = src/face_detection/detector.py
# dirname 1 lần  = src/face_detection/
# dirname 2 lần  = src/
# dirname 3 lần  = Face-Project/
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


class FaceDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                BASE_DIR, "models", "Face_Detection.pt"
            )
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)

        faces = []

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])

                cls = int(box.cls[0])

                face_crop = frame[y1:y2, x1:x2]

                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls,
                    "face_crop": face_crop
                })

        return faces