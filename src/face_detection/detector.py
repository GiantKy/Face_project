from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path="models/face_detection.pt"):
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