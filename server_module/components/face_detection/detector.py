import os
from typing import List, Dict, Any, Optional
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


def calculate_iou_and_iomin(box_a, box_b):
    """Tính IoU chuẩn và IoMin (Intersection over Min Area) giữa 2 bounding box."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area
    iou = (inter_area / union_area) if union_area > 0 else 0.0

    min_area = min(area_a, area_b)
    iomin = (inter_area / min_area) if min_area > 0 else 0.0

    return iou, iomin


def deduplicate_faces(
    faces: List[Dict[str, Any]],
    iou_thresh: float = 0.35,
    iomin_thresh: float = 0.60
) -> List[Dict[str, Any]]:
    """
    Loại bỏ các bounding box trùng lặp trên cùng 1 khuôn mặt:
    - Nếu 2 box có IoU >= iou_thresh (hai box đè lên nhau)
    - Hoặc IoMin >= iomin_thresh (một box nằm gọn bên trong box kia)
    -> Giữ lại box có độ tin cậy (confidence) cao hơn.
    """
    if len(faces) <= 1:
        return faces

    # Sắp xếp ưu tiên độ tin cậy cao nhất
    sorted_faces = sorted(faces, key=lambda f: f["confidence"], reverse=True)
    kept_faces: List[Dict[str, Any]] = []

    for f in sorted_faces:
        is_duplicate = False
        box_f = f["bbox"]
        for kf in kept_faces:
            box_k = kf["bbox"]
            iou, iomin = calculate_iou_and_iomin(box_f, box_k)
            if iou >= iou_thresh or iomin >= iomin_thresh:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_faces.append(f)

    return kept_faces


class FaceDetector:
    def __init__(self, model_path=None, conf_thresh=0.45, iou_thresh=0.40):
        if model_path is None:
            model_path = os.path.join(
                BASE_DIR, "models", "Face_Detection.pt"
            )
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, frame, conf: Optional[float] = None, iou: Optional[float] = None, min_size: int = 25):
        run_conf = conf if conf is not None else self.conf_thresh
        run_iou = iou if iou is not None else self.iou_thresh

        results = self.model(frame, conf=run_conf, iou=run_iou, verbose=False)

        faces = []
        h, w = frame.shape[:2]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                c_conf = float(box.conf[0])
                cls = int(box.cls[0])

                bw = x2 - x1
                bh = y2 - y1

                # Lọc các box quá nhỏ (nhiễu nền/ảnh trên tường/đồ vật)
                if bw < min_size or bh < min_size:
                    continue

                # Kẹp tọa độ trong khung ảnh
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                face_crop = frame[y1:y2, x1:x2]

                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": c_conf,
                    "class_id": cls,
                    "face_crop": face_crop
                })

        # Nếu chưa tìm thấy và ảnh có độ phân giải thấp (<= 320px như ESP32 240x240)
        # Tự động upscale 2x để phát hiện khuôn mặt nhạy hơn
        if not faces and (w <= 320 or h <= 320):
            import cv2
            scale_w = 2.0
            scale_h = 2.0
            up_frame = cv2.resize(frame, (int(w * scale_w), int(h * scale_h)), interpolation=cv2.INTER_LINEAR)
            up_results = self.model(up_frame, conf=max(0.18, run_conf * 0.75), iou=run_iou, verbose=False)
            for result in up_results:
                for box in result.boxes:
                    ux1, uy1, ux2, uy2 = map(int, box.xyxy[0])
                    c_conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    x1 = max(0, min(w - 1, int(ux1 / scale_w)))
                    y1 = max(0, min(h - 1, int(uy1 / scale_h)))
                    x2 = max(0, min(w, int(ux2 / scale_w)))
                    y2 = max(0, min(h, int(uy2 / scale_h)))
                    bw = x2 - x1
                    bh = y2 - y1

                    if bw < min_size or bh < min_size:
                        continue

                    faces.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": c_conf,
                        "class_id": cls,
                        "face_crop": frame[y1:y2, x1:x2]
                    })

        # NMS Deduplication loại bỏ box đè hoặc box con cùng 1 người
        return deduplicate_faces(faces, iou_thresh=0.35, iomin_thresh=0.60)