import cv2

def crop_face(frame, bbox, padding=20):
    x1, y1, x2, y2 = bbox

    h, w = frame.shape[:2]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return frame[y1:y2, x1:x2]