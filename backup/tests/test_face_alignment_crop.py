import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import cv2

from src.face_alignment_crop import FaceAligner

aligner = FaceAligner()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    landmarks = aligner.get_landmarks(frame)

    if landmarks:
        aligned = aligner.align_face(frame, landmarks)

        aligned_landmarks = aligner.get_landmarks(aligned)

        if aligned_landmarks:
            face_crop = aligner.crop_face(
                aligned,
                aligned_landmarks
            )

            cv2.imshow(
                "Aligned Face Crop",
                face_crop
            )

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()