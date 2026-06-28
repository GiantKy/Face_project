import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import cv2

from src.landmark_detection import LandmarkDetector
from src.landmark_detection.draw_landmarks import draw_landmarks
from src.landmark_detection.utils import get_landmark_point

from src.pose_validation import PoseValidator
from src.pose_validation.draw_pose import draw_pose_info


cap = cv2.VideoCapture(0)

detector = LandmarkDetector()
pose_validator = PoseValidator()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    landmarks = detector.detect(frame)

    if landmarks:

        frame = draw_landmarks(frame, landmarks)

        valid, text, pose = pose_validator.validate(
            landmarks,
            get_landmark_point
        )

        frame = draw_pose_info(
            frame,
            pose,
            text,
            valid
        )

    cv2.imshow("Pose Validation FINAL", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()