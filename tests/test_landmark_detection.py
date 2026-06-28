import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import cv2

from src.landmark_detection import (
    LandmarkDetector
)

from src.landmark_detection.draw_landmarks import (
    draw_landmarks
)

from src.landmark_detection.constants import (
    LEFT_EYE,
    RIGHT_EYE,
    NOSE,
    MOUTH
)

from src.landmark_detection.utils import (
    get_landmark_point
)

cap = cv2.VideoCapture(0)

detector = LandmarkDetector()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    landmarks = detector.detect(frame)

    if landmarks:

        frame = draw_landmarks(
            frame,
            landmarks
        )

        # LEFT EYE
        for idx in LEFT_EYE:

            point = get_landmark_point(
                landmarks,
                idx
            )

            if point:
                cv2.circle(
                    frame,
                    point,
                    4,
                    (255, 0, 0),
                    -1
                )

        # RIGHT EYE
        for idx in RIGHT_EYE:

            point = get_landmark_point(
                landmarks,
                idx
            )

            if point:
                cv2.circle(
                    frame,
                    point,
                    4,
                    (0, 0, 255),
                    -1
                )

        # NOSE
        for idx in NOSE:

            point = get_landmark_point(
                landmarks,
                idx
            )

            if point:
                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 255, 255),
                    -1
                )

        # MOUTH
        for idx in MOUTH:

            point = get_landmark_point(
                landmarks,
                idx
            )

            if point:
                cv2.circle(
                    frame,
                    point,
                    5,
                    (255, 255, 0),
                    -1
                )

    cv2.imshow(
        "Landmark Detection",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()