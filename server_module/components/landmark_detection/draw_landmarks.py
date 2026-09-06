import cv2

from .config import (
    DRAW_COLOR,
    DRAW_RADIUS
)


def draw_landmarks(frame, landmarks):

    for point in landmarks:

        cv2.circle(
            frame,
            point,
            DRAW_RADIUS,
            DRAW_COLOR,
            -1
        )

    return frame