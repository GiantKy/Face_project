import math


def calculate_distance(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def get_landmark_point(landmarks, index):

    if index >= len(landmarks):
        return None

    return landmarks[index]