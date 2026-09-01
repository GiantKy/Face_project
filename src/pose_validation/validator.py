from .head_pose_3d import HeadPoseEstimator
from src.landmark_detection.utils import get_landmark_point


class PoseValidator:

    def __init__(self):
        self.estimator = HeadPoseEstimator()

    def validate(self, landmarks, get_point=get_landmark_point):
        if get_point is None:
            get_point = get_landmark_point

        pose = self.estimator.estimate(landmarks, get_point)

        if pose is None:
            return False, "No Face Pose", None

        yaw = pose["yaw"]
        pitch = pose["pitch"]
        roll = pose["roll"]

        # =========================
        # TURN LEFT / RIGHT
        # =========================
        if yaw > 25:
            return False, "Turn Right", pose

        if yaw < -25:
            return False, "Turn Left", pose

        # =========================
        # UP / DOWN
        # =========================
        if pitch > 20:
            return False, "Head Down", pose

        if pitch < -20:
            return False, "Head Up", pose

        # =========================
        # TILT
        # =========================
        if abs(roll) > 15:
            return False, "Head Tilt", pose

        return True, "Valid Pose", pose

    def validate_pose(self, frame_or_landmarks, landmarks=None, get_point=get_landmark_point):
        lm = landmarks if landmarks is not None else frame_or_landmarks
        valid, text, pose = self.validate(lm, get_point)
        return {
            "is_valid": valid,
            "text": text,
            "pose": pose if pose is not None else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        }