from .head_pose_3d import HeadPoseEstimator
from ..landmark_detection.utils import get_landmark_point


class PoseValidator:

    def __init__(self, max_yaw: float = 32.0, max_pitch: float = 24.0, max_roll: float = 18.0):
        self.estimator = HeadPoseEstimator()
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll

    def validate(self, landmarks, get_point=get_landmark_point, img_w=None, img_h=None):
        if get_point is None:
            get_point = get_landmark_point

        pose = self.estimator.estimate(landmarks, get_point, img_w=img_w, img_h=img_h)

        if pose is None:
            return False, "No Face Pose", None

        yaw = pose["yaw"]
        pitch = pose["pitch"]
        roll = pose["roll"]

        # =========================
        # TURN LEFT / RIGHT
        # =========================
        if yaw > self.max_yaw:
            return False, "Turn Right", pose

        if yaw < -self.max_yaw:
            return False, "Turn Left", pose

        # =========================
        # UP / DOWN
        # =========================
        if pitch > self.max_pitch:
            return False, "Head Down", pose

        if pitch < -self.max_pitch:
            return False, "Head Up", pose

        # =========================
        # TILT
        # =========================
        if abs(roll) > self.max_roll:
            return False, "Head Tilt", pose

        return True, "Valid Pose", pose

    def validate_pose(self, frame_or_landmarks, landmarks=None, get_point=get_landmark_point, img_w=None, img_h=None):
        lm = landmarks if landmarks is not None else frame_or_landmarks
        valid, text, pose = self.validate(lm, get_point, img_w=img_w, img_h=img_h)
        return {
            "is_valid": valid,
            "text": text,
            "pose": pose if pose is not None else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        }