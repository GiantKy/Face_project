from .head_pose_3d import HeadPoseEstimator


class PoseValidator:

    def __init__(self):
        self.estimator = HeadPoseEstimator()

    def validate(self, landmarks, get_point):

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