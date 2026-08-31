import time
import math
import random
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


class HeadAction(Enum):
    NONE = "none"
    LOOK_STRAIGHT = "look_straight"  # Nhìn thẳng / Giữ vị trí trung tâm
    TURN_LEFT = "turn_left"          # Quay mặt sang trái (yaw < -threshold)
    TURN_RIGHT = "turn_right"        # Quay mặt sang phải (yaw > +threshold)
    LOOK_UP = "look_up"              # Ngước mặt lên trên (pitch < -threshold)
    LOOK_DOWN = "look_down"          # Cúi mặt xuống dưới (pitch > +threshold)
    TILT_LEFT = "tilt_left"          # Nghiêng đầu sang trái (roll < -threshold)
    TILT_RIGHT = "tilt_right"        # Nghiêng đầu sang phải (roll > +threshold)


class ChallengeState(Enum):
    IDLE = "IDLE"                    # Chưa bắt đầu
    WAITING_CENTER = "WAITING_CENTER"# Đang chờ người dùng nhìn thẳng
    IN_PROGRESS = "IN_PROGRESS"      # Đang thực hiện hành động
    COMPLETED = "COMPLETED"          # Đã hoàn thành thử thách
    FAILED = "FAILED"                # Thất bại (hết thời gian timeout)


class HeadMovementDetector:
    """
    Module phát hiện và xác thực cử động đầu (Active Head Movement Liveness Challenge).
    Hỗ trợ cả chế độ phân tích góc đơn lẻ (Single-frame) và theo dõi chuỗi chuyển động thời gian thực (Real-time Video/Webcam).
    """

    ACTION_PROMPTS = {
        HeadAction.NONE: "Giu khuon mat trong khung hinh",
        HeadAction.LOOK_STRAIGHT: "Vui long NHIN THANG vao camera",
        HeadAction.TURN_LEFT: "Vui long QUAY MAT SANG TRAI",
        HeadAction.TURN_RIGHT: "Vui long QUAY MAT SANG PHAI",
        HeadAction.LOOK_UP: "Vui long NGUOC MAT LEN TREN",
        HeadAction.LOOK_DOWN: "Vui long CUI MAT XUONG DUOI",
        HeadAction.TILT_LEFT: "Vui long NGHIENG DAU SANG TRAI",
        HeadAction.TILT_RIGHT: "Vui long NGHIENG DAU SANG PHAI",
    }

    def __init__(
        self,
        yaw_threshold: float = 16.0,
        pitch_threshold: float = 12.0,
        roll_threshold: float = 14.0,
        timeout: float = 6.0,
        min_consecutive_frames: int = 4
    ):
        """
        Khởi tạo bộ phát hiện cử động đầu.
        :param yaw_threshold: Ngưỡng góc quay trái/phải (độ)
        :param pitch_threshold: Ngưỡng góc ngước lên/cúi xuống (độ)
        :param roll_threshold: Ngưỡng góc nghiêng đầu (độ)
        :param timeout: Thời gian tối đa cho 1 thử thách (giây)
        :param min_consecutive_frames: Số frame liên tiếp duy trì góc để xác nhận vượt qua (chống nhiễu)
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.timeout = timeout
        self.min_consecutive_frames = min_consecutive_frames

        self.current_action: HeadAction = HeadAction.NONE
        self.state: ChallengeState = ChallengeState.IDLE
        self.action_start_time: float = 0.0
        self.consecutive_frames: int = 0
        self.max_reached_angle: float = 0.0
        self.history_poses: List[Dict[str, float]] = []

    def classify_static_pose(self, pose_dict: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Phân loại hướng đầu cho 1 frame / ảnh tĩnh.
        Trả về hướng hiện tại của đầu: Center, Turn Left, Turn Right, Look Up, Look Down, Tilt.
        """
        if pose_dict is None:
            return {
                "detected_action": HeadAction.NONE.value,
                "description": "No Pose Data",
                "is_straight": False,
                "dominant_direction": "UNKNOWN"
            }

        yaw = pose_dict.get("yaw", 0.0)
        pitch = pose_dict.get("pitch", 0.0)
        roll = pose_dict.get("roll", 0.0)

        is_straight = bool(
            abs(yaw) <= self.yaw_threshold and
            abs(pitch) <= self.pitch_threshold and
            abs(roll) <= self.roll_threshold
        )

        detected_action = HeadAction.LOOK_STRAIGHT
        dominant_direction = "STRAIGHT"

        if yaw < -self.yaw_threshold:
            detected_action = HeadAction.TURN_LEFT
            dominant_direction = "LEFT"
        elif yaw > self.yaw_threshold:
            detected_action = HeadAction.TURN_RIGHT
            dominant_direction = "RIGHT"
        elif pitch < -self.pitch_threshold:
            detected_action = HeadAction.LOOK_UP
            dominant_direction = "UP"
        elif pitch > self.pitch_threshold:
            detected_action = HeadAction.LOOK_DOWN
            dominant_direction = "DOWN"
        elif roll < -self.roll_threshold:
            detected_action = HeadAction.TILT_LEFT
            dominant_direction = "TILT_LEFT"
        elif roll > self.roll_threshold:
            detected_action = HeadAction.TILT_RIGHT
            dominant_direction = "TILT_RIGHT"

        return {
            "detected_action": detected_action.value,
            "description": self.ACTION_PROMPTS.get(detected_action, ""),
            "is_straight": bool(is_straight),
            "dominant_direction": dominant_direction,
            "angles": {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
        }

    def start_challenge(self, action: Optional[HeadAction] = None) -> HeadAction:
        """
        Bắt đầu một thử thách cử động đầu mới.
        Nếu action = None, hệ thống sẽ chọn ngẫu nhiên giữa (TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN).
        """
        if action is None or action == HeadAction.NONE:
            available_actions = [
                HeadAction.TURN_LEFT,
                HeadAction.TURN_RIGHT,
                HeadAction.LOOK_UP,
                HeadAction.LOOK_DOWN
            ]
            self.current_action = random.choice(available_actions)
        else:
            self.current_action = action

        self.state = ChallengeState.IN_PROGRESS
        self.action_start_time = time.time()
        self.consecutive_frames = 0
        self.max_reached_angle = 0.0
        self.history_poses = []
        return self.current_action

    def get_prompt(self) -> str:
        """Lấy câu lệnh hướng dẫn tương ứng với thử thách hiện tại"""
        return self.ACTION_PROMPTS.get(self.current_action, "")

    def update(self, pose_dict: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Cập nhật trạng thái thử thách theo thời gian thực (Real-time loop).
        :param pose_dict: Dict chứa {'yaw', 'pitch', 'roll'} từ HeadPoseEstimator
        :return: Dict chứa toàn bộ thông tin trạng thái cử động
        """
        if self.state != ChallengeState.IN_PROGRESS or pose_dict is None:
            time_left = 0.0
            if self.state == ChallengeState.IN_PROGRESS:
                time_left = max(0.0, self.timeout - (time.time() - self.action_start_time))
            return {
                "state": self.state.value,
                "action": self.current_action.value,
                "passed": self.state == ChallengeState.COMPLETED,
                "prompt": self.get_prompt() if self.state == ChallengeState.IN_PROGRESS else self._get_status_text(),
                "time_left": round(time_left, 1),
                "progress": 1.0 if self.state == ChallengeState.COMPLETED else 0.0,
                "current_angle": 0.0
            }

        yaw = pose_dict.get("yaw", 0.0)
        pitch = pose_dict.get("pitch", 0.0)
        roll = pose_dict.get("roll", 0.0)
        self.history_poses.append({"yaw": yaw, "pitch": pitch, "roll": roll})

        elapsed = time.time() - self.action_start_time
        time_left = max(0.0, self.timeout - elapsed)

        # 1. Kiểm tra hết thời gian (Timeout)
        if elapsed > self.timeout:
            self.state = ChallengeState.FAILED
            return {
                "state": self.state.value,
                "action": self.current_action.value,
                "passed": False,
                "prompt": "HET THOI GIAN THUC HIEN!",
                "time_left": 0.0,
                "progress": 0.0,
                "current_angle": 0.0
            }

        # 2. Kiểm tra điều kiện góc tương ứng với hành động yêu cầu
        is_matched = False
        current_angle = 0.0
        target_threshold = 1.0

        if self.current_action == HeadAction.TURN_LEFT:
            current_angle = -yaw
            target_threshold = self.yaw_threshold
            is_matched = (yaw < -self.yaw_threshold)

        elif self.current_action == HeadAction.TURN_RIGHT:
            current_angle = yaw
            target_threshold = self.yaw_threshold
            is_matched = (yaw > self.yaw_threshold)

        elif self.current_action == HeadAction.LOOK_UP:
            current_angle = -pitch
            target_threshold = self.pitch_threshold
            is_matched = (pitch < -self.pitch_threshold)

        elif self.current_action == HeadAction.LOOK_DOWN:
            current_angle = pitch
            target_threshold = self.pitch_threshold
            is_matched = (pitch > self.pitch_threshold)

        elif self.current_action == HeadAction.TILT_LEFT:
            current_angle = -roll
            target_threshold = self.roll_threshold
            is_matched = (roll < -self.roll_threshold)

        elif self.current_action == HeadAction.TILT_RIGHT:
            current_angle = roll
            target_threshold = self.roll_threshold
            is_matched = (roll > self.roll_threshold)

        elif self.current_action == HeadAction.LOOK_STRAIGHT:
            current_angle = max(abs(yaw), abs(pitch))
            target_threshold = self.yaw_threshold
            is_matched = (abs(yaw) <= 10.0 and abs(pitch) <= 10.0)

        # Cập nhật số frame liên tiếp đạt chuẩn
        if is_matched:
            self.consecutive_frames += 1
            self.max_reached_angle = max(self.max_reached_angle, current_angle)
            if self.consecutive_frames >= self.min_consecutive_frames:
                self.state = ChallengeState.COMPLETED
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)

        # Tính toán mức độ hoàn thành (% progress)
        progress = min(1.0, max(0.0, self.consecutive_frames / float(self.min_consecutive_frames)))
        if self.state == ChallengeState.COMPLETED:
            progress = 1.0

        return {
            "state": self.state.value,
            "action": self.current_action.value,
            "passed": self.state == ChallengeState.COMPLETED,
            "prompt": self.get_prompt() if self.state != ChallengeState.COMPLETED else "HOAN THANH CU DONG!",
            "time_left": round(time_left, 1),
            "progress": round(progress, 2),
            "current_angle": round(current_angle, 1),
            "target_threshold": target_threshold,
            "is_matched": is_matched
        }

    def _get_status_text(self) -> str:
        if self.state == ChallengeState.COMPLETED:
            return "HOAN THANH CU DONG THANH CONG!"
        elif self.state == ChallengeState.FAILED:
            return "THU THACH THAT BAI (TIMEOUT)!"
        elif self.state == ChallengeState.WAITING_CENTER:
            return "VUI LONG NHIN VAO CHINH GIUA"
        return "IDLE"

    def reset(self):
        """Reset toàn bộ trạng thái về ban đầu"""
        self.state = ChallengeState.IDLE
        self.current_action = HeadAction.NONE
        self.consecutive_frames = 0
        self.max_reached_angle = 0.0
        self.history_poses.clear()
