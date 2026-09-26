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
        timeout: float = 7.0,
        min_consecutive_frames: int = 2,
        delta_yaw_threshold: float = 3.5,
        delta_pitch_threshold: float = 4.0
    ):
        """
        Khởi tạo bộ phát hiện cử động đầu.
        :param yaw_threshold: Ngưỡng góc quay trái/phải tuyệt đối (độ)
        :param pitch_threshold: Ngưỡng góc ngước lên/cúi xuống tuyệt đối (độ)
        :param roll_threshold: Ngưỡng góc nghiêng đầu (độ)
        :param timeout: Thời gian tối đa cho 1 thử thách (giây)
        :param min_consecutive_frames: Số frame liên tiếp duy trì góc để xác nhận vượt qua
        :param delta_yaw_threshold: Ngưỡng nhích nhẹ đầu tối thiểu (8 độ chuyển động thực tế từ mốc ban đầu)
        :param delta_pitch_threshold: Ngưỡng nhích nhẹ gật đầu tối thiểu
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.timeout = timeout
        self.min_consecutive_frames = max(1, min_consecutive_frames)
        self.delta_yaw_threshold = delta_yaw_threshold
        self.delta_pitch_threshold = delta_pitch_threshold

        self.current_action: HeadAction = HeadAction.NONE
        self.state: ChallengeState = ChallengeState.IDLE
        self.action_start_time: float = 0.0
        self.consecutive_frames: int = 0
        self.max_reached_angle: float = 0.0
        self.last_progress: float = 0.0
        self.last_angle: float = 0.0
        self.history_poses: List[Dict[str, float]] = []

        # Baseline pose tracking (khắc phục lỗi không di chuyển đầu vẫn pass)
        self.baseline_frames: List[Dict[str, float]] = []
        self.baseline_yaw: Optional[float] = None
        self.baseline_pitch: Optional[float] = None
        self.baseline_roll: Optional[float] = None

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
        Nếu action = None, hệ thống sẽ chọn ngẫu nhiên giữa (TURN_LEFT, TURN_RIGHT).
        """
        if action is None or action == HeadAction.NONE:
            available_actions = [
                HeadAction.TURN_LEFT,
                HeadAction.TURN_RIGHT
            ]
            self.current_action = random.choice(available_actions)
        else:
            self.current_action = action

        self.state = ChallengeState.IN_PROGRESS
        self.action_start_time = time.time()
        self.consecutive_frames = 0
        self.max_reached_angle = 0.0
        self.last_progress = 0.0
        self.last_angle = 0.0
        self.history_poses = []
        self.baseline_frames = []
        self.baseline_yaw = None
        self.baseline_pitch = None
        self.baseline_roll = None
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
                "progress": 1.0 if self.state == ChallengeState.COMPLETED else round(self.last_progress, 2),
                "current_angle": round(self.last_angle, 1)
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

        # 2. Định hình góc xuất phát ban đầu (Baseline calibration: 2 frame đầu)
        if len(self.baseline_frames) < 2:
            self.baseline_frames.append({"yaw": yaw, "pitch": pitch, "roll": roll})
            self.baseline_yaw = sum(p["yaw"] for p in self.baseline_frames) / len(self.baseline_frames)
            self.baseline_pitch = sum(p["pitch"] for p in self.baseline_frames) / len(self.baseline_frames)
            self.baseline_roll = sum(p["roll"] for p in self.baseline_frames) / len(self.baseline_frames)
            return {
                "state": self.state.value,
                "action": self.current_action.value,
                "passed": False,
                "prompt": self.get_prompt(),
                "time_left": round(float(time_left), 1),
                "progress": 0.0,
                "current_angle": 0.0,
                "target_threshold": float(self.delta_yaw_threshold),
                "is_matched": False
            }

        # 3. Tính toán độ chuyển động thực tế (Delta movement) so với vị trí ban đầu
        # Đảm bảo người dùng BẮT BUỘC PHẢI DI CHUYỂN ĐẦU đúng hướng ít nhất delta_yaw_threshold (~8 độ)
        is_matched = False
        delta_movement = 0.0
        target_threshold = self.delta_yaw_threshold

        base_y = self.baseline_yaw if self.baseline_yaw is not None else 0.0
        base_p = self.baseline_pitch if self.baseline_pitch is not None else 0.0

        if self.current_action == HeadAction.TURN_LEFT:
            # Quay sang trái người dùng: delta_yaw tăng (dương hơn so với baseline)
            delta_movement = yaw - base_y
            target_threshold = self.delta_yaw_threshold
            # Yêu cầu: Đã nhích sang trái ít nhất delta_yaw_threshold VÀ góc hiện tại lệch trái so với mốc
            is_matched = bool(delta_movement >= self.delta_yaw_threshold and (yaw > (base_y + 1.2) or yaw >= 3.5))

        elif self.current_action == HeadAction.TURN_RIGHT:
            # Quay sang phải người dùng: delta_yaw giảm (âm hơn so với baseline)
            delta_movement = base_y - yaw
            target_threshold = self.delta_yaw_threshold
            # Yêu cầu: Đã nhích sang phải ít nhất delta_yaw_threshold VÀ góc hiện tại lệch phải so với mốc
            is_matched = bool(delta_movement >= self.delta_yaw_threshold and (yaw < (base_y - 1.2) or yaw <= -3.5))

        elif self.current_action == HeadAction.LOOK_UP:
            delta_movement = base_p - pitch
            target_threshold = self.delta_pitch_threshold
            is_matched = bool(delta_movement >= self.delta_pitch_threshold and pitch < (base_p - 1.5))

        elif self.current_action == HeadAction.LOOK_DOWN:
            delta_movement = pitch - base_p
            target_threshold = self.delta_pitch_threshold
            is_matched = bool(delta_movement >= self.delta_pitch_threshold and pitch > (base_p + 1.5))

        elif self.current_action == HeadAction.LOOK_STRAIGHT:
            delta_movement = max(0.0, 10.0 - max(abs(yaw - base_y), abs(pitch - base_p)))
            target_threshold = 10.0
            is_matched = bool(abs(yaw - base_y) <= 4.0 and abs(pitch - base_p) <= 4.0)

        # 4. Tính toán tiến trình thời gian thực
        # Không di chuyển hoặc di chuyển ngược hướng -> progress = 0%
        # Nhích nhẹ dần theo đúng hướng -> progress tăng mượt
        # Đạt ngưỡng nhích rõ ràng hoặc giữ trong 2 frames -> pass 100%
        clamped_movement = max(0.0, delta_movement)
        angle_ratio = min(1.0, clamped_movement / max(1.0, target_threshold))

        if is_matched:
            self.consecutive_frames += 1
            self.max_reached_angle = max(self.max_reached_angle, clamped_movement)

            # Nếu quay góc dứt khoát (>= target_threshold * 1.25 ~ 4.4 độ): hoàn thành ngay sau 1 frame rõ!
            if clamped_movement >= (target_threshold * 1.25) or self.consecutive_frames >= self.min_consecutive_frames:
                self.state = ChallengeState.COMPLETED
                progress = 1.0
            else:
                progress = min(0.95, max(0.80, angle_ratio))
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)
            progress = min(0.75, angle_ratio * 0.75)

        self.last_progress = progress
        self.last_angle = clamped_movement

        return {
            "state": self.state.value,
            "action": self.current_action.value,
            "passed": bool(self.state == ChallengeState.COMPLETED),
            "prompt": self.get_prompt() if self.state != ChallengeState.COMPLETED else "HOAN THANH CU DONG!",
            "time_left": round(float(time_left), 1),
            "progress": round(float(progress), 2),
            "current_angle": round(float(clamped_movement), 1),
            "target_threshold": float(target_threshold),
            "is_matched": bool(is_matched),
            "baseline_yaw": round(float(base_y), 2)
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
        self.last_progress = 0.0
        self.last_angle = 0.0
        self.history_poses.clear()
        self.baseline_frames.clear()
        self.baseline_yaw = None
        self.baseline_pitch = None
        self.baseline_roll = None
