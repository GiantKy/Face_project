"""
=============================================================================
Test Head Movement Challenge (Real-Time Webcam)
Kiểm tra tính năng phát hiện cử động đầu (Active Liveness Challenge):
  - Quay trái (Turn Left)
  - Quay phải (Turn Right)
  - Ngước lên (Look Up)
  - Cúi xuống (Look Down)

Phím tắt:
  - 'r' hoặc 'c': Đổi thử thách ngẫu nhiên mới
  - '1': Thử thách Quay Trái
  - '2': Thử thách Quay Phải
  - '3': Thử thách Ngước Lên
  - '4': Thử thách Cúi Xuống
  - 'q' hoặc ESC: Thoát
=============================================================================
"""

import sys
import os
import time
import cv2
import numpy as np

# Thêm project root vào sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)  # Face-Project/
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.landmark_detection import LandmarkDetector
from src.landmark_detection.draw_landmarks import draw_landmarks
from src.landmark_detection.utils import get_landmark_point
from src.pose_validation import PoseValidator
from src.head_movement import HeadMovementDetector, HeadAction, ChallengeState


def draw_ui_overlay(frame, hm_status, pose_dict, challenge_detector):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Khung HUD thông tin chính
    card_w = min(500, w - 40)
    card_h = 160
    cv2.rectangle(overlay, (20, 20), (20 + card_w, 20 + card_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (20, 20), (20 + card_w, 20 + card_h), (100, 100, 100), 1)

    cv2.putText(frame, "HEAD MOVEMENT LIVENESS TEST", (35, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
    cv2.line(frame, (35, 60), (20 + card_w - 20, 60), (80, 80, 80), 1)

    # Hiển thị trạng thái góc Pose
    if pose_dict:
        yaw = pose_dict.get("yaw", 0.0)
        pitch = pose_dict.get("pitch", 0.0)
        roll = pose_dict.get("roll", 0.0)
        pose_str = f"Pose: Y:{yaw:+5.1f} | P:{pitch:+5.1f} | R:{roll:+5.1f}"
        cv2.putText(frame, pose_str, (35, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Pose: Khong phat hien mat", (35, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1)

    # Hiển thị Thử thách & Hướng dẫn
    prompt_text = hm_status.get("prompt", "")
    time_left = hm_status.get("time_left", 0.0)
    passed = hm_status.get("passed", False)
    state = hm_status.get("state", "IDLE")

    if state == "COMPLETED" or passed:
        p_color = (0, 255, 0)
        status_msg = f"[PASS] {prompt_text}"
    elif state == "FAILED":
        p_color = (0, 0, 255)
        status_msg = f"[FAIL] {prompt_text}"
    else:
        p_color = (0, 255, 255)
        status_msg = f">> {prompt_text} ({time_left:.1f}s)"

    cv2.putText(frame, status_msg, (35, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)

    # Thanh tiến trình Progress Bar
    progress = hm_status.get("progress", 0.0)
    bar_x = 35
    bar_y = 145
    bar_w = card_w - 50
    bar_h = 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        bar_color = (0, 255, 0) if (passed or state == "COMPLETED") else (0, 200, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)

    # Hướng dẫn phím bấm ở góc dưới
    help_txt = "[r/c]: Random Challenge | [1]: Left | [2]: Right | [3]: Up | [4]: Down | [q]: Thoat"
    cv2.putText(frame, help_txt, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


def main():
    print("\n" + "=" * 65)
    print("      TEST HEAD MOVEMENT ACTIVE LIVENESS (WEBCAM)")
    print("=" * 65)

    detector = LandmarkDetector()
    pose_validator = PoseValidator()
    head_movement = HeadMovementDetector(yaw_threshold=16.0, pitch_threshold=12.0, timeout=6.0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Khong the mo Webcam!")
        return

    # Khởi động thử thách đầu tiên
    current_action = head_movement.start_challenge()
    print(f"[INFO] Khoi tao thu thach dau tien: {current_action.value}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 1. Phát hiện landmarks
        landmarks = detector.detect(frame)
        pose_dict = None

        if landmarks:
            frame = draw_landmarks(frame, landmarks)
            _, _, pose_dict = pose_validator.validate(landmarks, get_landmark_point)

        # 2. Cập nhật Head Movement Detector
        hm_status = head_movement.update(pose_dict)

        # 3. Vẽ giao diện HUD
        frame = draw_ui_overlay(frame, hm_status, pose_dict, head_movement)

        cv2.imshow("Head Movement Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r') or key == ord('c'):
            new_act = head_movement.start_challenge()
            print(f"\n[CHALLENGE MOI] {new_act.value} -> {head_movement.get_prompt()}")
        elif key == ord('1'):
            head_movement.start_challenge(HeadAction.TURN_LEFT)
            print(f"\n[CHALLENGE MOI] TURN_LEFT -> {head_movement.get_prompt()}")
        elif key == ord('2'):
            head_movement.start_challenge(HeadAction.TURN_RIGHT)
            print(f"\n[CHALLENGE MOI] TURN_RIGHT -> {head_movement.get_prompt()}")
        elif key == ord('3'):
            head_movement.start_challenge(HeadAction.LOOK_UP)
            print(f"\n[CHALLENGE MOI] LOOK_UP -> {head_movement.get_prompt()}")
        elif key == ord('4'):
            head_movement.start_challenge(HeadAction.LOOK_DOWN)
            print(f"\n[CHALLENGE MOI] LOOK_DOWN -> {head_movement.get_prompt()}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Da dung chuong trinh test Head Movement.")


if __name__ == "__main__":
    main()
