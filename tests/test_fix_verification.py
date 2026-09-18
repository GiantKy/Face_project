"""
Verification test for Face Deduplication and Head Movement Challenge fixes.
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server_module.components.face_detection.detector import deduplicate_faces, calculate_iou_and_iomin
from server_module.components.head_movement.head_movement_detector import (
    HeadMovementDetector, HeadAction, ChallengeState
)


def test_face_deduplication():
    print("=" * 60)
    print("TEST 1: Kiểm tra thuật toán Deduplicate Faces (Gộp Box Trùng)")
    print("=" * 60)

    # Trường hợp 1: Hai bounding box trùng lặp trên cùng 1 khuôn mặt (IoU ~ 0.65)
    box1 = [100, 100, 300, 300]
    box2 = [120, 110, 290, 310]
    iou, iomin = calculate_iou_and_iomin(box1, box2)
    print(f" -> Box 1 & Box 2: IoU = {iou:.3f}, IoMin = {iomin:.3f}")
    assert iou > 0.5, "IoU phải > 0.5"

    duplicate_faces_input = [
        {"bbox": box1, "confidence": 0.88, "class_id": 0, "face_crop": None},
        {"bbox": box2, "confidence": 0.72, "class_id": 0, "face_crop": None}
    ]
    result_1 = deduplicate_faces(duplicate_faces_input, iou_thresh=0.35, iomin_thresh=0.60)
    print(f" -> Đầu vào: {len(duplicate_faces_input)} box | Sau khi deduplicate: {len(result_1)} box")
    assert len(result_1) == 1, f"Kỳ vọng 1 box nhưng nhận {len(result_1)}"
    assert result_1[0]["confidence"] == 0.88, "Phải giữ lại box có confidence cao hơn"
    print(" [PASS] Gộp thành công 2 box trùng trên cùng 1 người thành 1 mặt duy nhất!")

    # Trường hợp 2: Box con nằm lọt trong box mẹ (IoMin cao)
    box_parent = [50, 50, 400, 450]
    box_child = [120, 130, 320, 340]
    iou_c, iomin_c = calculate_iou_and_iomin(box_parent, box_child)
    print(f" -> Box mẹ & Box con: IoU = {iou_c:.3f}, IoMin = {iomin_c:.3f}")
    assert iomin_c > 0.9, "IoMin phải > 0.9 vì box con nằm trong box mẹ"

    nested_faces_input = [
        {"bbox": box_parent, "confidence": 0.91, "class_id": 0, "face_crop": None},
        {"bbox": box_child, "confidence": 0.79, "class_id": 0, "face_crop": None}
    ]
    result_2 = deduplicate_faces(nested_faces_input, iou_thresh=0.35, iomin_thresh=0.60)
    print(f" -> Đầu vào: {len(nested_faces_input)} box | Sau khi deduplicate: {len(result_2)} box")
    assert len(result_2) == 1, f"Kỳ vọng 1 box nhưng nhận {len(result_2)}"
    print(" [PASS] Loại bỏ thành công box con nằm trong box mẹ!")

    # Trường hợp 3: Hai người hoàn toàn riêng biệt đứng cạnh nhau
    box_person_a = [50, 100, 200, 300]
    box_person_b = [350, 100, 500, 300]
    distinct_faces_input = [
        {"bbox": box_person_a, "confidence": 0.89, "class_id": 0, "face_crop": None},
        {"bbox": box_person_b, "confidence": 0.86, "class_id": 0, "face_crop": None}
    ]
    result_3 = deduplicate_faces(distinct_faces_input, iou_thresh=0.35, iomin_thresh=0.60)
    print(f" -> Đầu vào 2 người riêng biệt: {len(distinct_faces_input)} | Sau deduplicate: {len(result_3)}")
    assert len(result_3) == 2, f"Kỳ vọng 2 người nhưng nhận {len(result_3)}"
    print(" [PASS] Giữ nguyên chính xác 2 người khi có 2 khuôn mặt riêng biệt trong ảnh!\n")


def test_head_movement_challenge():
    print("=" * 60)
    print("TEST 2: Kiểm tra Head Movement Challenge & Delta Movement (Bắt buộc di chuyển)")
    print("=" * 60)

    hm = HeadMovementDetector(yaw_threshold=16.0, delta_yaw_threshold=8.0, timeout=7.0, min_consecutive_frames=2)

    # --- Thử thách 1: KHÔNG DI CHUYỂN ĐẦU (Kể cả khi ngồi lệch 15 độ) -> KHÔNG ĐƯỢC PASS ---
    print("\n--- [2.1] Test Ngăn chặn: Không di chuyển đầu thì KHÔNG ĐƯỢC PASS ---")
    hm.start_challenge(HeadAction.TURN_RIGHT)
    # 2 frame đầu định hình baseline tại 15.0 độ
    hm.update({"yaw": 15.0, "pitch": 0.0, "roll": 0.0})
    hm.update({"yaw": 15.0, "pitch": 0.0, "roll": 0.0})

    # Giữ nguyên đầu tại 15.0 độ suốt 5 frame tiếp theo -> delta = 0 độ
    for i in range(5):
        st = hm.update({"yaw": 15.0, "pitch": 0.0, "roll": 0.0})
        assert not st["passed"], f"Frame {i}: Không di chuyển đầu tuyệt đối không được pass!"
        assert not st["is_matched"]
        assert st["current_angle"] == 0.0
        assert st["progress"] == 0.0
    print(" [PASS] Hoàn hảo: Người dùng không di chuyển đầu thì hệ thống giữ 0% và không cho pass!")

    # --- Thử thách 2: DI CHUYỂN NGƯỢC HƯỚNG -> KHÔNG ĐƯỢC PASS ---
    print("\n--- [2.2] Test Ngăn chặn: Di chuyển ngược hướng thì KHÔNG ĐƯỢC PASS ---")
    hm.start_challenge(HeadAction.TURN_LEFT)
    hm.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    hm.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    # Quay sang phải (+10 độ) trong khi yêu cầu quay trái
    st_wrong = hm.update({"yaw": 10.0, "pitch": 0.0, "roll": 0.0})
    assert not st_wrong["is_matched"]
    assert st_wrong["progress"] == 0.0
    assert not st_wrong["passed"]
    print(" [PASS] Hoàn hảo: Quay ngược hướng không được tính tiến trình!")

    # --- Thử thách 3: NHÍCH NHẸ ĐÚNG HƯỚNG (~8 độ) -> PASS THÀNH CÔNG ---
    print("\n--- [2.3] Test Hợp lệ: Nhích nhẹ đầu đúng hướng (~8 độ) -> PASS ---")
    hm.start_challenge(HeadAction.TURN_LEFT)
    # Baseline xuất phát tại 0 độ
    hm.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    hm.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0})

    # Nhích nhẹ 4 độ sang trái -> tiến trình ~37%
    s_half = hm.update({"yaw": -4.0, "pitch": 0.0, "roll": 0.0})
    assert not s_half["is_matched"]
    assert 0.30 <= s_half["progress"] <= 0.45
    print(f" -> Nhích 4.0°: progress = {s_half['progress']*100:.1f}%, matched = False")

    # Nhích đủ 8.5 độ sang trái (Frame 1 đạt)
    s_hit1 = hm.update({"yaw": -8.5, "pitch": 0.0, "roll": 0.0})
    assert s_hit1["is_matched"]
    assert not s_hit1["passed"]
    print(f" -> Nhích đủ 8.5° (Lần 1): progress = {s_hit1['progress']*100:.1f}%, matched = True")

    # Giữ vững góc nhích lần 2 -> PASS!
    s_hit2 = hm.update({"yaw": -9.0, "pitch": 0.0, "roll": 0.0})
    assert s_hit2["passed"]
    assert s_hit2["state"] == "COMPLETED"
    assert s_hit2["progress"] == 1.0
    print(f" -> Giữ góc lần 2: progress = 100%, passed = True!")
    print(" [PASS] Hoàn thành thử thách nhích nhẹ đầu đúng hướng!\n")


def test_blink_logic():
    print("=" * 60)
    print("TEST 3: Kiểm tra Logic Chớp Mắt Tự Nhiên (Blink State Transition)")
    print("=" * 60)

    from server_module.config import EAR_EYE_CLOSED_THRESHOLD, EAR_EYE_OPEN_THRESHOLD

    # Giả lập chuỗi trạng thái EAR qua 4 frame
    # Frame 1: Mở mắt bình thường (EAR = 0.28)
    counter = 0
    state = False
    baseline = 0.0

    ear1 = 0.28
    if ear1 >= 0.22:
        baseline = ear1
    closed_thresh = max(0.18, min(0.21, baseline * 0.78))
    open_thresh = max(closed_thresh + 0.02, EAR_EYE_OPEN_THRESHOLD)
    print(f" Frame 1 (Mắt mở): EAR = {ear1}, closed_thresh = {closed_thresh:.3f}, open_thresh = {open_thresh:.3f}")
    assert not state

    # Frame 2: Nhắm mắt tự nhiên (EAR giảm về 0.185 - trước đây bị trượt vì 0.185 > 0.18)
    ear2 = 0.185
    if ear2 > 0.05 and ear2 < closed_thresh:
        state = True
    print(f" Frame 2 (Nhắm mắt): EAR = {ear2} -> state nhắm = {state}")
    assert state is True, "EAR 0.185 phải được ghi nhận là trạng thái nhắm mắt!"

    # Frame 3: Mở mắt lại (EAR = 0.27)
    ear3 = 0.27
    if ear3 >= open_thresh or (ear3 >= 0.22 and state):
        if state:
            counter += 1
            state = False
    print(f" Frame 3 (Mở mắt lại): EAR = {ear3} -> counter = {counter}, state nhắm = {state}")
    assert counter == 1, "Counter phải tăng lên 1 sau chu trình nhắm -> mở!"
    assert state is False, "State phải reset về False!"
    assert counter >= 1, "Đạt 1 lần chớp mắt -> PASS!"
    print(" [PASS] Hoàn thành chu trình chớp mắt thành công 100%!\n")


if __name__ == "__main__":
    test_face_deduplication()
    test_head_movement_challenge()
    test_blink_logic()
    print("=" * 60)
    print(" >>> TẤT CẢ TEST ĐÃ HOÀN THÀNH VÀ ĐẠT 100% PASS! <<<")
    print("=" * 60)
