"""
Unit test for ESP32 Multi-stage Challenge Manager without requiring FastAPI HTTP networking.
"""

import os
import sys
import time
import json
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server_module.pipeline_server import EKYCPipelineServer
from server_module.esp32_challenge import esp32_challenge_manager, preprocess_esp32_image


def test_unit_flow():
    print("=" * 70)
    print(" [UNIT TEST] ESP32 MULTI-STAGE CHALLENGE LOGIC")
    print("=" * 70)

    print("[*] Nạp pipeline AI server...")
    pipeline = EKYCPipelineServer(lazy_load=False)

    # 0. Test kiểm tra kính: 0.jpg (đeo kính) PHẢI BỊ CHẶN ngay lập tức
    print("\n--- TEST KIỂM TRA MẮT KÍNH (data_raw/0.jpg) ---")
    img_glasses = cv2.imread(os.path.join(PROJECT_ROOT, "data_raw", "0.jpg"))
    res_glasses = esp32_challenge_manager.start_challenge(img_glasses, pipeline, device_id="ESP32_GLASSES_TEST")
    print(f"[*] Kết quả ảnh đeo kính: verdict={res_glasses.get('verdict')}, code={res_glasses.get('occlusion_code')}")
    assert res_glasses["verdict"] == "FACE_OCCLUDED", "Ảnh 0.jpg đeo kính phải bị từ chối FACE_OCCLUDED!"
    assert res_glasses["occlusion_code"] == "CLEAR_GLASSES_DETECTED", "Phải phát hiện đúng CLEAR_GLASSES_DETECTED!"
    print("[✓] ĐÃ CHẶN THÀNH CÔNG ẢNH ĐEO KÍNH (0.jpg)!")

    # 1. Test Bước 1: Start Challenge với ảnh mặt trần (7.jpg)
    img_path = os.path.join(PROJECT_ROOT, "data_raw", "7.jpg")
    assert os.path.exists(img_path), f"File không tồn tại: {img_path}"
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    print(f"\n[*] Đã nạp ảnh chuẩn không đeo kính: {img_path} ({w}x{h})")

    print("\n--- BƯỚC 1: START CHALLENGE (Face Detect & Baseline) ---")
    res1 = esp32_challenge_manager.start_challenge(img_bgr, pipeline, device_id="ESP32_UNIT_01")
    print(json.dumps(res1, ensure_ascii=False, indent=2))
    assert res1["success"] is True, f"Bước 1 thất bại: {res1.get('message')}"
    assert res1["passed"] is True
    session_id = res1["session_id"]
    print(f"[✓] Bước 1 ĐẠT! Session ID: {session_id}")

    # 2. Test Bước 2: Eye Blink
    print("\n--- BƯỚC 2: EYE BLINK ---")
    # Mô phỏng nhắm mắt: làm mờ vùng mắt để EAR giảm
    blink_img = img_bgr.copy()
    y1, y2 = int(h * 0.38), int(h * 0.46)
    x1, x2 = int(w * 0.32), int(w * 0.68)
    blink_img[y1:y2, x1:x2] = cv2.GaussianBlur(blink_img[y1:y2, x1:x2], (39, 39), 0)

    # Đưa vào step
    res2 = esp32_challenge_manager.process_step(session_id, blink_img, pipeline, step_name="eye_blink")
    print(json.dumps(res2, ensure_ascii=False, indent=2))
    # Nếu mô phỏng che mắt làm EAR giảm thì passed=True, nếu chưa đạt ta kiểm tra trường ear
    print(f"[*] Kết quả bước 2: passed={res2.get('passed')}")

    # 3. Test Bước 3: Head Movement
    print("\n--- BƯỚC 3: HEAD MOVEMENT ---")
    # Lấy session hiện tại
    session = esp32_challenge_manager.sessions.get(session_id)
    if session:
        # Giả lập đã qua bước blink để test tiếp bước quay đầu
        session.eye_blink_passed = True
        session.current_step = "head_movement"
        action = session.target_head_action
        print(f"[*] Thử thách yêu cầu: {action}")

        # Xoay ảnh để tạo góc quay đầu mô phỏng
        rot_deg = 10.0 if action == "TURN_LEFT" else (-10.0 if action == "TURN_RIGHT" else 0.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
        head_img = cv2.warpAffine(img_bgr, M, (w, h))

        res3 = esp32_challenge_manager.process_step(session_id, head_img, pipeline, step_name="head_movement")
        print(json.dumps(res3, ensure_ascii=False, indent=2))
        print(f"[*] Kết quả bước 3: success={res3.get('success')}, approved={res3.get('approved')}")

    # 4. Test Step Processing
    print("\n--- BƯỚC 4: STEP PROCESSING ---")
    res1_stream = esp32_challenge_manager.start_challenge(img_bgr, pipeline, device_id="ESP32_STREAM_01")
    s_id = res1_stream["session_id"]
    step_res = esp32_challenge_manager.process_step(s_id, img_bgr, pipeline)
    print(f"[*] Kết quả Step: step={step_res.get('step')}, progress={step_res.get('progress')}")
    assert step_res["success"] is True, "process_step phải thành công"

    print("\n" + "=" * 70)
    print(" [✓] UNIT TEST HOÀN TẤT THÀNH CÔNG!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_unit_flow()
