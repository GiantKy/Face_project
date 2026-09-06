"""
Test script for server_module (E-KYC YOLO Engine).
Kiểm thử trực tiếp các hàm của server_module trên ảnh mẫu trong data_raw/.
"""

import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server_module import EKYCPipelineServer, load_image, image_to_base64


def test_server_pipeline():
    print("=" * 70)
    print("      KIỂM THỬ E-KYC SERVER MODULE (YOLO ENGINE)")
    print("=" * 70)

    test_img_path = os.path.join(PROJECT_ROOT, "data_raw", "0.jpg")
    if not os.path.exists(test_img_path):
        print(f"[CẢNH BÁO] Không tìm thấy ảnh test: {test_img_path}")
        return

    print(f"[TEST 1] Đang nạp ảnh test: {test_img_path}")
    img = load_image(test_img_path)
    print(f" -> Kích thước ảnh: {img.shape}")

    # Chuyển đổi thử sang Base64
    b64_str = image_to_base64(img)
    print(f" -> Base64 string length: {len(b64_str)} chars")

    # Khởi tạo Pipeline Server
    print("\n[TEST 2] Khởi tạo EKYCPipelineServer...")
    server = EKYCPipelineServer()

    # 1. Test validate_pose
    print("\n[TEST 3] Kiểm tra validate_pose() với ảnh Base64...")
    pose_res = server.validate_pose(b64_str)
    print(" -> Kết quả Pose:")
    print(json.dumps(pose_res, ensure_ascii=False, indent=2))

    # 2. Test check_antispoof
    print("\n[TEST 4] Kiểm tra check_antispoof() với file path...")
    spoof_res = server.check_antispoof(test_img_path)
    # Ẩn bớt trường Base64 crop khi in ra màn hình để tránh làm đầy console
    print_spoof = {k: v for k, v in spoof_res.items() if k != "crop_face_base64"}
    print(" -> Kết quả Anti-Spoof:")
    print(json.dumps(print_spoof, ensure_ascii=False, indent=2))

    # 3. Test full_verify
    print("\n[TEST 5] Kiểm tra full_verify() tổng hợp quy trình...")
    test_out_dir = os.path.join(PROJECT_ROOT, "output", "test_server_module")
    report = server.full_verify(
        image_input=img,
        img_id="test_0",
        blink_passed=True,
        head_movement_passed=True,
        head_action_name="TURN_LEFT",
        output_dir=test_out_dir,
        save_visuals=True
    )
    print(" -> Kết quả Final Decision:")
    print(json.dumps(report["final_decision"], ensure_ascii=False, indent=2))
    print(f"\n[OK] Đã xuất báo cáo và các ảnh HUD vào: {test_out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    test_server_pipeline()
