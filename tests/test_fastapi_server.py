"""
Test script for FastAPI Server (E-KYC Face Verification).
Kiểm thử các endpoint của FastAPI server:
1. Healthcheck: GET /api/v1/health
2. Web UI: GET /
3. Validate Pose: POST /api/v1/validate-pose (Multipart)
4. Full Verify: POST /api/v1/verify (Multipart)
5. Full Verify: POST /api/v1/verify (JSON Base64)
"""

import os
import sys
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from server_module.app import app

def run_test():
    print("=" * 70)
    print("        KIỂM THỬ TÍCH HỢP FASTAPI E-KYC SERVER")
    print("=" * 70)

    # Tìm ảnh test
    sample_img_path = os.path.join(PROJECT_ROOT, "data_raw", "0.jpg")
    if not os.path.exists(sample_img_path):
        print(f"[CẢNH BÁO] Không tìm thấy ảnh {sample_img_path}")
        return

    # Sử dụng TestClient với context manager để kích hoạt lifespan (nạp models)
    with TestClient(app) as client:
        # TEST 1: Healthcheck
        print("\n[TEST 1] GET /api/v1/health")
        res_health = client.get("/api/v1/health")
        print(f"Status Code: {res_health.status_code}")
        print("Response:", json.dumps(res_health.json(), indent=2, ensure_ascii=False))
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"

        # TEST 2: Web UI HTML
        print("\n[TEST 2] GET /")
        res_ui = client.get("/")
        print(f"Status Code: {res_ui.status_code}")
        assert res_ui.status_code == 200
        assert "<title>eKYC AI" in res_ui.text
        print(" -> Giao diện HTML được phục vụ thành công!")

        # TEST 3: Validate Pose (Multipart)
        print("\n[TEST 3] POST /api/v1/validate-pose (Multipart File)")
        with open(sample_img_path, "rb") as f:
            res_pose = client.post(
                "/api/v1/validate-pose",
                files={"file": ("0.jpg", f, "image/jpeg")}
            )
        print(f"Status Code: {res_pose.status_code}")
        pose_data = res_pose.json()
        print(f" -> Has face: {pose_data['has_face']}, Pose valid: {pose_data['is_valid']}, Latency: {pose_data['processing_time_ms']} ms")
        assert res_pose.status_code == 200

        # TEST 4: Full Verify (Multipart - Mô phỏng Node.js gửi lên)
        print("\n[TEST 4] POST /api/v1/verify (Multipart Form-Data chuẩn Node.js)")
        with open(sample_img_path, "rb") as f:
            res_verify = client.post(
                "/api/v1/verify",
                files={"file": ("0.jpg", f, "image/jpeg")},
                data={
                    "img_id": "TEST_NODEJS_UPLOAD",
                    "user_id": "USER_999",
                    "blink_passed": "true",
                    "head_passed": "true",
                    "head_action": "TURN_LEFT",
                    "return_crop_image": "true",
                    "return_annotated_image": "true"
                }
            )
        print(f"Status Code: {res_verify.status_code}")
        assert res_verify.status_code == 200
        verify_data = res_verify.json()

        print(f" -> Approved: {verify_data['approved']}")
        print(f" -> Verdict: {verify_data['verdict']}")
        print(f" -> Is Real: {verify_data['is_real']} (Confidence: {verify_data['confidence']*100:.1f}%)")
        print(f" -> Processing Time: {verify_data['processing_time_ms']} ms")
        print(f" -> Has Crop Base64: {bool(verify_data.get('crop_face_base64'))}")
        print(f" -> Has Annotated Base64: {bool(verify_data.get('annotated_image_base64'))}")
        print(f" -> Reasons: {verify_data['reasons']}")
        print(" -> Criteria Detail:")
        for k, v in verify_data["criteria"].items():
            print(f"    * {k}: {'PASS' if v else 'FAIL'}")

        # TEST 5: Full Verify (JSON Base64)
        print("\n[TEST 5] POST /api/v1/verify (JSON Base64 Payload)")
        import base64
        with open(sample_img_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")

        res_json = client.post(
            "/api/v1/verify",
            json={
                "image_base64": f"data:image/jpeg;base64,{b64_str}",
                "img_id": "TEST_JSON_B64",
                "blink_passed": True,
                "head_passed": True,
                "return_crop_image": True,
                "return_annotated_image": True
            }
        )
        print(f"Status Code: {res_json.status_code}")
        assert res_json.status_code == 200
        print(f" -> Verdict: {res_json.json()['verdict']}, Latency: {res_json.json()['processing_time_ms']} ms")

        # TEST 6: Active Liveness - Blink Frame
        print("\n[TEST 6] POST /api/v1/liveness/blink-frame")
        with open(sample_img_path, "rb") as f:
            res_blink = client.post(
                "/api/v1/liveness/blink-frame",
                files={"file": ("0.jpg", f, "image/jpeg")},
                data={"blink_counter": 0, "blink_state": "false"}
            )
        print(f"Status Code: {res_blink.status_code}")
        assert res_blink.status_code == 200
        blink_data = res_blink.json()
        print(f" -> Has Face: {blink_data.get('has_face')}, EAR Avg: {blink_data.get('ear_avg')}, Passed: {blink_data.get('passed')}")

        # TEST 7: Active Liveness - Head Challenge
        print("\n[TEST 7] POST /api/v1/liveness/start-head & update-head")
        res_start_head = client.post("/api/v1/liveness/start-head")
        assert res_start_head.status_code == 200
        head_start_data = res_start_head.json()
        print(f" -> Started Head Action: {head_start_data.get('action')}, Prompt: {head_start_data.get('prompt')}")

        with open(sample_img_path, "rb") as f:
            res_update_head = client.post(
                "/api/v1/liveness/update-head",
                files={"file": ("0.jpg", f, "image/jpeg")}
            )
        assert res_update_head.status_code == 200
        head_upd_data = res_update_head.json()
        print(f" -> Head Progress: {head_upd_data.get('progress')}, Time Left: {head_upd_data.get('time_left')}s, Passed: {head_upd_data.get('passed')}")

    print("\n" + "=" * 70)
    print("      TẤT CẢ 7 BƯỚC KIỂM THỬ FASTAPI SERVER ĐÃ THÀNH CÔNG!")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
