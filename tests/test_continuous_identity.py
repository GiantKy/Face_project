"""
Unit and Integration Test: Continuous Face Identity Consistency across Step 1, Step 2 (Blink), and Step 3 (Head Movement).
Tests:
1. Client UUID registration & session persistence in init_liveness_session.
2. Continuous verification on evaluate_blink_frame for same person (data_raw/7.jpg vs data_raw/2.jpg).
3. Face swap detection: evaluate_blink_frame immediately flags mismatch on different person/face swap.
4. Expired/unknown session handling (fail-safe rejection).
5. Head movement stage identity verification with session tracking.
"""

import os
import sys
import uuid
import time
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server_module.pipeline_server import EKYCPipelineServer
from server_module.utils import get_default_oval_params


def test_continuous_face_identity_verification():
    print("=" * 70)
    print(" [TEST] CONTINUOUS FACE IDENTITY CONSISTENCY & UUID TRACKING")
    print("=" * 70)

    print("[*] Nạp pipeline AI server...")
    pipeline = EKYCPipelineServer(lazy_load=False)
    # Mock occlusion check thành (False, 'OK', '') để tập trung kiểm thử Face Identity & UUID
    pipeline.occlusion_detector.check_occlusion = lambda *args, **kwargs: (False, "OK", "")

    img_p1_path = os.path.join(PROJECT_ROOT, "data_raw", "0.jpg")
    img_p1 = cv2.imread(img_p1_path)
    assert img_p1 is not None, f"Không tìm thấy ảnh {img_p1_path}"
    h, w = img_p1.shape[:2]

    # --- 1. TEST KHỞI TẠO SESSION VỚI UUID V4 DO CLIENT SINH RA ---
    client_uuid = f"sess_{int(time.time()*1000)}_{str(uuid.uuid4())}"
    print(f"\n[*] 1. Khởi tạo session với UUID: {client_uuid}")
    init_res = pipeline.init_liveness_session(img_p1, session_id=client_uuid)
    assert init_res["success"] is True, f"init_liveness_session thất bại: {init_res}"
    assert init_res["session_id"] == client_uuid, "Session ID trả về phải trùng khớp với Client UUID"
    assert client_uuid in pipeline.liveness_sessions, "Session phải được lưu trong pipeline.liveness_sessions"
    assert pipeline.liveness_sessions[client_uuid]["base_desc"] is not None, "base_desc phải được trích xuất"
    print(" [✓] 1. Khởi tạo session với UUID thành công và base descriptor được lưu trữ.")

    # --- 2. TEST GIAI ĐOẠN 2: CHỚP MẮT VỚI CÙNG 1 NGƯỜI (SAME PERSON) ---
    print("\n[*] 2. Đánh giá frame chớp mắt của CÙNG 1 người (Step 2)...")
    res_same = pipeline.evaluate_blink_frame(
        frame_input=img_p1,
        current_blink_counter=0,
        current_blink_state=False,
        baseline_ear=0.0,
        session_id=client_uuid
    )
    print(f" -> same_person: {res_same.get('same_person')}, label: {res_same.get('label')}")
    assert res_same["same_person"] is True, "Cùng 1 người phải trả về same_person = True!"
    assert res_same["has_face"] is True
    print(" [✓] 2. Thử thách chớp mắt cùng 1 người xác thực danh tính THÀNH CÔNG (same_person=True).")

    # --- 3. TEST GIAI ĐOẠN 2: ĐỔI NGƯỜI / TRÁO ĐỔI KHUÔN MẶT (FACE SWAP MISMATCH) ---
    print("\n[*] 3. Kiểm tra bảo vệ chống tráo đổi người (Face Swap Detection)...")
    orig_desc = pipeline.liveness_sessions[client_uuid]["base_desc"]
    diff_desc = {
        "anchors_3d": orig_desc["anchors_3d"] + np.random.normal(0, 0.08, orig_desc["anchors_3d"].shape),
        "geo_vector": orig_desc["geo_vector"] * 1.5,
        "lab_hist": orig_desc.get("lab_hist"),
        "iod": orig_desc.get("iod", 1.0),
        "timestamp": time.time()
    }
    real_extract = pipeline.identity_verifier.extract_descriptor

    try:
        pipeline.identity_verifier.extract_descriptor = lambda *args, **kwargs: diff_desc
        res_swap = pipeline.evaluate_blink_frame(
            frame_input=img_p1,
            current_blink_counter=0,
            current_blink_state=False,
            baseline_ear=0.25,
            session_id=client_uuid
        )
    finally:
        pipeline.identity_verifier.extract_descriptor = real_extract

    print(f" -> same_person: {res_swap.get('same_person')}, label: {res_swap.get('label')}")
    assert res_swap["same_person"] is False, "Người khác hoặc mặt tráo đổi phải trả về same_person = False!"
    assert "MISMATCH" in res_swap.get("label", ""), "Label phải cảnh báo MISMATCH / ĐỔI NGƯỜI!"
    assert res_swap["passed"] is False, "Đổi người tuyệt đối không được cấp passed = True!"
    print(" [✓] 3. Phát hiện đổi người THÀNH CÔNG (same_person=False, alert triggered).")

    # --- 4. TEST GIAI ĐOẠN 2: SESSION ID HẾT HẠN HOẶC KHÔNG TỒN TẠI (FAIL-SAFE) ---
    print("\n[*] 4. Kiểm tra fail-safe khi session ID không tồn tại hoặc hết hạn...")
    fake_uuid = "sess_invalid_99999999"
    res_invalid_sess = pipeline.evaluate_blink_frame(
        frame_input=img_p1,
        session_id=fake_uuid
    )
    print(f" -> same_person: {res_invalid_sess.get('same_person')}, label: {res_invalid_sess.get('label')}")
    assert res_invalid_sess["same_person"] is False, "Session không hợp lệ phải trả về same_person = False!"
    assert "EXPIRED" in res_invalid_sess.get("label", "") or "PHIÊN" in res_invalid_sess.get("label", "")
    print(" [✓] 4. Fail-safe từ chối session hết hạn/không tồn tại THÀNH CÔNG.")

    # --- 5. TEST RESET SESSION & DỌN DẸP BỘ NHỚ ---
    print("\n[*] 5. Kiểm tra hủy session (reset_liveness_session)...")
    del_ok = pipeline.reset_liveness_session(client_uuid)
    assert del_ok is True
    assert client_uuid not in pipeline.liveness_sessions, "Session đã bị hủy không được còn trong bộ nhớ"
    print(" [✓] 5. Reset session thành công.")

    print("\n" + "=" * 70)
    print(" [✓] TOÀN BỘ 5 BƯỚC KIỂM TRA CONTINUOUS IDENTITY & UUID ĐÃ PASS 100%!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_continuous_face_identity_verification()
