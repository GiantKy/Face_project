"""
Unit and Integration Tests for Continuous Oval Face Fit and UUID Session Management.
Tests:
1. Geometric Oval Fit: Small face (is_too_far=True) vs Standard face (fit_oval=True).
2. Step 1 (init_liveness_session):
   - Reject distant face with FACE_TOO_FAR (preventing low-res descriptor & false session).
   - Accept standard face and issue session UUID.
3. Step 2 (evaluate_blink_frame):
   - Distant face returns fit_oval=False and same_person=True (guiding user closer without false swap alert).
   - Standard face of same person returns fit_oval=True, same_person=True.
   - Face swap returns fit_oval=True, same_person=False with MISMATCH warning.
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
from server_module.utils import get_default_oval_params, check_face_oval_fit
from server_module.config import MIN_FACE_HEIGHT, OVAL_FIT_MIN_RATIO, OVAL_FIT_MAX_RATIO


def test_geometric_oval_fit():
    print("\n--- TEST 1: KIỂM TRA HÌNH HỌC OVAL FIT (check_face_oval_fit) ---")
    w, h = 640, 480
    oval_center, oval_axes = get_default_oval_params(w, h)
    cx, cy = oval_center
    ax, ay = oval_axes
    oval_h = 2 * ay # 364px

    print(f"[*] Oval Center: {oval_center}, Axes: {oval_axes}, Chiều cao Oval: {oval_h}px")

    # Case 1: Mặt nhỏ ở xa (face_h = 100px, ~27.5% oval < 40%)
    lms_small = [
        (cx, cy - 50),
        (cx - 35, cy),
        (cx + 35, cy),
        (cx, cy + 50)
    ]
    fit_small = check_face_oval_fit(lms_small, w, h, oval_center, oval_axes, min_ratio=OVAL_FIT_MIN_RATIO, min_face_height=MIN_FACE_HEIGHT)
    print(f" -> Mặt nhỏ ({fit_small['face_size_h']}px, {fit_small['ratio_to_oval']*100:.1f}% oval): fit_oval={fit_small['fit_oval']}, is_too_far={fit_small['is_too_far']}")
    assert fit_small["fit_oval"] is False, "Mặt nhỏ không được fit_oval!"
    assert fit_small["is_too_far"] is True, "Mặt nhỏ phải báo is_too_far = True!"

    # Case 2: Mặt chuẩn vừa vặn (face_h = 240px, ~65.9% oval trong khoảng 40% - 90%)
    lms_standard = [
        (cx, cy - 120),
        (cx - 75, cy),
        (cx + 75, cy),
        (cx, cy + 120)
    ]
    fit_std = check_face_oval_fit(lms_standard, w, h, oval_center, oval_axes, min_ratio=OVAL_FIT_MIN_RATIO, min_face_height=MIN_FACE_HEIGHT)
    print(f" -> Mặt chuẩn ({fit_std['face_size_h']}px, {fit_std['ratio_to_oval']*100:.1f}% oval): fit_oval={fit_std['fit_oval']}, is_too_far={fit_std['is_too_far']}, is_too_close={fit_std['is_too_close']}")
    assert fit_std["fit_oval"] is True, "Mặt chuẩn phải fit_oval = True!"
    assert fit_std["is_too_far"] is False, "Mặt chuẩn không được báo is_too_far!"
    assert fit_std["is_too_close"] is False, "Mặt chuẩn không được báo is_too_close!"
    assert fit_std["face_in_oval"] is True, "Mặt chuẩn phải nằm trong oval!"

    # Case 3: Mặt quá gần (face_h = 340px, ~93.4% oval > 90%)
    lms_too_close = [
        (cx, cy - 170),
        (cx - 90, cy),
        (cx + 90, cy),
        (cx, cy + 170)
    ]
    fit_close = check_face_oval_fit(lms_too_close, w, h, oval_center, oval_axes, min_ratio=OVAL_FIT_MIN_RATIO, min_face_height=MIN_FACE_HEIGHT)
    print(f" -> Mặt quá gần ({fit_close['face_size_h']}px, {fit_close['ratio_to_oval']*100:.1f}% oval): fit_oval={fit_close['fit_oval']}, is_too_close={fit_close['is_too_close']}")
    assert fit_close["fit_oval"] is False, "Mặt quá gần không được fit_oval!"
    assert fit_close["is_too_close"] is True, "Mặt quá gần phải báo is_too_close = True!"

    # Case 4: Mặt lệch tâm (dx = 110px > 0.35 * ax = 41px)
    lms_off_center = [
        (cx + 90, cy - 91),
        (cx + 10, cy),
        (cx + 170, cy),
        (cx + 90, cy + 91)
    ]
    fit_off = check_face_oval_fit(lms_off_center, w, h, oval_center, oval_axes, min_ratio=OVAL_FIT_MIN_RATIO, min_face_height=MIN_FACE_HEIGHT)
    print(f" -> Mặt lệch tâm: fit_oval={fit_off['fit_oval']}, is_off_center={fit_off['is_off_center']}")
    assert fit_off["fit_oval"] is False, "Mặt lệch tâm không được fit_oval!"
    assert fit_off["is_off_center"] is True, "Mặt lệch tâm phải báo is_off_center = True!"

    print(" [✓] TEST 1 THÀNH CÔNG: check_face_oval_fit phân định chính xác kích thước và vị trí!")


def test_init_session_and_continuous_blink():
    print("\n--- TEST 2: KIỂM TRA INIT SESSION VÀ CONTINUOUS BLINK VERIFICATION ---")
    pipeline = EKYCPipelineServer(lazy_load=False)
    # Mock occlusion check thành (False, 'OK', '') để tập trung kiểm tra Oval Fit & Identity
    pipeline.occlusion_detector.check_occlusion = lambda *args, **kwargs: (False, "OK", "")

    img_std_path = os.path.join(PROJECT_ROOT, "data_raw", "0.jpg")
    img_std = cv2.imread(img_std_path)
    assert img_std is not None, f"Không tìm thấy ảnh {img_std_path}"

    img_far_path = os.path.join(PROJECT_ROOT, "data_raw", "2.jpg")
    img_far = cv2.imread(img_far_path)
    assert img_far is not None, f"Không tìm thấy ảnh {img_far_path}"

    # 1. Thử khởi tạo phiên bằng ảnh mặt ở xa (data_raw/2.jpg) -> Phải bị từ chối với FACE_TOO_FAR
    print("\n[*] 2.1 Thử khởi tạo phiên bằng ảnh ở xa (2.jpg)...")
    res_far_init = pipeline.init_liveness_session(img_far)
    print(f" -> success: {res_far_init.get('success')}, error: {res_far_init.get('error')}, message: {res_far_init.get('message')}")
    assert res_far_init["success"] is False, "Ảnh ở xa tuyệt đối KHÔNG được cấp session UUID!"
    assert res_far_init["error"] == "FACE_TOO_FAR", f"Lỗi phải là FACE_TOO_FAR, nhận được: {res_far_init.get('error')}"
    print(" [✓] 2.1 Chặn thành công ảnh ở xa ở Bước 1 (FACE_TOO_FAR)!")

    # 2. Khởi tạo phiên bằng ảnh chuẩn vừa vặn Oval (data_raw/0.jpg) -> Phải thành công
    client_uuid = f"sess_{int(time.time()*1000)}_{str(uuid.uuid4())}"
    print(f"\n[*] 2.2 Khởi tạo phiên bằng ảnh chuẩn vừa vặn (0.jpg) với UUID: {client_uuid}...")
    res_std_init = pipeline.init_liveness_session(img_std, session_id=client_uuid)
    print(f" -> success: {res_std_init.get('success')}, fit_oval: {res_std_init.get('fit_oval')}, session_id: {res_std_init.get('session_id')}")
    assert res_std_init["success"] is True, f"Khởi tạo session chuẩn thất bại: {res_std_init}"
    assert res_std_init["fit_oval"] is True, "fit_oval phải là True!"
    assert client_uuid in pipeline.liveness_sessions, "Session phải được ghi nhớ trong pipeline"
    print(" [✓] 2.2 Khởi tạo session thành công với ảnh chuẩn khớp Oval!")

    # 3. Giai đoạn 2 (Chớp mắt): Khi người dùng lùi ra xa (2.jpg)
    # Phải trả về: fit_oval=False, is_too_far=True, same_person=True (KHÔNG được báo nhầm là đổi người!)
    print("\n[*] 2.3 Gửi frame ở xa (2.jpg) trong Bước 2 (Chớp mắt)...")
    res_far_blink = pipeline.evaluate_blink_frame(
        frame_input=img_far,
        current_blink_counter=0,
        current_blink_state=False,
        baseline_ear=0.0,
        session_id=client_uuid
    )
    print(f" -> fit_oval: {res_far_blink.get('fit_oval')}, is_too_far: {res_far_blink.get('is_too_far')}, same_person: {res_far_blink.get('same_person')}, label: {res_far_blink.get('label')}")
    assert res_far_blink["fit_oval"] is False, "Frame ở xa hoặc lệch phải có fit_oval = False!"
    assert res_far_blink["same_person"] is True, "Frame ở xa KHÔNG ĐƯỢC báo đổi người (same_person phải là True)!"
    print(" [✓] 2.3 Frame ở xa được phân biệt chính xác: yêu cầu căn chỉnh oval và KHÔNG báo nhầm đổi người!")

    # 4. Giai đoạn 2 (Chớp mắt): Khi mặt đã khớp Oval của cùng 1 người (0.jpg)
    print("\n[*] 2.4 Gửi frame chuẩn khớp Oval (0.jpg) của cùng 1 người...")
    res_std_blink = pipeline.evaluate_blink_frame(
        frame_input=img_std,
        current_blink_counter=0,
        current_blink_state=False,
        baseline_ear=0.0,
        session_id=client_uuid
    )
    print(f" -> fit_oval: {res_std_blink.get('fit_oval')}, same_person: {res_std_blink.get('same_person')}, label: {res_std_blink.get('label')}")
    assert res_std_blink["fit_oval"] is True, "Frame chuẩn phải có fit_oval = True!"
    assert res_std_blink["same_person"] is True, "Cùng 1 người phải có same_person = True!"
    print(" [✓] 2.4 Mặt khớp Oval của cùng 1 người xác thực thành công (fit_oval=True, same_person=True)!")

    # 5. Giai đoạn 2 (Chớp mắt): Khi một người khác nhảy vào trong khung Oval
    print("\n[*] 2.5 Kiểm tra tráo đổi người khi mặt khớp Oval (Face Swap Mismatch)...")
    orig_desc = pipeline.liveness_sessions[client_uuid]["base_desc"]
    diff_desc = {
        "anchors_3d": orig_desc["anchors_3d"] + np.random.normal(0, 0.12, orig_desc["anchors_3d"].shape),
        "geo_vector": orig_desc["geo_vector"] * 1.6,
        "lab_hist": orig_desc.get("lab_hist"),
        "iod": orig_desc.get("iod", 1.0),
        "timestamp": time.time()
    }
    real_extract = pipeline.identity_verifier.extract_descriptor
    try:
        pipeline.identity_verifier.extract_descriptor = lambda *args, **kwargs: diff_desc
        res_swap = pipeline.evaluate_blink_frame(
            frame_input=img_std,
            current_blink_counter=0,
            current_blink_state=False,
            baseline_ear=0.25,
            session_id=client_uuid
        )
    finally:
        pipeline.identity_verifier.extract_descriptor = real_extract

    print(f" -> fit_oval: {res_swap.get('fit_oval')}, same_person: {res_swap.get('same_person')}, label: {res_swap.get('label')}")
    assert res_swap["fit_oval"] is True, "Mặt nằm trong oval nên fit_oval = True"
    assert res_swap["same_person"] is False, "Người khác trong oval phải báo same_person = False!"
    assert "MISMATCH" in res_swap.get("label", ""), "Label phải cảnh báo MISMATCH!"
    print(" [✓] 2.5 Phát hiện tráo đổi người chính xác khi mặt nằm trong Oval!")


def test_validate_pose_oval_fit():
    print("\n--- TEST 3: KIỂM TRA VALIDATE POSE OVAL FIT CỦA PRE-CAPTURE ---")
    pipeline = EKYCPipelineServer(lazy_load=False)
    pipeline.occlusion_detector.check_occlusion = lambda *args, **kwargs: (False, "OK", "")

    img_far = cv2.imread(os.path.join(PROJECT_ROOT, "data_raw", "2.jpg"))
    img_std = cv2.imread(os.path.join(PROJECT_ROOT, "data_raw", "0.jpg"))

    pose_far = pipeline.validate_pose(img_far)
    print(f"[*] validate_pose với ảnh ở xa (2.jpg): is_aligned_good={pose_far['is_aligned_good']}, fit_oval={pose_far['fit_oval']}, is_too_far={pose_far['is_too_far']}")
    assert pose_far["is_aligned_good"] is False, "Ảnh ở xa không được is_aligned_good!"
    assert pose_far["fit_oval"] is False, "Ảnh ở xa phải fit_oval = False!"
    assert pose_far["is_too_far"] is True, "Ảnh ở xa phải is_too_far = True!"

    pose_std = pipeline.validate_pose(img_std)
    print(f"[*] validate_pose với ảnh chuẩn (0.jpg): is_aligned_good={pose_std['is_aligned_good']}, fit_oval={pose_std['fit_oval']}, is_too_far={pose_std['is_too_far']}")
    assert pose_std["is_aligned_good"] is True, "Ảnh chuẩn phải is_aligned_good = True!"
    assert pose_std["fit_oval"] is True, "Ảnh chuẩn phải fit_oval = True!"
    assert pose_std["is_too_far"] is False, "Ảnh chuẩn không được is_too_far!"
    print(" [✓] TEST 3 THÀNH CÔNG: validate_pose chặn ảnh xa và chỉ duyệt ảnh chuẩn khớp Oval!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" BẮT ĐẦU CHUỖI KIỂM THỬ: OVAL FACE FIT & CONTINUOUS VERIFICATION")
    print("=" * 70)
    test_geometric_oval_fit()
    test_init_session_and_continuous_blink()
    test_validate_pose_oval_fit()
    print("\n" + "=" * 70)
    print(" TẤT CẢ KIỂM THỬ ĐÃ HOÀN TẤT VÀ VƯỢT QUA 100%!")
    print("=" * 70)
