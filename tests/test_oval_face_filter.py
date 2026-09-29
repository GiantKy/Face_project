"""
Unit and Integration Tests for Oval Face Filtering in E-KYC System.
Verifies that:
1. When 1 face is in the oval and 1+ faces are outside the oval:
   - System only accepts the face in the oval and ignores outside faces.
   - num_faces == 1, is_single == True, no MULTI_FACES warnings.
2. When 2 faces are both inside the oval:
   - num_faces >= 2, MULTI_FACES warning is correctly triggered.
3. When 0 faces are inside the oval (face is outside):
   - num_faces == 1, face_in_oval == False, guiding user to move into oval.
"""

import os
import sys
import json
import numpy as np
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server_module.utils import (
    load_image,
    image_to_base64,
    get_default_oval_params,
    is_face_in_oval,
    is_landmarks_in_oval,
    filter_faces_in_oval
)
from server_module.pipeline_server import EKYCPipelineServer
from server_module.esp32_challenge import ESP32ChallengeManager


def test_geometric_oval_filtering():
    print("\n" + "=" * 60)
    print(" [TEST 1] Geometric Unit Tests for Oval Filtering")
    print("=" * 60)

    w, h = 640, 480
    oval_center, oval_axes = get_default_oval_params(w, h)
    cx, cy = oval_center
    ax, ay = oval_axes
    print(f"Default oval params for 640x480: center={oval_center}, axes={oval_axes}")

    # Face inside oval (center near (cx, cy))
    face_in = {
        "bbox": [cx - 60, cy - 70, cx + 60, cy + 70],
        "confidence": 0.95,
        "name": "face_inside_oval"
    }

    # Face outside oval (top-left corner)
    face_out_1 = {
        "bbox": [20, 20, 120, 140],
        "confidence": 0.90,
        "name": "face_outside_top_left"
    }

    # Face outside oval (top-right corner)
    face_out_2 = {
        "bbox": [500, 20, 600, 140],
        "confidence": 0.88,
        "name": "face_outside_top_right"
    }

    # Second face also inside oval
    face_in_2 = {
        "bbox": [cx - 30, cy - 40, cx + 50, cy + 60],
        "confidence": 0.92,
        "name": "face_2_inside_oval"
    }

    # Test is_face_in_oval
    assert is_face_in_oval(face_in["bbox"], oval_center, oval_axes), "face_in should be inside oval!"
    assert not is_face_in_oval(face_out_1["bbox"], oval_center, oval_axes), "face_out_1 should be outside oval!"
    assert not is_face_in_oval(face_out_2["bbox"], oval_center, oval_axes), "face_out_2 should be outside oval!"
    print(" -> is_face_in_oval checks passed successfully.")

    # Test filter_faces_in_oval with 1 inside, 2 outside
    all_faces = [face_in, face_out_1, face_out_2]
    filtered = filter_faces_in_oval(all_faces, oval_center, oval_axes)
    assert len(filtered) == 1, f"Expected 1 face inside oval, got {len(filtered)}"
    assert filtered[0]["name"] == "face_inside_oval", "Filtered face must be face_inside_oval"
    print(" -> filter_faces_in_oval (1 in, 2 out) -> correctly filtered to 1 face inside oval.")

    # Test filter_faces_in_oval with 2 inside
    two_in = [face_in, face_in_2, face_out_1]
    filtered_two = filter_faces_in_oval(two_in, oval_center, oval_axes)
    assert len(filtered_two) == 2, f"Expected 2 faces inside oval, got {len(filtered_two)}"
    print(" -> filter_faces_in_oval (2 in, 1 out) -> correctly retained 2 faces inside oval.")

    # Test filter_faces_in_oval with 0 inside (2 outside)
    none_in = [face_out_1, face_out_2]
    filtered_none = filter_faces_in_oval(none_in, oval_center, oval_axes)
    assert len(filtered_none) == 2, "Expected to retain all faces when 0 faces in oval (so prompt works)"
    print(" -> filter_faces_in_oval (0 in, 2 out) -> correctly kept original faces for guidance prompt.")

    # Test is_landmarks_in_oval
    landmarks_inside = [(cx - 20, cy - 20), (cx + 20, cy + 20), (cx, cy)]
    landmarks_outside = [(40, 40), (80, 80), (60, 60)]
    assert is_landmarks_in_oval(landmarks_inside, w, h, oval_center, oval_axes), "landmarks_inside should be in oval"
    assert not is_landmarks_in_oval(landmarks_outside, w, h, oval_center, oval_axes), "landmarks_outside should not be in oval"
    print(" -> is_landmarks_in_oval checks passed successfully.")


def test_server_with_composite_image():
    print("\n" + "=" * 60)
    print(" [TEST 2] End-to-End Server & Pipeline Tests with 2 Faces (1 In, 1 Out)")
    print("=" * 60)

    # Initialize server
    server = EKYCPipelineServer()

    # Load canonical centered face image (0.jpg)
    test_img_path = os.path.join(PROJECT_ROOT, "data_raw", "0.jpg")
    assert os.path.exists(test_img_path), f"Test image not found at {test_img_path}"

    base_img = load_image(test_img_path)
    h, w = base_img.shape[:2]
    initial_faces = server.detector.detect(base_img)
    assert len(initial_faces) >= 1, "Base image must have at least 1 face detected"
    print(f"Loaded base image {test_img_path}: shape=({h}, {w}), detected {len(initial_faces)} face(s)")

    # Create composite image with TWO faces:
    # 1. Face in center oval (original base_img)
    # 2. Paste a second face outside oval (top-left corner x=15, y=15)
    two_face_img = base_img.copy()

    second_img_path = os.path.join(PROJECT_ROOT, "data_raw", "2.jpg")
    sec_img = load_image(second_img_path) if os.path.exists(second_img_path) else None
    sec_faces = server.detector.detect(sec_img) if sec_img is not None else []

    if sec_faces:
        sbx = sec_faces[0]["bbox"]
        src_img = sec_img
    else:
        sbx = initial_faces[0]["bbox"]
        src_img = base_img

    pad = 35
    sy1, sy2 = max(0, sbx[1] - pad), min(src_img.shape[0], sbx[3] + pad)
    sx1, sx2 = max(0, sbx[0] - pad), min(src_img.shape[1], sbx[2] + pad)
    face_crop = src_img[sy1:sy2, sx1:sx2].copy()
    fc_h, fc_w = face_crop.shape[:2]

    target_fc_h = 150
    target_fc_w = int(fc_w * (target_fc_h / fc_h))
    resized_crop = cv2.resize(face_crop, (target_fc_w, target_fc_h))

    # Paste at top-left outside oval (x=15, y=15)
    two_face_img[15:15+target_fc_h, 15:15+target_fc_w] = resized_crop

    # Verify detector detects >= 2 faces on two_face_img
    detected_faces = server.detector.detect(two_face_img)
    print(f"Detections on composite image without oval filter: {len(detected_faces)} faces")
    assert len(detected_faces) >= 2, f"Expected >= 2 faces detected on composite image, got {len(detected_faces)}"

    oval_center, oval_axes = get_default_oval_params(w, h)

    # 1. Test identity_verifier.count_faces with oval filter
    num_faces_filtered, is_single_filtered = server.identity_verifier.count_faces(
        two_face_img, server.detector,
        oval_center=oval_center, oval_axes=oval_axes, filter_oval=True
    )
    print(f"identity_verifier.count_faces (filter_oval=True) -> num_faces={num_faces_filtered}, is_single={is_single_filtered}")
    assert num_faces_filtered == 1, f"Expected num_faces == 1 with oval filter, got {num_faces_filtered}"
    assert is_single_filtered is True, "Expected is_single == True with oval filter"

    # Verify without oval filter it detects >= 2
    num_faces_unfiltered, is_single_unfiltered = server.identity_verifier.count_faces(
        two_face_img, server.detector, filter_oval=False
    )
    print(f"identity_verifier.count_faces (filter_oval=False) -> num_faces={num_faces_unfiltered}, is_single={is_single_unfiltered}")
    assert num_faces_unfiltered >= 2, "Expected >= 2 faces without oval filter"
    assert is_single_unfiltered is False, "Expected is_single == False without oval filter"

    # 2. Test validate_pose on composite image
    two_face_b64 = image_to_base64(two_face_img)
    pose_res = server.validate_pose(two_face_b64)
    print(f"validate_pose result: num_faces={pose_res.get('num_faces')}, face_in_oval={pose_res.get('face_in_oval')}, warning={pose_res.get('warning')}")
    assert pose_res.get("num_faces") == 1, f"validate_pose must report num_faces=1, got {pose_res.get('num_faces')}"
    assert pose_res.get("face_in_oval") is True, "validate_pose must report face_in_oval=True"
    assert pose_res.get("warning") != "MULTI_FACES", f"validate_pose must NOT report MULTI_FACES warning! Got: {pose_res.get('warning')}"

    # 3. Test init_liveness_session on composite image (must NOT fail due to MULTI_FACES)
    session_res = server.init_liveness_session(two_face_b64)
    print(f"init_liveness_session result: error={session_res.get('error')}, reason={session_res.get('occlusion_reason')}")
    assert session_res.get("error") != "MULTI_FACES", f"init_liveness_session must NOT fail with MULTI_FACES! Got: {session_res.get('error')}"

    # 4. Test check_antispoof on composite image (must NOT fail due to MULTI_FACES)
    spoof_res = server.check_antispoof(two_face_img)
    print(f"check_antispoof result: success={spoof_res.get('success')}, error={spoof_res.get('error')}")
    assert spoof_res.get("error") != "MULTI_FACES", "check_antispoof should not trigger MULTI_FACES"

    print("\n" + "=" * 60)
    print(" [TEST 3] ESP32 Challenge Manager with 2 Faces (1 In, 1 Out)")
    print("=" * 60)

    # 5. Test ESP32 Challenge Manager (must NOT report MULTI_FACES)
    esp32_mgr = ESP32ChallengeManager()
    esp32_res = esp32_mgr.start_challenge(two_face_img, server)
    print(f"esp32 start_challenge verdict: {esp32_res.get('verdict')}, num_faces={esp32_res.get('num_faces', 1)}")
    assert esp32_res.get("verdict") != "MULTI_FACES", f"ESP32 start_challenge must NOT report MULTI_FACES! Got: {esp32_res.get('verdict')}"

    print("\n" + "=" * 60)
    print(" [TEST 4] Edge Case: 0 Faces Inside Oval (1 Face Outside)")
    print("=" * 60)

    # Create image with ONLY the corner face, center blanked out
    corner_only_img = np.zeros_like(base_img)
    corner_only_img[15:15+target_fc_h, 15:15+target_fc_w] = resized_crop
    corner_b64 = image_to_base64(corner_only_img)

    corner_pose_res = server.validate_pose(corner_b64)
    print(f"validate_pose (face outside oval): num_faces={corner_pose_res.get('num_faces')}, face_in_oval={corner_pose_res.get('face_in_oval')}, guide={corner_pose_res.get('guide')}")
    # When face is outside oval, face_in_oval must be False, guiding user
    assert corner_pose_res.get("face_in_oval") is False, "face_in_oval should be False when face is outside oval"
    assert "OVAL" in (corner_pose_res.get("message", "") + corner_pose_res.get("guide", "")).upper(), "Message or guide should prompt user to move into oval"

    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)


if __name__ == "__main__":
    test_geometric_oval_filtering()
    test_server_with_composite_image()
