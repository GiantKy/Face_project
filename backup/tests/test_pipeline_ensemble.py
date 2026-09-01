# -*- coding: utf-8 -*-
"""
=============================================================================
Full E-KYC Pipeline: Dual-Model Official Anti-Spoof Ensemble + Active Liveness
=============================================================================
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import math
import json
import csv
import glob
import argparse
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import cv2
import numpy as np
import torch

# Thiết lập đường dẫn import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.dirname(BACKUP_DIR)

for p in [BASE_DIR, BACKUP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.face_detection import FaceDetector
from src.landmark_detection import LandmarkDetector
from src.landmark_detection.draw_landmarks import draw_landmarks
from src.pose_validation import PoseValidator
from src.pose_validation.draw_pose import draw_pose_info
from src.face_alignment_crop import FaceAligner
from src.head_movement import HeadMovementDetector, HeadAction, ChallengeState
from src.anti_spoof.minifasnet_official import (
    AntiSpoofOfficialEnsemble,
    OfficialImageCropper,
    find_official_ensemble_models
)

DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output", "pipeline_ensemble")
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calc_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_eye_aspect_ratio(landmarks):
    if not landmarks or len(landmarks) < 468:
        return 0.0, 0.0, 0.0

    l_top = (calc_dist(landmarks[160], landmarks[144]) + calc_dist(landmarks[158], landmarks[153])) / 2.0
    l_width = calc_dist(landmarks[33], landmarks[133])
    ear_left = (l_top / l_width) if l_width > 0 else 0.0

    r_top = (calc_dist(landmarks[385], landmarks[380]) + calc_dist(landmarks[387], landmarks[373])) / 2.0
    r_width = calc_dist(landmarks[362], landmarks[263])
    ear_right = (r_top / r_width) if r_width > 0 else 0.0

    ear_avg = (ear_left + ear_right) / 2.0
    return ear_left, ear_right, ear_avg


def get_next_image_index(data_dir=DATA_RAW_DIR):
    existing_files = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
    indices = []
    for f in existing_files:
        stem = Path(f).stem
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices) + 1 if indices else 1


def json_serialize_helper(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return str(obj)


def draw_ui_card(image, x, y, w, h, bg_color=(15, 15, 20), alpha=0.85):
    sub_img = image[y:y+h, x:x+w]
    if sub_img.size == 0:
        return
    rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 1.0)
    image[y:y+h, x:x+w] = res
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 90, 90), 1)


def draw_pipeline_ensemble_hud(
    res_img: np.ndarray,
    img_idx: int,
    primary_face: dict,
    num_faces: int,
    pose_dict: dict,
    pose_valid: bool,
    spoof_info: dict,
    blink_passed: bool,
    blink_counter: int,
    head_movement_passed: bool,
    current_head_action: str,
    final_pass: bool,
    reasons: list
) -> np.ndarray:
    h, w = res_img.shape[:2]
    canvas = res_img.copy()

    panel_w = 420
    panel_h = min(h - 20, 520)
    px = w - panel_w - 15
    py = 10

    draw_ui_card(canvas, px, py, panel_w, panel_h, bg_color=(12, 14, 18), alpha=0.88)

    title = f"eKYC ENSEMBLE REPORT #{img_idx}"
    cv2.putText(canvas, title, (px + 15, py + 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (px + 10, py + 38), (px + panel_w - 10, py + 38), (0, 255, 127), 2)

    font = cv2.FONT_HERSHEY_DUPLEX
    f_scale = 0.42
    cy = py + 62
    step_y = 26

    c_face = (primary_face is not None)
    c_single = (num_faces == 1)
    f_col = (0, 255, 127) if (c_face and c_single) else (0, 0, 255)
    f_conf = primary_face['confidence'] * 100.0 if primary_face else 0.0
    cv2.putText(canvas, f"1. Face Detect : {num_faces} face(s) | Conf: {f_conf:.1f}%", (px + 15, cy), font, f_scale, f_col, 1, cv2.LINE_AA)
    cy += step_y

    if pose_dict:
        y_val, p_val, r_val = pose_dict['yaw'], pose_dict['pitch'], pose_dict['roll']
        p_col = (0, 255, 127) if pose_valid else (0, 165, 255)
        p_txt = f"2. Pose 3D     : Y:{y_val:+.1f}  P:{p_val:+.1f}  R:{r_val:+.1f}"
    else:
        p_col = (128, 128, 128)
        p_txt = "2. Pose 3D     : N/A"
    cv2.putText(canvas, p_txt, (px + 15, cy), font, f_scale, p_col, 1, cv2.LINE_AA)
    cy += step_y

    if spoof_info:
        as_real = spoof_info["is_real"]
        as_lbl = spoof_info["label"]
        r_score = spoof_info["real_score"] * 100.0
        cs = spoof_info.get("class_scores", {})
        p2 = cs.get("spoof_2d", 0.0) * 100.0
        p3 = cs.get("spoof_3d", 0.0) * 100.0
        m1_r = spoof_info.get("model1_scores", {}).get("real", 0.0) * 100.0
        m2_r = spoof_info.get("model2_scores", {}).get("real", 0.0) * 100.0

        as_col = (0, 255, 127) if as_real else (0, 0, 255)
        cv2.putText(canvas, f"3. Anti-Spoof  : {as_lbl} (Real: {r_score:.1f}%)", (px + 15, cy), font, f_scale, as_col, 1, cv2.LINE_AA)
        cy += step_y
        cv2.putText(canvas, f"   - 2D Paper : {p2:4.1f}%  |  3D Screen : {p3:4.1f}%", (px + 15, cy), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cy += step_y - 4
        cv2.putText(canvas, f"   - M1(2.7x) : {m1_r:4.1f}%  |  M2(4.0x)  : {m2_r:4.1f}%", (px + 15, cy), font, 0.38, (140, 200, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "3. Anti-Spoof  : NO DATA", (px + 15, cy), font, f_scale, (128, 128, 128), 1, cv2.LINE_AA)
    cy += step_y

    b_col = (0, 255, 127) if blink_passed else (0, 0, 255)
    cv2.putText(canvas, f"4. Blink Active: {'PASSED' if blink_passed else 'FAILED'} (Blinks: {blink_counter})", (px + 15, cy), font, f_scale, b_col, 1, cv2.LINE_AA)
    cy += step_y

    h_col = (0, 255, 127) if head_movement_passed else (0, 0, 255)
    cv2.putText(canvas, f"5. Head Action : {'PASSED' if head_movement_passed else 'FAILED'} ({current_head_action})", (px + 15, cy), font, f_scale, h_col, 1, cv2.LINE_AA)
    cy += step_y + 6

    cv2.line(canvas, (px + 10, cy), (px + panel_w - 10, cy), (100, 100, 100), 1)
    cy += 16

    verdict_text = "eKYC APPROVED [THÀNH CÔNG]" if final_pass else "eKYC REJECTED [TỪ CHỐI]"
    v_col = (0, 255, 127) if final_pass else (0, 0, 255)

    cv2.rectangle(canvas, (px + 15, cy), (px + panel_w - 15, cy + 40), (25, 25, 30), cv2.FILLED)
    cv2.rectangle(canvas, (px + 15, cy), (px + panel_w - 15, cy + 40), v_col, 2)
    cv2.putText(canvas, verdict_text, (px + 28, cy + 26), font, 0.52, v_col, 1, cv2.LINE_AA)
    cy += 55

    if not final_pass and reasons:
        cv2.putText(canvas, "Lý do từ chối:", (px + 15, cy), font, 0.38, (0, 165, 255), 1, cv2.LINE_AA)
        cy += 18
        for r in reasons[:3]:
            cv2.putText(canvas, f"• {r}", (px + 20, cy), font, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
            cy += 16

    return canvas


class PipelineStage(Enum):
    PREVIEW_ALIGN = 1
    RUN_AI_STATIC = 2
    LIVE_BLINK = 3
    LIVE_HEAD_MOVEMENT = 4
    FINAL_DECISION = 5
    SHOW_RESULT = 6


def run_pipeline_ensemble(
    camera_id: int = 0,
    quick_snapshot_mode: bool = False,
    skip_liveness: bool = False,
    auto_capture_default: bool = False
):
    print("\n" + "=" * 85)
    print(" 🌟 KHỞI ĐỘNG HỆ THỐNG FULL E-KYC DUAL-MODEL ENSEMBLE PIPELINE (BACKUP)")
    print("=" * 85)

    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    pose_validator = PoseValidator()
    face_aligner = FaceAligner()
    anti_spoof_ensemble = AntiSpoofOfficialEnsemble()
    head_movement_detector = HeadMovementDetector()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Không thể kết nối tới Camera #{camera_id}!")
        return

    stage = PipelineStage.PREVIEW_ALIGN
    auto_capture = auto_capture_default
    current_img_idx = get_next_image_index()
    stable_frames = 0
    REQUIRED_STABLE = 15

    captured_frame = None
    captured_img_path = None
    captured_result_dir = None

    primary_face = None
    num_faces = 0
    all_face_crops_info = []
    landmarks_static = None
    pose_dict_static = None
    pose_valid_static = False
    face_crop_static = None
    aligned_img_static = None
    spoof_info_static = None

    blink_counter = 0
    blink_state = False
    blink_passed = False
    head_movement_passed = False
    current_head_action = HeadAction.TURN_LEFT
    challenge_start_time = 0

    final_pass = False
    reasons = []

    def reset_for_next_session():
        nonlocal stage, current_img_idx, stable_frames, captured_frame
        nonlocal primary_face, num_faces, all_face_crops_info, landmarks_static
        nonlocal pose_dict_static, pose_valid_static, face_crop_static, aligned_img_static, spoof_info_static
        nonlocal blink_counter, blink_state, blink_passed, head_movement_passed
        nonlocal current_head_action, final_pass, reasons

        current_img_idx = get_next_image_index()
        stable_frames = 0
        captured_frame = None
        primary_face = None
        num_faces = 0
        all_face_crops_info = []
        landmarks_static = None
        pose_dict_static = None
        pose_valid_static = False
        face_crop_static = None
        aligned_img_static = None
        spoof_info_static = None
        blink_counter = 0
        blink_state = False
        blink_passed = False
        head_movement_passed = False
        final_pass = False
        reasons = []
        stage = PipelineStage.PREVIEW_ALIGN
        print(f"\n[INFO] Đã sẵn sàng cho phiên eKYC mới: ID #{current_img_idx}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        display = frame.copy()

        if stage == PipelineStage.PREVIEW_ALIGN:
            faces = face_detector.detect(frame)
            num_faces = len(faces)
            landmarks = landmark_detector.detect(frame)

            pose_res = pose_validator.validate_pose(frame, landmarks) if landmarks else None
            is_pose_ok = pose_res["is_valid"] if pose_res else False

            guide_color = (0, 255, 127) if (num_faces == 1 and is_pose_ok) else (100, 100, 255)
            cv2.ellipse(display, (w // 2, h // 2), (w // 6, h // 4), 0, 0, 360, guide_color, 2)

            for f in faces:
                bx1, by1, bx2, by2 = f["bbox"]
                cv2.rectangle(display, (bx1, by1), (bx2, by2), guide_color, 2)

            if landmarks:
                display = draw_landmarks(display, landmarks)

            if auto_capture and (num_faces == 1) and is_pose_ok:
                stable_frames += 1
                if stable_frames >= REQUIRED_STABLE:
                    captured_frame = frame.copy()
                    stage = PipelineStage.RUN_AI_STATIC
            else:
                stable_frames = 0

            draw_ui_card(display, 10, 10, w - 20, 50, bg_color=(15, 15, 20), alpha=0.8)
            info_txt = f"[{current_img_idx}] Giu mat thang huong vao khung elip | Faces: {num_faces} | Auto: {'ON' if auto_capture else 'OFF'}"
            cv2.putText(display, info_txt, (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        elif stage == PipelineStage.RUN_AI_STATIC:
            captured_result_dir = os.path.join(OUTPUT_DIR, str(current_img_idx))
            os.makedirs(captured_result_dir, exist_ok=True)
            captured_img_path = os.path.join(DATA_RAW_DIR, f"{current_img_idx}.jpg")
            cv2.imwrite(captured_img_path, captured_frame)

            faces = face_detector.detect(captured_frame)
            num_faces = len(faces)
            primary_face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])) if faces else None

            landmarks_static = landmark_detector.detect(captured_frame)

            if landmarks_static:
                p_res = pose_validator.validate_pose(captured_frame, landmarks_static)
                pose_dict_static = p_res.get("pose", None)
                pose_valid_static = p_res.get("is_valid", False)

            if primary_face:
                aligned_img_static = face_aligner.align_face(captured_frame, landmarks_static, bbox=primary_face["bbox"])
                face_crop_static = face_aligner.crop_face(captured_frame, landmarks_static, bbox=primary_face["bbox"], output_size=(224, 224))

            if primary_face:
                spoof_info_static = anti_spoof_ensemble.predict_face(captured_frame, primary_face["bbox"])

            if spoof_info_static:
                c27 = spoof_info_static.get("crop_27", None)
                c40 = spoof_info_static.get("crop_40", None)
                if c27 is not None:
                    cv2.imwrite(os.path.join(captured_result_dir, "4_crop_2.7x_minifasnet.jpg"), c27)
                if c40 is not None:
                    cv2.imwrite(os.path.join(captured_result_dir, "5_crop_4.0x_minifasnet.jpg"), c40)

            if quick_snapshot_mode or skip_liveness:
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION
            else:
                stage = PipelineStage.LIVE_BLINK
                blink_counter = 0
                blink_state = False
                blink_passed = False

        elif stage == PipelineStage.LIVE_BLINK:
            landmarks_live = landmark_detector.detect(frame)
            _, _, ear_avg = compute_eye_aspect_ratio(landmarks_live) if landmarks_live else (0.0, 0.0, 0.0)

            if landmarks_live:
                display = draw_landmarks(display, landmarks_live)

            if 0.05 < ear_avg < 0.18:
                if not blink_state:
                    blink_state = True
            elif ear_avg >= 0.22:
                if blink_state:
                    blink_counter += 1
                    blink_state = False

            if blink_counter >= 1:
                blink_passed = True
                stage = PipelineStage.LIVE_HEAD_MOVEMENT
                current_head_action = head_movement_detector.start_challenge()

            draw_ui_card(display, 20, 20, w - 40, 75, bg_color=(15, 25, 45), alpha=0.9)
            cv2.putText(display, "BƯỚC 1/2: ACTIVE LIVENESS - XÁC THỰC CHỚP MẮT", (35, 48), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display, f"Vui long chop mat tu nhien | EAR: {ear_avg:.2f} | Da chop: {blink_counter}/1", (35, 78), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        elif stage == PipelineStage.LIVE_HEAD_MOVEMENT:
            landmarks_live = landmark_detector.detect(frame)
            if landmarks_live:
                display = draw_landmarks(display, landmarks_live)
                p_info = pose_validator.validate_pose(frame, landmarks_live)
                c_res = head_movement_detector.update(p_info.get("pose", {}))

                if c_res.get("passed", False):
                    head_movement_passed = True
                    stage = PipelineStage.FINAL_DECISION

            draw_ui_card(display, 20, 20, w - 40, 75, bg_color=(15, 45, 25), alpha=0.9)
            cv2.putText(display, f"BƯỚC 2/2: ACTIVE LIVENESS - THỬ THÁCH QUAY ĐẦU", (35, 48), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 127), 1, cv2.LINE_AA)
            action_name = current_head_action.value if hasattr(current_head_action, 'value') else str(current_head_action)
            cv2.putText(display, f"YEU CAU: {action_name.upper()}", (35, 78), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        elif stage == PipelineStage.FINAL_DECISION:
            reasons = []
            c_face = (primary_face is not None)
            c_single = (num_faces == 1)
            c_pose = bool(pose_valid_static)
            c_spoof = bool(spoof_info_static["is_real"]) if spoof_info_static else False
            c_blink = bool(blink_passed)
            c_head = bool(head_movement_passed)

            if not c_face:
                reasons.append("Không phát hiện khuôn mặt trong ảnh")
            if not c_single:
                reasons.append(f"Phát hiện {num_faces} khuôn mặt (Yêu cầu duy nhất 1 người)")
            if not c_pose:
                reasons.append("Góc quay mặt chưa đạt chuẩn nhìn thẳng")
            if not c_spoof:
                reasons.append("Phát hiện giả mạo qua Dual-Model Anti-Spoofing Ensemble")
            if not c_blink:
                reasons.append("Chưa hoàn thành xác thực chớp mắt")
            if not c_head:
                reasons.append("Chưa hoàn thành thử thách cử động đầu")

            final_pass = (c_face and c_single and c_pose and c_spoof and c_blink and c_head)

            res_img = captured_frame.copy()
            if primary_face:
                bx1, by1, bx2, by2 = primary_face["bbox"]
                b_color = (0, 255, 127) if final_pass else (0, 0, 255)
                cv2.rectangle(res_img, (bx1, by1), (bx2, by2), b_color, 2)

            if landmarks_static:
                res_img = draw_landmarks(res_img, landmarks_static)

            final_display_img = draw_pipeline_ensemble_hud(
                res_img,
                current_img_idx,
                primary_face,
                num_faces,
                pose_dict_static,
                pose_valid_static,
                spoof_info_static,
                blink_passed,
                blink_counter,
                head_movement_passed,
                current_head_action.value if hasattr(current_head_action, 'value') else str(current_head_action),
                final_pass,
                reasons
            )

            out_res_p = os.path.join(captured_result_dir, "1_pipeline_result.jpg")
            cv2.imwrite(out_res_p, final_display_img)

            if face_crop_static is not None:
                cv2.imwrite(os.path.join(captured_result_dir, "2_face_crop_224.jpg"), face_crop_static)

            if aligned_img_static is not None:
                cv2.imwrite(os.path.join(captured_result_dir, "3_aligned_full.jpg"), aligned_img_static)

            report_data = {
                "session_id": current_img_idx,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_image": captured_img_path,
                "face_detection": {
                    "face_detected": c_face,
                    "num_faces": num_faces,
                    "single_person_passed": c_single,
                    "bbox": primary_face["bbox"] if primary_face else None,
                    "confidence": float(primary_face["confidence"]) if primary_face else 0.0
                },
                "pose_validation": {
                    "is_valid": c_pose,
                    "angles": pose_dict_static
                },
                "anti_spoof_ensemble": {
                    "is_real": c_spoof,
                    "label": spoof_info_static["label"] if spoof_info_static else "NONE",
                    "confidence": spoof_info_static["confidence"] if spoof_info_static else 0.0,
                    "real_score": spoof_info_static["real_score"] if spoof_info_static else 0.0,
                    "class_scores": spoof_info_static.get("class_scores", {}) if spoof_info_static else {},
                    "model1_2.7x": spoof_info_static.get("model1_scores", {}) if spoof_info_static else {},
                    "model2_4.0x": spoof_info_static.get("model2_scores", {}) if spoof_info_static else {},
                },
                "active_liveness": {
                    "blink_passed": c_blink,
                    "blink_count": blink_counter,
                    "head_movement_passed": c_head,
                    "head_action": current_head_action.value if hasattr(current_head_action, 'value') else str(current_head_action),
                },
                "final_verdict": "APPROVED" if final_pass else "REJECTED",
                "reasons": reasons
            }

            out_json_p = os.path.join(captured_result_dir, "6_report.json")
            with open(out_json_p, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2, default=json_serialize_helper)

            summary_csv = os.path.join(OUTPUT_DIR, "batch_summary.csv")
            file_exists = os.path.exists(summary_csv)
            with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["ID", "Timestamp", "Verdict", "Num_Faces", "Pose_Valid", "AntiSpoof_Label", "Real_Pct", "2D_Spoof_Pct", "3D_Spoof_Pct", "Blink", "Head_Movement", "Reasons"])
                writer.writerow([
                    current_img_idx,
                    report_data["timestamp"],
                    report_data["final_verdict"],
                    num_faces,
                    c_pose,
                    spoof_info_static["label"] if spoof_info_static else "NONE",
                    f"{spoof_info_static['real_score']*100:.1f}%" if spoof_info_static else "0.0%",
                    f"{spoof_info_static['class_scores']['spoof_2d']*100:.1f}%" if spoof_info_static else "0.0%",
                    f"{spoof_info_static['class_scores']['spoof_3d']*100:.1f}%" if spoof_info_static else "0.0%",
                    c_blink,
                    c_head,
                    "; ".join(reasons)
                ])

            stage = PipelineStage.SHOW_RESULT

        elif stage == PipelineStage.SHOW_RESULT:
            display = final_display_img.copy()

        cv2.imshow("Full E-KYC Dual-Model Ensemble Pipeline", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('c'), ord('C'), 32):
            if stage == PipelineStage.PREVIEW_ALIGN:
                captured_frame = frame.copy()
                stage = PipelineStage.RUN_AI_STATIC
        elif key in (ord('a'), ord('A')):
            auto_capture = not auto_capture
        elif key in (ord('s'), ord('S')):
            if stage in (PipelineStage.LIVE_BLINK, PipelineStage.LIVE_HEAD_MOVEMENT):
                blink_passed = True
                head_movement_passed = True
                stage = PipelineStage.FINAL_DECISION
        elif key in (ord('r'), ord('R')):
            reset_for_next_session()

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Full eKYC Dual-Model Ensemble Pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số Camera Webcam (mặc định 0)")
    parser.add_argument("--quick", action="store_true", help="Chế độ Quick Snapshot (bỏ qua Active Liveness)")
    parser.add_argument("--skip-liveness", action="store_true", help="Bỏ qua bước chớp mắt và quay đầu")
    parser.add_argument("--auto", action="store_true", help="Bật Auto Capture khi mặt chuẩn")
    args = parser.parse_args()

    run_pipeline_ensemble(
        camera_id=args.camera,
        quick_snapshot_mode=args.quick,
        skip_liveness=args.skip_liveness,
        auto_capture_default=args.auto
    )


if __name__ == "__main__":
    main()
