"""
Pipeline E-KYC test (trước bước Anti-Spoof)
Quy trình:
  1. Face Detection (YOLO)
  2. Landmark Detection (mediapipe)
  3. Pose Validation (head pose 3D)
  4. Face Alignment + Crop
"""

import sys
import os

# Trỏ import vào backup/src
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import cv2

from src.face_detection import FaceDetector
from src.landmark_detection import LandmarkDetector
from src.landmark_detection.draw_landmarks import draw_landmarks
from src.landmark_detection.utils import get_landmark_point
from src.pose_validation import PoseValidator
from src.pose_validation.draw_pose import draw_pose_info
from src.face_alignment_crop import FaceAligner

# =========================
# INIT
# =========================
print("[INFO] Initializing modules...")

detector = FaceDetector()
print("[OK] FaceDetector loaded")

landmark_detector = LandmarkDetector()
print("[OK] LandmarkDetector loaded")

pose_validator = PoseValidator()
print("[OK] PoseValidator loaded")

aligner = FaceAligner()
print("[OK] FaceAligner loaded")

# =========================
# LOAD IMAGE
# =========================
# Go up: tests -> Face-Project (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(BASE_DIR, "data_raw", "0.jpg")

print(f"[INFO] Loading image: {IMG_PATH}")
image = cv2.imread(IMG_PATH)

if image is None:
    print(f"[ERROR] Cannot load image: {IMG_PATH}")
    sys.exit(1)

print(f"[INFO] Image size: {image.shape[1]}x{image.shape[0]}")

# =========================
# STEP 1: FACE DETECTION
# =========================
print("\n========== STEP 1: Face Detection ==========")

faces = detector.detect(image)

print(f"[RESULT] Detected {len(faces)} face(s)")

if len(faces) == 0:
    print("[ERROR] No face detected! Exiting.")
    sys.exit(1)

# Vẽ bounding box lên bản copy
display = image.copy()

for i, face in enumerate(faces):
    x1, y1, x2, y2 = face["bbox"]
    conf = face["confidence"]

    print(f"  Face {i+1}: bbox=({x1},{y1},{x2},{y2}), conf={conf:.3f}")

    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        display,
        f"Face {conf:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# =========================
# STEP 2: LANDMARK DETECTION
# =========================
print("\n========== STEP 2: Landmark Detection ==========")

landmarks = landmark_detector.detect(image)

print(f"[RESULT] Detected {len(landmarks)} landmarks")

if len(landmarks) == 0:
    print("[ERROR] No landmarks detected! Exiting.")
    sys.exit(1)

# Vẽ landmarks lên display
display = draw_landmarks(display, landmarks)

# =========================
# STEP 3: POSE VALIDATION
# =========================
print("\n========== STEP 3: Pose Validation ==========")

valid, text, pose = pose_validator.validate(
    landmarks,
    get_landmark_point
)

if pose:
    print(f"  Yaw:   {pose['yaw']:.2f}")
    print(f"  Pitch: {pose['pitch']:.2f}")
    print(f"  Roll:  {pose['roll']:.2f}")

print(f"[RESULT] {text} ({'PASS' if valid else 'FAIL'})")

display = draw_pose_info(display, pose, text, valid)

# =========================
# STEP 4: FACE ALIGNMENT + CROP
# =========================
print("\n========== STEP 4: Face Alignment + Crop ==========")

aligned = aligner.align_face(image, landmarks)
print("[OK] Face aligned")

aligned_landmarks = aligner.get_landmarks(aligned)

if aligned_landmarks:
    face_crop = aligner.crop_face(aligned, aligned_landmarks)
    print(f"[OK] Face cropped: {face_crop.shape[1]}x{face_crop.shape[0]}")
else:
    face_crop = None
    print("[WARN] No landmarks on aligned face, skipping crop")

# =========================
# SAVE RESULTS
# =========================
print("\n========== SAVE RESULTS ==========")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save pipeline result
out_path_1 = os.path.join(OUTPUT_DIR, "step1234_pipeline.jpg")
cv2.imwrite(out_path_1, display)
print(f"[SAVED] {out_path_1}")

if face_crop is not None:
    out_path_2 = os.path.join(OUTPUT_DIR, "step4_face_crop.jpg")
    cv2.imwrite(out_path_2, face_crop)
    print(f"[SAVED] {out_path_2}")

print("\n[INFO] Pipeline done (before Anti-Spoof step).")

