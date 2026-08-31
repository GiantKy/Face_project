# Backup - Face Project (Fixed)

Thư mục này chứa các file đã sửa lỗi, hoạt động độc lập với thư mục gốc.  
File gốc **không bị thay đổi**.

---

## Lỗi đã sửa

| # | File gốc | Lỗi | Nguyên nhân | Cách sửa |
|---|----------|------|-------------|----------|
| 1 | `src/face_alignment_crop/face_align_crop.py` | `AttributeError: module 'mediapipe' has no attribute 'solutions'` | mediapipe 0.10.14+ xóa API `mp.solutions` | Chuyển sang `mp.tasks.vision.FaceLandmarker` |
| 2 | `src/landmark_detection/landmark_detector.py` | `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Cùng lỗi trên | Chuyển sang `mp.tasks.vision.FaceLandmarker` |
| 3 | `tests/test_anti_spoof.py` | `UnpicklingError` + `TypeError: load_state_dict` | File `Anti_Spoof.pt` là YOLO model, code cũ dùng EfficientNet | Dùng `ultralytics.YOLO()` để load trực tiếp |

---

## Cấu trúc thư mục

```
backup/
├── README.md
├── src/
│   ├── face_alignment_crop/
│   │   ├── __init__.py
│   │   └── face_align_crop.py       ← FIXED (mp.tasks API)
│   ├── face_detection/               ← Copy từ gốc (path đã sửa)
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── face_crop.py
│   ├── landmark_detection/
│   │   ├── __init__.py
│   │   ├── landmark_detector.py      ← FIXED (mp.tasks API)
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── draw_landmarks.py
│   │   └── utils.py
│   ├── pose_validation/              ← Copy từ gốc (không lỗi)
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── draw_pose.py
│   │   ├── head_pose_3d.py
│   │   ├── utils.py
│   │   └── validator.py
│    ├── head_movement/                ← MỚI (Active Head Movement Liveness)
│   │   ├── __init__.py
│   │   └── head_movement_detector.py
│   └── anti_spoof/                   ← MỚI (MiniFASNetV2 Anti-Spoofing Architecture)
│       ├── __init__.py
│       └── minifasnet.py
└── tests/
    ├── test_anti_spoof.py            ← YOLO Anti-Spoof (v7.pt)
    ├── test_anti_spoof_minifasnet.py ← MỚI (Test MiniFASNetV2: Webcam, Image, Directory)
    ├── test_anti_spoof_minisfasr.py  ← MỚI (Alias cho test_anti_spoof_minifasnet.py)
    ├── test_face_alignment_crop.py
    ├── test_face_detection.py
    ├── test_landmark_detection.py
    ├── test_pose_validation.py
    ├── test_head_movement.py         ← MỚI (Test thử thách cử động đầu)
    ├── test_pipeline.py              ← Pipeline 1 (Face + Landmark + Pose + Align/Crop)
    ├── test_pipeline_2.py            ← Pipeline 2 (Full E-KYC: Anti-Spoof, Liveness EAR/MAR)
    ├── test_pipeline_3.py            ← Pipeline 3 (Batch / Full Interactive Liveness)
    ├── test_pipeline_4.py            ← PIPELINE 4 HOÀN CHỈNH (Mở Webcam -> Chụp ảnh -> AI Model -> Blink & Head Movement -> Output)
    └── output/                       ← Ảnh và báo cáo JSON/CSV kết quả

---

## Cách chạy

```bash
cd backup/tests

# Test từng module
python test_face_detection.py
python test_landmark_detection.py
python test_face_alignment_crop.py
python test_pose_validation.py
python test_head_movement.py          # Test thử thách cử động đầu trên Webcam
python test_anti_spoof.py             # Test Anti-Spoof YOLO

# Test Anti-Spoof MiniFASNetV2
python test_anti_spoof_minifasnet.py                    # Chế độ Live Webcam
python test_anti_spoof_minifasnet.py --dir ../../data_raw  # Test toàn bộ ảnh trong data_raw
python test_anti_spoof_minifasnet.py --image ../../data_raw/0.jpg # Test 1 ảnh cụ thể
# Hoặc dùng alias:
python test_anti_spoof_minisfasr.py

# Test Pipeline 1, 2, 3
python test_pipeline.py
python test_pipeline_2.py
python test_pipeline_3.py

# Test Pipeline 4 (Quy trình hoàn chỉnh: Mở Webcam -> Chụp ảnh lưu vào data_raw/ -> Chạy AI Model -> Live Blink & Head Movement -> Xuất kết quả output/<id>/):
python test_pipeline_4.py
```

---

## Yêu cầu

- Python 3.11
- mediapipe >= 0.10.14 (đã test với 0.10.33)
- ultralytics (YOLO)
- torch, torchvision
- opencv-python

---

## Model files cần có

Các model nằm ở thư mục gốc `Face-Project/models/`:

| Model | File | Dùng bởi |
|-------|------|----------|
| Face Detection | `Face_Detection.pt` | `face_detection/detector.py` |
| Face Landmark | `face_landmarker.task` | `face_alignment_crop/`, `landmark_detection/` |
| Anti-Spoof YOLO | `Anti_Spoof_v7.pt` | `tests/test_anti_spoof.py` |
| Anti-Spoof MiniFASNet | `Anti_Spoof_minifasnetv2.pth` | `src/anti_spoof/`, `tests/test_anti_spoof_minifasnet.py` |

> **Lưu ý:** File `face_landmarker.task` được tải từ Google MediaPipe.  
> Nếu thiếu, chạy:
> ```powershell
> Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "models/face_landmarker.task"
> ```

