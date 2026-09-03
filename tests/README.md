# 🧪 Face-Project Tests Directory

Tài liệu chi tiết hướng dẫn chạy và ghi chú tất cả các bài test đã được tổng hợp tại:
👉 **[TEST_NOTES.md](../TEST_NOTES.md)**

### Danh sách các bài test nhanh:
1. `test_face_detection.py` - Kiểm tra phát hiện khuôn mặt (YOLO Face)
2. `test_landmark_detection.py` - Kiểm tra trích xuất MediaPipe Landmarks
3. `test_pose_validation.py` - Kiểm tra tính góc xoay đầu (Yaw, Pitch, Roll)
4. `test_face_alignment_crop.py` - Kiểm tra căn thẳng mặt & crop kích thước chuẩn
5. `test_anti_spoof.py` - Kiểm tra chống giả mạo trực tiếp với YOLO
6. `test_anti_spoof_minifasnet.py` - Kiểm tra Anti-Spoofing với MiniFASNetV2
7. `test_anti_spoof_mobilenetv2.py` - Kiểm tra Anti-Spoofing với MobileNetV2 (Hugging Face / Safetensors 224x224)
8. `test_anti_spoof_official_ensemble.py` - Kiểm tra Ensemble đa mô hình MiniFASNet
9. `test_head_movement.py` - Thử thách cử động đầu Active Liveness
10. `test_pipeline_full.py` - Toàn bộ chu trình eKYC tương tác & xuất báo cáo

