# -*- coding: utf-8 -*-
"""
=============================================================================
Công cụ Tải File Trọng Số (Weights) từ Roboflow về Thư Mục models/ của Dự Án
=============================================================================
Sử dụng thư viện `inference.get_model` để tải toàn bộ weights.onnx, nhãn classes,
và cấu hình metadata về thư mục `models/roboflow/<model_id>/`, đồng thời tự động
sao lưu trực tiếp ra thư mục `models/` của project để sử dụng offline 100% không
cần kết nối mạng hay tốn credit Roboflow Serverless.

Mô hình mặc định:
  - face-spoof-detection-liika-qopyy/1 -> models/anti_spoof_roboflow.onnx
  - anti-spoof-qhqvq-yhn2n/1           -> models/anti_spoof_roboflow_v2.onnx

Cách dùng:
  # 1. Tải và đồng bộ toàn bộ model mặc định:
  py -3.11 tools/download_roboflow_weights.py

  # 2. Tải mô hình cụ thể bằng Model ID:
  py -3.11 tools/download_roboflow_weights.py --model-id anti-spoof-qhqvq-yhn2n/1
  py -3.11 tools/download_roboflow_weights.py --model-id face-spoof-detection-liika-qopyy/1
=============================================================================
"""

import sys
import os
import shutil
import argparse
import warnings

os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
ROBOFLOW_CACHE_DIR = os.path.join(MODELS_DIR, "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR
os.makedirs(ROBOFLOW_CACHE_DIR, exist_ok=True)

DEFAULT_API_KEY = "LiYT7osRW01duX3ao91S"

# Ánh xạ model ID -> tên file weights trực tiếp trong thư mục models/
MODEL_SHORT_ALIASES = {
    "face-spoof-detection-liika-qopyy/1": "anti_spoof_roboflow.onnx",
    "anti-spoof-qhqvq-yhn2n/1": "anti_spoof_roboflow_v2.onnx",
}


def download_and_sync_model(model_id: str, api_key: str = DEFAULT_API_KEY):
    from inference import get_model

    print(f"\n[INFO] Đang tải mô hình: {model_id}...")
    target_dir = os.path.join(ROBOFLOW_CACHE_DIR, *model_id.split("/"))
    print(f"[INFO] Thư mục cache: {target_dir}")

    try:
        model = get_model(model_id=model_id, api_key=api_key)
        print(f"[OK] Đã nạp thành công mô hình: {model_id}")

        weights_file = os.path.join(target_dir, "weights.onnx")
        if os.path.exists(weights_file):
            size_mb = os.path.getsize(weights_file) / (1024 * 1024)
            print(f"[OK] File trọng số: {weights_file} ({size_mb:.2f} MB)")

            # Tự động đồng bộ ra models/
            alias_name = MODEL_SHORT_ALIASES.get(model_id)
            if not alias_name:
                clean_name = model_id.replace("/", "_").replace("-", "_") + ".onnx"
                alias_name = clean_name

            dst_path = os.path.join(MODELS_DIR, alias_name)
            shutil.copy2(weights_file, dst_path)
            print(f"[OK] Đã đồng bộ file trọng số vào thư mục models/: {dst_path}")

        return model
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải mô hình {model_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Tải trọng số mô hình từ Roboflow về thư mục models/ của dự án")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Model ID cần tải (VD: face-spoof-detection-liika-qopyy/1). Nếu để trống sẽ tải các model mặc định.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY,
                        help="Roboflow API Key")
    args = parser.parse_args()

    print("=" * 75)
    print("      CÔNG CỤ TẢI TRỌNG SỐ (WEIGHTS) ROBOFLOW VỀ MÁY CỤC BỘ")
    print("=" * 75)
    print(f"Thư mục models của dự án : {MODELS_DIR}")
    print(f"Thư mục cache Roboflow  : {ROBOFLOW_CACHE_DIR}\n")

    if args.model_id:
        download_and_sync_model(args.model_id, args.api_key)
    else:
        print(f"[INFO] Bắt đầu tải và đồng bộ các mô hình Anti-Spoofing mặc định:")
        for mid in MODEL_SHORT_ALIASES.keys():
            download_and_sync_model(mid, args.api_key)

    print("\n" + "=" * 75)
    print("[HOÀN THÀNH] Toàn bộ file weight đã sẵn sàng trong models/ và models/roboflow/!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
