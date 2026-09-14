# -*- coding: utf-8 -*-
"""
Script tải mô hình RF-DETR Small từ Roboflow về thư mục models/
"""
import os
import sys
import json
import urllib.request
import urllib.error
import shutil
import warnings

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

API_KEY = "ydUs8YBnVWjyjFFVvcpx"
WORKSPACE_NAME = "k-thi-gia-s-workspace"
WORKFLOW_ID = "face-spoof-detection-vface-spoof-detection-liika-owgrl-1-rfdetr-small-t1-logic"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
ROBOFLOW_CACHE_DIR = os.path.join(MODELS_DIR, "roboflow")
os.environ["MODEL_CACHE_DIR"] = ROBOFLOW_CACHE_DIR

print(f"[INFO] Python: {sys.executable}")
print(f"[INFO] Workspace: {WORKSPACE_NAME}")
print(f"[INFO] Workflow ID: {WORKFLOW_ID}")
print(f"[INFO] Cache Dir: {ROBOFLOW_CACHE_DIR}")

# 1. Thử truy vấn chi tiết Workflow từ Roboflow API
def inspect_workflow():
    urls = [
        f"https://api.roboflow.com/workflows/{WORKSPACE_NAME}/{WORKFLOW_ID}?api_key={API_KEY}",
        f"https://serverless.roboflow.com/workflows/{WORKSPACE_NAME}/{WORKFLOW_ID}?api_key={API_KEY}",
        f"https://api.roboflow.com/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}?api_key={API_KEY}",
    ]
    for url in urls:
        try:
            print(f"[INFO] Đang kiểm tra API: {url.split('?')[0]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print("[OK] Nhận thông tin Workflow thành công!")
                print(json.dumps(data, indent=2)[:1000])
                return data
        except Exception as e:
            print(f"[DEBUG] Thử endpoint {url.split('?')[0]} thất bại: {e}")
    return None

# 2. Thử tải model bằng inference get_model với các candidate ID
def try_download_model():
    candidate_ids = [
        f"{WORKSPACE_NAME}/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1",
        "face-spoof-detection-liika-owgrl-1-rfdetr-small-t1",
        "face-spoof-detection-liika-owgrl/1",
        f"{WORKSPACE_NAME}/face-spoof-detection-liika-owgrl/1",
    ]

    print("\n[INFO] Đang thử tải weights bằng inference.get_model...")
    try:
        from inference import get_model
    except ImportError:
        print("[ERROR] Không tìm thấy thư viện inference!")
        return

    for cid in candidate_ids:
        print(f"\n---> Thử model_id: '{cid}'...")
        try:
            model = get_model(model_id=cid, api_key=API_KEY)
            print(f"[THÀNH CÔNG] Đã nạp model: {cid}")
            return cid, model
        except Exception as e:
            print(f"[THẤT BẠI] Lỗi với {cid}: {e}")

    return None, None

if __name__ == "__main__":
    wf_data = inspect_workflow()
    cid, model = try_download_model()
