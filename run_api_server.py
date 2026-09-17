"""
Launcher script for E-KYC FastAPI AI Server.
Khởi động máy chủ AI API phục vụ Node.js Backend và Web Client.
"""

import sys
import os

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="E-KYC Face Verification & Anti-Spoofing FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ IP lắng nghe (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng dịch vụ (mặc định: 8000)")
    parser.add_argument("--reload", action="store_true", help="Bật chế độ Auto-Reload khi sửa code")
    args = parser.parse_args()

    print("=" * 75)
    print("      E-KYC BIOMETRIC VERIFICATION & ENSEMBLE ANTI-SPOOFING SERVER")
    print("=" * 75)
    print(f"[*] API Base URL      : http://127.0.0.1:{args.port}")
    print(f"[*] Interactive Docs  : http://127.0.0.1:{args.port}/docs  (Swagger UI)")
    print(f"[*] Alternative Docs  : http://127.0.0.1:{args.port}/redoc (ReDoc)")
    print(f"[*] Web Demo Portal   : http://127.0.0.1:{args.port}/      (Webcam & File Test)")
    print(f"[*] Node.js Endpoint  : POST http://127.0.0.1:{args.port}/api/v1/verify")
    print("=" * 75)
    print("[*] Đang khởi tạo AI Pipeline và nạp mô hình vào bộ nhớ...")

    uvicorn.run(
        "server_module.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1
    )
