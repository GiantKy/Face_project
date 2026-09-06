"""
CLI & IPC Bridge Runner for Server Module.
Cho phép gọi các tác vụ AI từ terminal hoặc từ Server Node.js (qua child_process)
bằng chuẩn dữ liệu JSON (in/out).
"""

import sys
import os
import argparse
import json
from typing import Dict, Any

# Hỗ trợ UTF-8 output trên Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Hỗ trợ chạy độc lập cả khi ở trong server_module lẫn khi import như package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from server_module.pipeline_server import EKYCPipelineServer
    from server_module.utils import json_serialize_helper
except ImportError:
    from pipeline_server import EKYCPipelineServer
    from utils import json_serialize_helper


def main():
    parser = argparse.ArgumentParser(description="E-KYC Server Module Runner (YOLO AI Engine)")
    parser.add_argument(
        "--action",
        type=str,
        default="full_verify",
        choices=["validate_pose", "check_antispoof", "full_verify", "stdin_json"],
        help="Hành động AI cần thực hiện"
    )
    parser.add_argument("--input", type=str, default=None, help="Đường dẫn file ảnh hoặc chuỗi Base64")
    parser.add_argument("--img-id", type=str, default="1", help="ID của lượt kiểm tra eKYC")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục xuất báo cáo và ảnh crop")
    parser.add_argument("--blink-passed", action="store_true", default=True, help="Trạng thái chớp mắt")
    parser.add_argument("--head-passed", action="store_true", default=True, help="Trạng thái quay đầu")
    parser.add_argument("--head-action", type=str, default="TURN_LEFT", help="Tên hành động quay đầu")

    args = parser.parse_args()

    # Khởi tạo Pipeline Server
    server = EKYCPipelineServer()

    if args.action == "stdin_json":
        # Đọc dữ liệu JSON từ stdin (Dành cho Node.js gửi stream/IPC)
        input_raw = sys.stdin.read().strip()
        if not input_raw:
            print(json.dumps({"error": "Empty stdin payload"}))
            sys.exit(1)
        payload = json.loads(input_raw)
        action = payload.get("action", "full_verify")
        image_input = payload.get("image") or payload.get("image_base64")
        img_id = payload.get("img_id", 1)
        out_dir = payload.get("output_dir", None)
        blink_p = payload.get("blink_passed", True)
        head_p = payload.get("head_passed", True)
        head_act = payload.get("head_action", "TURN_LEFT")

        if action == "validate_pose":
            res = server.validate_pose(image_input)
        elif action == "check_antispoof":
            res = server.check_antispoof(image_input)
        else:
            res = server.full_verify(
                image_input,
                img_id=img_id,
                blink_passed=blink_p,
                head_movement_passed=head_p,
                head_action_name=head_act,
                output_dir=out_dir
            )

        print(json.dumps(res, ensure_ascii=False, indent=2, default=json_serialize_helper))
        return

    # Xử lý theo argument CLI
    if not args.input:
        print(json.dumps({"error": "Vui lòng cung cấp tham số --input <đường dẫn ảnh hoặc base64>"}))
        sys.exit(1)

    if args.action == "validate_pose":
        result = server.validate_pose(args.input)
    elif args.action == "check_antispoof":
        result = server.check_antispoof(args.input)
    elif args.action == "full_verify":
        result = server.full_verify(
            args.input,
            img_id=args.img_id,
            blink_passed=args.blink_passed,
            head_movement_passed=args.head_passed,
            head_action_name=args.head_action,
            output_dir=args.output_dir
        )
    else:
        result = {"error": f"Unknown action: {args.action}"}

    # In kết quả JSON ra stdout để ứng dụng gọi (Node.js/Shell) đọc
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_serialize_helper))


if __name__ == "__main__":
    main()
