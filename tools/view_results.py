"""
eKYC Result Viewer - Dual Window & Side-by-Side Viewer Tool
Công cụ trực quan mở và xem các file kết quả eKYC thành 2 cửa sổ độc lập:
- Cửa sổ 1: Ảnh khuôn mặt sạch / Bounding Box & Landmarks tinh tế (không bị bảng che)
- Cửa sổ 2: Bảng điều khiển thông số Dashboard chi tiết
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server_module.utils import (
    create_pipeline_result_dashboard,
    create_side_by_side_result,
    show_dual_window_result
)


def find_result_sessions(search_root: str) -> List[str]:
    """Tìm tất cả các thư mục session kết quả có chứa file kết quả eKYC."""
    sessions = []
    if not os.path.exists(search_root):
        return sessions

    # Kiểm tra nếu chính search_root là một session
    if any(os.path.exists(os.path.join(search_root, f)) for f in ["1_pipeline_result.jpg", "4_report.json", "1_dashboard_panel.jpg"]):
        return [search_root]

    # Quét tất cả các thư mục con
    for root, dirs, files in os.walk(search_root):
        if "4_report.json" in files or "1_pipeline_result.jpg" in files or "1_dashboard_panel.jpg" in files:
            sessions.append(root)

    sessions.sort()
    return sessions


class EKYCResultViewer:
    def __init__(self, session_paths: List[str]):
        self.session_paths = session_paths
        self.current_idx = 0
        self.mode_dual_window = True  # True: 2 window riêng biệt; False: 1 window ghép ngang
        self.show_raw_face = False    # True: Xem ảnh gốc 0_raw_image; False: Xem ảnh annotated sạch
        self.win_face = "Window 1: eKYC Face & Detection View"
        self.win_dash = "Window 2: eKYC Metrics Dashboard"
        self.win_combined = "eKYC Results - Side by Side View"

    def run(self):
        if not self.session_paths:
            print("[CẢNH BÁO] Không tìm thấy thư mục kết quả nào để hiển thị.")
            return

        print("=" * 75)
        print("          eKYC RESULT VIEWER - DUAL WINDOW MODE ACTIVATED")
        print("=" * 75)
        print(f" Tìm thấy {len(self.session_paths)} session kết quả.")
        print("\n [PHÍM TẮT ĐIỀU KHIỂN]:")
        print("   -> [D] hoặc [Mũi tên Phải]: Xem session tiếp theo")
        print("   -> [A] hoặc [Mũi tên Trái] : Xem session trước đó")
        print("   -> [M]                    : Chuyển đổi giữa [2 Cửa Sổ Rời] và [1 Cửa Sổ Ghép]")
        print("   -> [C]                    : Bật/Tắt xem ảnh gốc (Raw Image vs Annotated)")
        print("   -> [Q] hoặc [ESC]         : Thoát viewer")
        print("=" * 75)

        while True:
            sess_dir = self.session_paths[self.current_idx]
            self._render_current_session(sess_dir)

            key = cv2.waitKey(0) & 0xFF

            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('d'), ord('D'), 83, 2555904):  # 'd' or Right Arrow
                if self.current_idx < len(self.session_paths) - 1:
                    self.current_idx += 1
                else:
                    print("[INFO] Đã ở session cuối cùng.")
            elif key in (ord('a'), ord('A'), 81, 2424832):  # 'a' or Left Arrow
                if self.current_idx > 0:
                    self.current_idx -= 1
                else:
                    print("[INFO] Đã ở session đầu tiên.")
            elif key in (ord('m'), ord('M')):
                self.mode_dual_window = not self.mode_dual_window
                # Đóng các cửa sổ cũ khi đổi mode
                cv2.destroyAllWindows()
                print(f"[MODE] Chế độ xem hiện tại: {'[2 CỬA SỔ RỜI (DUAL WINDOW)]' if self.mode_dual_window else '[1 CỬA SỔ GHÉP (SIDE-BY-SIDE)]'}")
            elif key in (ord('c'), ord('C')):
                self.show_raw_face = not self.show_raw_face
                print(f"[VIEW] Chế độ ảnh khuôn mặt: {'[ẢNH GỐC THUẦN (RAW)]' if self.show_raw_face else '[ẢNH ANNOTATED SẠCH]'}")

        cv2.destroyAllWindows()
        print("\n[INFO] Đã đóng Viewer.")

    def _render_current_session(self, sess_dir: str):
        session_name = os.path.basename(sess_dir)
        parent_name = os.path.basename(os.path.dirname(sess_dir))
        display_title = f"{parent_name}/{session_name}" if parent_name else session_name

        print(f"\n[HIỂN THỊ {self.current_idx + 1}/{len(self.session_paths)}] Session: {display_title}")

        # 1. Nạp ảnh khuôn mặt
        clean_path = os.path.join(sess_dir, "1_pipeline_result_clean.jpg")
        raw_path = os.path.join(sess_dir, "0_raw_image.jpg")
        default_res_path = os.path.join(sess_dir, "1_pipeline_result.jpg")

        face_img = None
        if self.show_raw_face and os.path.exists(raw_path):
            face_img = cv2.imread(raw_path)
        elif os.path.exists(clean_path):
            face_img = cv2.imread(clean_path)
        elif os.path.exists(raw_path):
            face_img = cv2.imread(raw_path)
        elif os.path.exists(default_res_path):
            face_img = cv2.imread(default_res_path)

        if face_img is None:
            face_img = np.full((480, 640, 3), (30, 30, 35), dtype=np.uint8)
            cv2.putText(face_img, "NO FACE IMAGE FOUND", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 2. Nạp hoặc tạo ảnh Dashboard Panel
        dash_path = os.path.join(sess_dir, "1_dashboard_panel.jpg")
        report_path = os.path.join(sess_dir, "4_report.json")
        crop_path = os.path.join(sess_dir, "2_face_crop_224.jpg")

        dash_img = None
        if os.path.exists(dash_path):
            dash_img = cv2.imread(dash_path)
        elif os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                crop_face = cv2.imread(crop_path) if os.path.exists(crop_path) else None
                face_dec = rep.get("face_detection", {})
                p_val = rep.get("pose_validation", {})
                a_spoof = rep.get("anti_spoofing", {})
                b_val = rep.get("blink_validation", {})
                h_val = rep.get("head_movement", {})
                f_dec = rep.get("final_decision", {})

                dash_img = create_pipeline_result_dashboard(
                    img_idx=session_name,
                    face_info={"confidence": face_dec.get("confidence", 0.0)} if face_dec.get("num_faces", 0) > 0 else None,
                    num_faces=face_dec.get("num_faces", 0),
                    pose_info=p_val,
                    pose_valid=p_val.get("is_valid", False),
                    anti_spoof_info=a_spoof,
                    spoof_iou=a_spoof.get("spoof_iou", 0.0),
                    blink_passed=b_val.get("passed", False),
                    blink_count=b_val.get("blink_count", 0),
                    head_movement_passed=h_val.get("passed", False),
                    head_action_name=h_val.get("action_performed", "NONE"),
                    final_pass=f_dec.get("passed", False),
                    reasons=f_dec.get("reasons", []),
                    face_crop=crop_face,
                    target_height=face_img.shape[0]
                )
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể tự động tạo dashboard từ report.json: {e}")

        if dash_img is None:
            dash_img = np.full((face_img.shape[0], 560, 3), (20, 22, 28), dtype=np.uint8)
            cv2.putText(dash_img, "DASHBOARD NOT AVAILABLE", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # 3. Hiển thị theo chế độ
        if self.mode_dual_window:
            # Gắn watermark chỉ số session lên góc dưới ảnh mặt
            vis_face = face_img.copy()
            tag_text = f"[{self.current_idx + 1}/{len(self.session_paths)}] ID: {display_title}"
            cv2.rectangle(vis_face, (10, vis_face.shape[0] - 38), (340, vis_face.shape[0] - 10), (15, 15, 20), -1)
            cv2.putText(vis_face, tag_text, (18, vis_face.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

            show_dual_window_result(
                win_img_name=self.win_face,
                img=vis_face,
                win_dash_name=self.win_dash,
                dash=dash_img,
                offset_x=60,
                offset_y=60,
                wait_key=False
            )
        else:
            # Chế độ 1 cửa sổ ghép đôi
            side_by_side = create_side_by_side_result(face_img, dash_img)
            cv2.namedWindow(self.win_combined, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(self.win_combined, side_by_side)
            cv2.moveWindow(self.win_combined, 60, 60)


def main():
    parser = argparse.ArgumentParser(description="eKYC Dual-Window Result Viewer")
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Đường dẫn tới thư mục kết quả cần xem (ví dụ: output/test_server_module/test_0)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Quét và xem toàn bộ tất cả các kết quả trong thư mục output/"
    )
    args = parser.parse_args()

    target_dir = args.dir
    if target_dir is None:
        default_out = os.path.join(PROJECT_ROOT, "output")
        if not os.path.exists(default_out):
            print(f"[LỖI] Thư mục output không tồn tại: {default_out}")
            sys.exit(1)
        target_dir = default_out

    sessions = find_result_sessions(target_dir)
    if not sessions:
        print(f"[CẢNH BÁO] Không tìm thấy session kết quả nào trong: {target_dir}")
        sys.exit(0)

    viewer = EKYCResultViewer(sessions)
    viewer.run()


if __name__ == "__main__":
    main()
