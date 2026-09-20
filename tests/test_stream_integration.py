# -*- coding: utf-8 -*-
"""
Test Stream Integration: Webhook, Relay Trigger & Fail-Fast Logic
"""
import sys
import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Setup path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tests.test_pipeline_ensemble_full import send_nodejs_webhook, trigger_esp32_relay

class MockReceiverHandler(BaseHTTPRequestHandler):
    received_post = None
    received_get = False

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        MockReceiverHandler.received_post = json.loads(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "SUCCESS"}')

    def do_GET(self):
        MockReceiverHandler.received_get = True
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "DOOR_OPENED"}')

    def log_message(self, format, *args):
        pass  # Quiet test output

def test_stream_integration():
    print("[TEST] Khởi tạo Mock HTTP Server...")
    server = HTTPServer(('127.0.0.1', 0), MockReceiverHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        # Test 1: Node.js Webhook
        print(f"[TEST 1] Kiểm tra send_nodejs_webhook tới port {port}...")
        test_payload = {
            "image_id": 999,
            "approved": True,
            "verdict": "REAL",
            "is_real": True,
            "confidence": 0.985,
            "liveness": {"blink": True, "head_movement": True}
        }
        res = send_nodejs_webhook(f"http://127.0.0.1:{port}/api/ekyc/result", test_payload)
        assert res is True, "send_nodejs_webhook phải trả về True"
        assert MockReceiverHandler.received_post["verdict"] == "REAL"
        print("  -> [PASS] Webhook gửi và nhận chính xác!")

        # Test 2: ESP32 Relay Trigger
        print(f"[TEST 2] Kiểm tra trigger_esp32_relay tới port {port}...")
        res_relay = trigger_esp32_relay(f"127.0.0.1:{port}")
        assert res_relay is True, "trigger_esp32_relay phải trả về True"
        assert MockReceiverHandler.received_get is True
        print("  -> [PASS] Relay trigger gửi và nhận chính xác!")

        print("\n[ALL TESTS PASSED] Toàn bộ module tích hợp hoạt động 100% chuẩn xác!\n")
    finally:
        server.shutdown()
        server.server_close()

if __name__ == "__main__":
    test_stream_integration()
