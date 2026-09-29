"""
Unit tests to verify all code review fixes across Standards and Spec axes.
"""

import os
import sys
import time
import pytest
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server_module.config import RFDETR_API_KEY, ENSEMBLE_SPOOF_VETO_THRESHOLD
from server_module.schemas import ChallengeStep, ChallengeAction, VerificationVerdict
from server_module.utils import extract_landmarks_with_fallback
from server_module.esp32_challenge import esp32_challenge_manager, ChallengeSession
from server_module.components.face_occlusion_detector import FaceOcclusionDetector
from server_module.app import trigger_esp32_relay


def test_config_security_and_veto_threshold():
    """Verify that hardcoded API secrets are eliminated and veto threshold is 0.65."""
    assert RFDETR_API_KEY != "ydUs8YBnVWjyjFFVvcpx", "Hardcoded RFDETR_API_KEY must be removed!"
    assert ENSEMBLE_SPOOF_VETO_THRESHOLD == 0.65, "Veto threshold must be synchronized to 0.65!"


def test_face_occlusion_detector_security():
    """Verify that FaceOcclusionDetector does not hardcode the Roboflow API key."""
    # When ROBOFLOW_API_KEY is not in env, api_key should be empty string or from env
    detector = FaceOcclusionDetector(strict_glasses=False)
    assert detector.api_key != "lGvF9eLaX4ZhhERgN5u2", "Hardcoded occlusion API key must be removed!"


def test_schemas_enums():
    """Verify that typed Enums are available and values match."""
    assert ChallengeStep.FACE_DETECT == "face_detect"
    assert ChallengeStep.EYE_BLINK == "eye_blink"
    assert ChallengeStep.HEAD_MOVEMENT == "head_movement"
    assert ChallengeStep.COMPLETED == "completed"

    assert ChallengeAction.TURN_LEFT == "TURN_LEFT"
    assert ChallengeAction.TURN_RIGHT == "TURN_RIGHT"

    assert VerificationVerdict.REAL == "REAL"
    assert VerificationVerdict.SPOOF == "SPOOF"
    assert VerificationVerdict.TIMEOUT_BLINK == "TIMEOUT_BLINK"


def test_extract_landmarks_with_fallback():
    """Verify extract_landmarks_with_fallback works cleanly with dummy/fallback inputs."""
    # When detector is None
    assert extract_landmarks_with_fallback(None, np.zeros((10, 10, 3), dtype=np.uint8)) is None

    class MockDetector:
        def __init__(self, fail_first=True):
            self.fail_first = fail_first
            self.calls = []

        def detect(self, img):
            self.calls.append(img.shape)
            if self.fail_first and len(self.calls) == 1:
                return []
            return [(i, i) for i in range(468)]

    raw = np.zeros((100, 100, 3), dtype=np.uint8)
    proc = np.zeros((100, 100, 3), dtype=np.uint8)

    mock = MockDetector(fail_first=True)
    res = extract_landmarks_with_fallback(mock, raw, proc, min_landmarks=468)
    assert res is not None
    assert len(res) == 468
    assert len(mock.calls) == 2, "Should have fallen back to raw frame!"


def test_esp32_relay_loopback_guard():
    """Verify that trigger_esp32_relay rejects loopback and localhost destinations."""
    assert trigger_esp32_relay("127.0.0.1") is False
    assert trigger_esp32_relay("localhost") is False
    assert trigger_esp32_relay("http://127.0.0.1:80/open") is False
    assert trigger_esp32_relay("") is False


def test_blink_timeout_fail_fast():
    """Verify that when blink times out (>10s), Fail-Fast terminates the session immediately."""
    dummy_session_id = "test_timeout_sess_123"
    sess = ChallengeSession(session_id=dummy_session_id, device_id="TEST_DEV")
    sess.current_step = "eye_blink"
    # Simulate step_start_time 11 seconds ago
    sess.step_start_time = time.time() - 11.0
    sess.frontal_frame = np.zeros((240, 240, 3), dtype=np.uint8)
    sess.primary_face_bbox = [20, 20, 180, 180]

    esp32_challenge_manager.sessions[dummy_session_id] = sess

    # Create dummy pipeline with mock landmark detector
    class DummyPipeline:
        class DummyLandmark:
            def detect(self, img):
                return [(i, i) for i in range(468)]
        landmark_detector = DummyLandmark()
        identity_verifier = None
        occlusion_detector = None

    pipe = DummyPipeline()
    dummy_frame = np.zeros((240, 240, 3), dtype=np.uint8)

    res = esp32_challenge_manager.process_step(
        session_id=dummy_session_id,
        image_input=dummy_frame,
        pipeline=pipe,
        step_name="eye_blink"
    )

    # Must Fail-Fast
    assert res["success"] is False, "Blink timeout must return success: False"
    assert res["approved"] is False, "Blink timeout must not approve"
    assert res["timed_out"] is True, "Must report timed_out: True"
    assert res["verdict"] == "TIMEOUT_BLINK", f"Verdict must be TIMEOUT_BLINK, got {res.get('verdict')}"
    assert "next_step" not in res or res.get("next_step") != "head_movement", "Must NOT advance to head_movement on timeout!"
    assert dummy_session_id not in esp32_challenge_manager.sessions, "Session must be deleted from memory on Fail-Fast!"


if __name__ == "__main__":
    print("[*] Running test_config_security_and_veto_threshold...")
    test_config_security_and_veto_threshold()
    print("[✓] Passed!")

    print("[*] Running test_face_occlusion_detector_security...")
    test_face_occlusion_detector_security()
    print("[✓] Passed!")

    print("[*] Running test_schemas_enums...")
    test_schemas_enums()
    print("[✓] Passed!")

    print("[*] Running test_extract_landmarks_with_fallback...")
    test_extract_landmarks_with_fallback()
    print("[✓] Passed!")

    print("[*] Running test_esp32_relay_loopback_guard...")
    test_esp32_relay_loopback_guard()
    print("[✓] Passed!")

    print("[*] Running test_blink_timeout_fail_fast...")
    test_blink_timeout_fail_fast()
    print("[✓] Passed!")

    print("\n" + "=" * 60)
    print(" ALL CODE REVIEW FIXES VERIFIED SUCCESSFULLY!")
    print("=" * 60)
