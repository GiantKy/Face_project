"""
=============================================================================
Test Module: MiniFASNet Anti-Spoofing (Alias: test_anti_spoof_minisfasr.py)
=============================================================================
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from test_anti_spoof_minifasnet import main

if __name__ == "__main__":
    main()
