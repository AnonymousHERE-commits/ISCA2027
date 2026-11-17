"""SPLICE-V processing-system runtime (Python on the A72s, XRT).

The PS loads the release, owns buffers and frame state, submits work, waits with
bounded timeouts and records evidence.  Per-fragment decisions, trust iterations and
microsecond scheduling stay on the device (roadmap 11.1); in the host-orchestrated
bring-up mode (gates G05-G07) Python sequences the steps and that mode is labelled
as such in every evidence file.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "reference", REPO / "generated" / "py"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__all__ = ["REPO"]
