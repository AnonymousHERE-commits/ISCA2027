"""SPLICE-V integer reference: the executed-manifest oracle of P1 and the
forward semantics that quantization-aware training must reproduce.

Every function here defines bit-exact behaviour.  Numbers come from the
generated module ``splicev_gen`` (generated/py), never from literals typed in
this package, so the reference, the PS host, the AIE kernels and the HLS
engines share one numeric contract.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# SPLICEV_GEN_DIR selects another generated profile (e.g. a trained model's manifest)
_GEN = Path(os.environ.get("SPLICEV_GEN_DIR") or (REPO / "generated" / "py"))
if str(_GEN) in sys.path:
    sys.path.remove(str(_GEN))
sys.path.insert(0, str(_GEN))

import splicev_gen as gen  # noqa: E402

__all__ = ["gen", "REPO"]
