"""SPLICE detector: YOLO26-S with the SPLICE graph-wavelet neck (SPLICE-B of SPLICE-V, in its
seven readout arms; deployed arm (e)).

Layout
  config.py          SpliceConfig, the readout arms and the forecast descriptors
  backend.py         numpy / torch array shim: the neck math runs unchanged on both
  ops.py             neck operations (walks, bands, restriction, descriptor, admissibility,
                     exposure discount, forecast, readout)
  exchange_float.py  float exchange (Eq. (1) posterior, Eq. (2) trust solve, one-hop dH) and its decisions
  exchange_torch.py  the same exchange, differentiable on fixed decisions
  targets.py         supervision targets from ground-truth boxes (ownership, affinities, hypotheses)
  linkage_bridge.py  reserve selection and certified linkage through the integer reference (CPU)
  neck.py            torch modules of the SPLICE-B neck
  yolo26.py          integration with an Ultralytics YOLO26-S model
  data.py            dataset views (COCO 2017, CrowdHuman) and loaders
  engine.py          model construction, resumable training, calibration, detection collection
  metrics.py         COCO AP, CrowdHuman AP / MR^-2, bootstrap tests
  diagnostics.py     train-channel quantities (forecast error, admissibility, Lemma 3, leakage)
  losses.py, qat.py, calibrate.py, export.py, selfcheck.py, train.py, evaluate.py, toytrain.py

The integer reference (reference/splicev_ref) defines model M bit for bit; this package is
model DFP32 / Db and its quantization-aware forward follows the manifest's rounding sites.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "reference", REPO / "generated" / "py"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__all__ = ["REPO"]
