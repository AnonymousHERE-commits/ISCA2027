"""SPLICE on an Ultralytics YOLO26-S model.

SPLICE-B (SPLICE-V section 2): the backbone and the detection head of YOLO26-S are kept, the
native neck is replaced by the SPLICE core.  The co-designed readout feeds the head's box
branch with the detail part Q X'' and its class branch with the shared part Pi X'' (plus the
arm's residual M r O), and the class score is p = sigma(l) gamma when the gate is applied.

Only stable parts of the Ultralytics API are used: `YOLO(weights).model` (DetectionModel
with `.model` layers, `.yaml`, `.loss`), the Detect head's raw training output and
`head._inference` / `head.postprocess`.  Two raw formats are handled:
  current releases (YOLO26 era)  dict {"boxes" (N, 4 reg_max, A), "scores" (N, nc, A), "feats"},
                                 nested as {"one2many": ..., "one2one": ...} when the head has
                                 a one-to-one branch
  older releases                 list of per-level maps (N, 4 reg_max + nc, H, W), or
                                 {"one2many": [...], "one2one": [...]}
The smoke scale of the Arrhenius suite checks them on the installed version first.
"""
from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import SpliceConfig
from .neck import SpliceNeck


def load_yolo(weights: str = "yolo26s.pt"):
    from ultralytics import YOLO
    return YOLO(weights).model


def _route(m, x, y):
    if m.f == -1:
        return x
    if isinstance(m.f, int):
        return y[m.f]
    return [x if j == -1 else y[j] for j in m.f]


def probe_layout(model, imgsz: int = 640) -> Dict:
    """Backbone taps (last backbone layer at strides 8/16/32) and the head's input channels."""
    layers = model.model
    n_backbone = len(model.yaml["backbone"])
    head = layers[-1]
    shapes, head_in = {}, {}

    def hook_layer(i):
        def fn(_m, _inp, out):
            if torch.is_tensor(out):
                shapes[i] = tuple(out.shape)
        return fn

    def hook_head(_m, inp):
        head_in["shapes"] = [tuple(t.shape) for t in inp[0]]

    hs = [layers[i].register_forward_hook(hook_layer(i)) for i in range(n_backbone)]
    hh = head.register_forward_pre_hook(hook_head)
    was = model.training
    model.eval()
    with torch.no_grad():
        p = next(model.parameters())
        model(torch.zeros(1, 3, imgsz, imgsz, device=p.device, dtype=p.dtype))
    model.train(was)
    for h in hs:
        h.remove()
    hh.remove()
    taps = []
    for stride in (8, 16, 32):
        size = imgsz // stride
        cand = [i for i, s in shapes.items() if len(s) == 4 and s[-1] == size]
        if not cand:
            raise RuntimeError(f"no backbone layer at stride {stride}: shapes {shapes}")
        taps.append(max(cand))
    return {"n_backbone": n_backbone, "taps": taps, "tap_channels": [shapes[i][1] for i in taps],
            "head_channels": [s[1] for s in head_in["shapes"]], "head_strides": [imgsz // s[-1] for s in head_in["shapes"]]}


def _combine(a, b, nb: int):
    """Box outputs from the box pass, class outputs from the class pass (same layout)."""
    if isinstance(a, dict):
        if "boxes" in a and "scores" in a:                    # current format: one branch's dict
            out = dict(a)
            out["scores"] = b["scores"]
            return out
        return {k: _combine(a[k], b[k], nb) for k in a}
    if isinstance(a, (list, tuple)):
        return type(a)(_combine(x, y, nb) for x, y in zip(a, b))
    return torch.cat([a[:, :nb], b[:, nb:]], 1)               # older format: per-level map


def _postprocess(head, y: torch.Tensor):
    """head.postprocess on (N, A, 4 + nc): postprocess(preds) in current releases,
    postprocess(preds, max_det, nc) in older ones."""
    preds = y.permute(0, 2, 1)
    try:
        old_api = "max_det" in inspect.signature(head.postprocess).parameters
    except (TypeError, ValueError):
        old_api = False
    if old_api:
        return head.postprocess(preds, getattr(head, "max_det", 300), head.nc)
    return head.postprocess(preds)


def _add_offset(raw, offsets: Sequence[torch.Tensor], nb: int):
    """Arm (c): add a per-cell class-logit offset (one map (N, 1, H, W) per level) to the raw
    class outputs of every branch."""
    if isinstance(raw, dict):
        if "boxes" in raw and "scores" in raw:                # current format: (N, nc, A), levels in order
            out = dict(raw)
            out["scores"] = raw["scores"] + torch.cat([o.flatten(2) for o in offsets], 2).to(raw["scores"].dtype)
            return out
        return {k: _add_offset(v, offsets, nb) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):                        # older format: per-level maps
        return type(raw)(torch.cat([x[:, :nb], x[:, nb:] + o.to(x.dtype)], 1) for x, o in zip(raw, offsets))
    return raw


class NativeYOLO(nn.Module):
    """YOLO26-S with its native neck behind the interface of SpliceYOLO (baseline rows of
    Table 7: the published checkpoint, or the same model trained with this package's loop)."""

    def __init__(self, yolo_model):
        super().__init__()
        self.yolo = yolo_model
        self.neck = None

    def forward(self, x: torch.Tensor, boxes=None, probe: bool = False, stats: bool = False, gate=True):
        y = self.yolo(x)
        if self.training:
            return y, None
        return (y[0] if isinstance(y, (list, tuple)) else y), None

    def detection_loss(self, raw, batch: Dict):
        return self.yolo.loss(batch, raw)


class SpliceYOLO(nn.Module):
    def __init__(self, yolo_model, cfg: Optional[SpliceConfig] = None, workers: int = 0, imgsz: int = 640):
        super().__init__()
        self.cfg = cfg or SpliceConfig()
        self.yolo = yolo_model
        self.layout = probe_layout(yolo_model, imgsz)
        self.n_backbone = self.layout["n_backbone"]
        self.taps = self.layout["taps"]
        self.nb = int(getattr(self.head, "reg_max", 16)) * 4
        self.neck = SpliceNeck(self.layout["tap_channels"], self.layout["head_channels"], self.cfg, workers,
                               tuple(self.layout["head_strides"]))
        p = next(yolo_model.parameters())
        self.neck.to(device=p.device)

    @property
    def head(self):
        """The YOLO26 Detect head (not registered twice: one state-dict entry per weight)."""
        return self.yolo.model[-1]

    @property
    def names(self):
        return getattr(self.yolo, "names", None)

    def backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        y = []
        for m in self.yolo.model[:self.n_backbone]:
            x = m(_route(m, x, y))
            y.append(x)
        return [y[i] for i in self.taps]

    def head_raw(self, feats: Sequence[torch.Tensor]):
        """The head's raw (training-format) output without switching its layers to train mode."""
        flag = self.head.training
        self.head.training = True
        try:
            return self.head(list(feats))
        finally:
            self.head.training = flag

    def forward(self, x: torch.Tensor, boxes: Optional[List[np.ndarray]] = None, probe: bool = False,
                stats: bool = False, gate=True):
        """Training mode: (raw head outputs, neck outputs).  Evaluation mode: (decoded
        predictions, neck outputs); gate=True applies p = sigma(l) gamma, gate=False leaves the
        class score unchanged, gate='both' returns {'gate': ..., 'nogate': ...}."""
        taps = self.backbone(x)
        with torch.autocast(device_type=x.device.type, enabled=False):        # the neck runs in float32
            nk = self.neck([t.float() for t in taps], boxes, probe, stats)
        raw = _combine(self.head_raw(nk["box_feats"]), self.head_raw(nk["cls_feats"]), self.nb)
        if nk.get("score_offset") is not None:
            raw = _add_offset(raw, nk["score_offset"], self.nb)
        if self.training:
            return raw, nk
        if gate == "both":
            return {"gate": self.decode(raw, nk["gates"], True), "nogate": self.decode(raw, nk["gates"], False)}, nk
        return self.decode(raw, nk["gates"], bool(gate)), nk

    def decode(self, raw, gates: Sequence[torch.Tensor], gate: bool = True):
        """Decoded predictions; with gate=True the class score is multiplied by the objectness
        gate, p = sigma(l) gamma (Eq. (5)).

        End-to-end heads return (N, max_det, 6) [x0, y0, x1, y1, score, class]; otherwise
        (N, 4 + nc, A) with xywh boxes for NMS."""
        head = self.head
        e2e = bool(getattr(head, "end2end", False))
        feats = raw
        if isinstance(raw, dict) and "one2many" in raw:
            feats = raw["one2one"] if e2e else raw["one2many"]
        y = head._inference(feats if isinstance(feats, dict) else list(feats))
        if gate:
            g = torch.cat([gt.flatten(1) for gt in gates], 1).to(y.dtype)
            y = torch.cat([y[:, :4], y[:, 4:] * g[:, None, :]], 1)
        if e2e and hasattr(head, "postprocess"):
            return _postprocess(head, y)
        return y

    def detection_loss(self, raw, batch: Dict):
        """YOLO26's own criterion on the combined raw predictions."""
        return self.yolo.loss(batch, raw)
