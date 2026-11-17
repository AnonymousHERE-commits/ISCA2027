"""Quantization-aware training at the manifest's sites (roadmap 6.3: the forward pass executes
the eventual rounding; the gradient estimator is the straight-through estimator).

Activation sites of the neck (SpliceNeck._q) use power-of-two scales so every integer
product on the device is followed by a plain shift:
  latent, H, K, Uv, VO, VO_T, Xsyn, cls_in*, box_in*   int8,  lsb = transport_lsb
  dHhat, dH                                            transport format (symmetric +-127)
  band<l><j>                                           int8 band codes, lsb = transport_lsb * 2^-e_lj
Weights of the neck's linear maps are fake-quantized to int8 with a per-tensor power-of-two
scale.  Backbone and head convolutions get int8 activation quantization at their outputs
(per-layer power-of-two scale fixed by calibrate.py); their weights are quantized after BN
folding at export (documented approximation: Db/M differences are measured by P2).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def fake_quant(x: torch.Tensor, lsb: float, lo: int, hi: int) -> torch.Tensor:
    return torch.clamp(ste_round(x / lsb), lo, hi) * lsb


def pot_exponent(max_abs: float, qmax: int = 127) -> int:
    """Largest k with max_abs * 2^k <= qmax (weights w_int = round(w * 2^k))."""
    if max_abs <= 0:
        return 0
    return int(math.floor(math.log2(qmax / max_abs)))


class PoTWeight(nn.Module):
    """Parametrization: per-tensor power-of-two int8 fake quantization of a weight."""

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        k = pot_exponent(float(w.detach().abs().max()))
        return fake_quant(w, 2.0 ** -k, -127, 127)


class NeckQuant:
    """Callable installed as SpliceNeck.quant."""

    def __init__(self, neck, transport_lsb: float):
        self.neck = neck
        self.lsb = transport_lsb

    def __call__(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name.startswith("band"):
            l, j = int(name[4]), int(name[5])
            e = float(self.neck.band_exp[l, j])
            return fake_quant(x, self.lsb * 2.0 ** -e, -128, 127)
        if name in ("dHhat", "dH"):
            return fake_quant(x, self.lsb, -127, 127)
        return fake_quant(x, self.lsb, -128, 127)


def attach(model, act_scales: Optional[Dict[str, float]] = None) -> Dict:
    """Enable QAT on a SpliceYOLO.  Returns a summary of what was quantized."""
    neck = model.neck
    neck.quant = NeckQuant(neck, neck.cfg.transport_lsb)
    lin = 0
    for name, m in neck.named_modules():
        if isinstance(m, nn.Linear) and not parametrize.is_parametrized(m, "weight"):
            parametrize.register_parametrization(m, "weight", PoTWeight())
            lin += 1
    hooks = 0
    act_scales = act_scales or {}

    def make_hook(lsb):
        def hook(_m, _inp, out):
            return fake_quant(out, lsb, -128, 127) if torch.is_tensor(out) else out
        return hook

    handles = []
    for name, m in model.yolo.model.named_modules():
        if name in act_scales:
            handles.append(m.register_forward_hook(make_hook(act_scales[name])))
            hooks += 1
    model._qat_hooks = handles
    return {"neck_linear_weights": lin, "activation_hooks": hooks, "neck_sites": "all SpliceNeck._q sites"}


def detach(model) -> None:
    """Remove the weight parametrizations (keeps the float weights) and the neck hook."""
    neck = model.neck
    neck.quant = None
    for h in getattr(model, "_qat_hooks", []):
        h.remove()
    model._qat_hooks = []
    for m in neck.modules():
        if isinstance(m, nn.Linear) and parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=False)
