"""Export a trained SPLICE detector: integer weights, coefficients and the model manifest.

Outputs in <out>/:
  tensors.svb           int8 weights (per-channel / per-tensor power-of-two exponents) and the
                        integer coefficients of the neck (SVB1, reference/splicev_ref/svb.py)
  float_tensors.npz     float biases, BN-folded float weights and activation scales (the
                        program generator of the full detector turns them into int32 biases
                        and shifts once the activation graph is fixed, gate G10)
  inventory.json        every exported layer: name, op, shapes, exponents, consumer engine
  calibration.json      the calibrated values (band exponents, gate thresholds, prior, ...)
  manifest.model.json   extends contracts/manifest.dev.json; every SYN-xx value replaced by
                        a calibrated or model value (kind 'calibrated' / 'model'); stays
                        non-deployable while partition.catalog is null (hardware track, G08)
  export_report.json    hashes of everything above and of the source checkpoint

The manifest writer and the quantizers are numpy-only (tested here); export_torch() needs torch.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import REPO
from splicev_ref import svb


def pot_exponent(max_abs: float, qmax: int = 127) -> int:
    if not max_abs > 0:
        return 0
    return int(math.floor(math.log2(qmax / max_abs)))


def quantize_tensor(w: np.ndarray, qmax: int = 127) -> Tuple[np.ndarray, int]:
    k = pot_exponent(float(np.abs(w).max()) if w.size else 0.0, qmax)
    return np.clip(np.round(w * 2.0 ** k), -qmax, qmax).astype(np.int64), k


def quantize_per_channel(w: np.ndarray, qmax: int = 127) -> Tuple[np.ndarray, np.ndarray]:
    """w (Cout, ...) -> int8 codes and one exponent per output channel."""
    flat = w.reshape(w.shape[0], -1)
    ks = np.array([pot_exponent(float(np.abs(r).max()) if r.size else 0.0, qmax) for r in flat], dtype=np.int64)
    q = np.clip(np.round(flat * (2.0 ** ks)[:, None]), -qmax, qmax).astype(np.int64)
    return q.reshape(w.shape), ks


def forecast_multipliers(lsb_k: float, lsb_u: float, d: int, k_u0: int, k_ua: int, lsb_nu: float = 1.0,
                         logit_frac: int = 4, sig_frac: int = 8) -> Dict[str, int]:
    """Integer logit multipliers of the forecast HEAD unit from the export scales.

    l_code = rne((kk * kk_mult + uu * uu_mult) >> shift) with l_code = 2^logit_frac * l and
    l = kk lsb_k^2 / sqrt(d) + uu lsb_u^2;  null and alpha logits from nu . u (u = u_int 2^-k)."""
    def mult(x: float) -> Tuple[int, int]:
        for s in range(0, 31):
            m = x * 2.0 ** s
            if m >= 64 or s == 30:
                return max(1, int(round(m))), s
        return 1, 0
    kk = (2 ** logit_frac) * lsb_k ** 2 / math.sqrt(d)
    uu = (2 ** logit_frac) * lsb_u ** 2
    s = 0
    while max(kk, uu) * 2.0 ** s < 64 and s < 30:
        s += 1
    nm, ns = mult((2 ** logit_frac) * lsb_nu * 2.0 ** -k_u0)
    am, ash = mult((2 ** sig_frac) * lsb_nu * 2.0 ** -k_ua)
    return {"kk_mult": max(1, int(round(kk * 2 ** s))), "uu_mult": max(1, int(round(uu * 2 ** s))), "shift": s,
            "null_mult": nm, "null_shift": ns, "alpha_mult": am, "alpha_shift": ash}


def model_manifest(run: str, overrides: Dict, decisions: Dict, note: str, deployable: bool = False,
                   out_dir: Optional[Path] = None) -> Dict:
    """Manifest of a trained model.  It extends contracts/manifest.dev.json through a path
    relative to where the manifest is written (out_dir), so it loads from that folder."""
    import os
    base = REPO / "contracts" / "manifest.dev.json"
    ext = os.path.relpath(base, Path(out_dir).resolve()) if out_dir is not None else "manifest.dev.json"
    return {
        "schema": "splicev-manifest/1",
        "manifest_id": f"splicev-model-{run}",
        "extends": ext,
        "profile": "model",
        "deployable": bool(deployable),
        "note": note,
        "overrides": overrides,
        "decisions": decisions,
    }


def calibrated_overrides(calib: Dict, cfg: Dict, mults: Optional[Dict] = None, rz_shift: Optional[int] = None,
                         dup_edge_bound: Optional[int] = None, provenance: str = "T") -> Tuple[Dict, Dict]:
    """Manifest overrides + decision records from a calibration record and the model config."""
    ov, dec = {}, {}

    def put(path, value, kind, why, ident):
        ov[path] = value
        dec[path] = {"id": ident, "kind": kind, "why": why, "provenance": provenance}

    ex = calib.get("band_exponents")
    if ex is not None:
        put("bands.exponents", {"P3": list(map(int, ex[0])), "P4": list(map(int, ex[1])), "P5": list(map(int, ex[2]))},
            "calibrated", "floor(log2(127 / q99.99 |Z| / lsb)) on the calibration split", "CAL-01")
        put("bands.lowpass_exponent", int(calib.get("lowpass_exponent", 0)), "calibrated", "low-pass band range on the calibration split", "CAL-01")
    if calib.get("gate_tau") is not None:
        put("gate.thresholds", [float(t) for t in calib["gate_tau"]], "calibrated",
            f"Mondrian conformal quantile of same-instance d^2, delta = {calib.get('delta', 0.01)}", "CAL-02")
        put("gate.strata", {"by": "fragment_count_F", "bounds": list(map(int, cfg.get("gate_strata", (32, 96, 192))))},
            "model", "strata of the calibration", "CAL-02")
    if calib.get("prior") is not None:
        put("posterior.prior_table", [[float(p) for p in calib["prior"]]], "calibrated",
            "frequencies of min(K, Kmax) true partners among admissible partners", "CAL-03")
    for k in ("a_r", "b_r", "c_r"):
        if k in cfg:
            put(f"trust.{k}", float(cfg[k]), "model", "trust constant of the trained model", "CAL-04")
    if mults is not None:
        put("forecast.logit_multipliers", {k: int(v) for k, v in mults.items()}, "model", "derived from the export scales", "CAL-05")
    if "u_dim" in cfg:
        put("forecast.u_dim", int(cfg["u_dim"]), "model", "width of U in the trained model", "CAL-05")
    if rz_shift is not None:
        put("readout.rz_shift", int(rz_shift), "model", "exponent of the exported residual maps Z_l", "CAL-06")
    if "probe_sigma_lsb" in cfg:
        put("forecast.probe_distribution", f"normal(0, ({cfg['probe_sigma_lsb']} LSB)^2) per transported channel and fragment",
            "model", "the probe used in training", "CAL-07")
    if dup_edge_bound is not None:
        put("posterior.duplicate_edge_bound", int(dup_edge_bound), "calibrated", "largest duplicate-edge count on the calibration split, with margin", "CAL-08")
    return ov, dec


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(out: Path, bundle: svb.Bundle, floats: Dict[str, np.ndarray], inventory: List[Dict], calib: Dict,
                  manifest: Dict, source: Dict) -> Dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle.write(out / "tensors.svb")
    np.savez(out / "float_tensors.npz", **floats)
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (out / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n")
    (out / "manifest.model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = ["tensors.svb", "float_tensors.npz", "inventory.json", "calibration.json", "manifest.model.json"]
    report = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": source,
              "sha256": {f: sha256_file(out / f) for f in files}, "n_tensors": len(bundle.items), "n_layers": len(inventory)}
    (out / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


# --------------------------------------------------------------------------- torch export

def export_torch(ckpt_path: str, out: str, run: str = "run") -> Dict:
    """Export a final checkpoint of this package (keys: 'model' state, 'cfg', 'calibration',
    'weights', 'nc').  The integer contract of the device is the deployed arm (e)."""
    import torch
    import torch.nn as nn
    from .engine import load_model

    model, ck = load_model(ckpt_path, torch.device("cpu"), 0, use_ema=False)
    if getattr(model, "neck", None) is None:
        raise ValueError("the native YOLO26-S has no SPLICE export")
    cfg = model.neck.cfg
    model.eval()
    calib = ck.get("calibration", {})
    bundle, floats, inv = svb.Bundle(), {}, []
    # the native YOLO26 neck (layers between the backbone and the head) is replaced: not exported
    unused = [f"yolo.model.{i}" for i in range(model.n_backbone, len(model.yolo.model) - 1)]
    live = [(n, m) for n, m in model.named_modules() if not any(n == u or n.startswith(u + ".") for u in unused)]
    pairs, handled = [], set()
    for name, m in live:
        conv, bn = getattr(m, "conv", None), getattr(m, "bn", None)                  # Ultralytics Conv blocks
        mix = getattr(m, "mix", None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            pairs.append((name, conv, bn, getattr(m, "act", None)))
        elif isinstance(mix, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):           # ModFuse: modulated 3x3 as 1x1 over 9 taps
            pairs.append((name + ".mix", mix, bn, getattr(m, "act", None)))
        elif isinstance(m, nn.Sequential) and len(m) >= 2 and isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.BatchNorm2d):
            pairs.append((name, m[0], m[1], m[2] if len(m) > 2 else None))       # the neck's conv-BN-act blocks
    for name, conv, bn, act in pairs:
        handled.add(id(conv))
        scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
        w = (conv.weight.detach() * scale[:, None, None, None]).numpy()
        b = ((conv.bias.detach() if conv.bias is not None else 0) - bn.running_mean.detach()) * scale + bn.bias.detach()
        q, ks = quantize_per_channel(w)
        bundle.add(f"{name}.w", q)
        bundle.add(f"{name}.k", ks)
        floats[f"{name}.b"] = b.numpy()
        floats[f"{name}.w_folded"] = w
        inv.append({"name": name, "op": "conv+bn (folded)", "shape": list(w.shape), "stride": list(conv.stride),
                    "groups": conv.groups, "act": act.__class__.__name__ if act is not None else None,
                    "engine": "AIE CONV unit"})
    for name, m in live:
        if isinstance(m, nn.Conv2d) and id(m) not in handled:
            w = m.weight.detach().numpy()
            q, ks = quantize_per_channel(w)
            bundle.add(f"{name}.w", q)
            bundle.add(f"{name}.k", ks)
            if m.bias is not None:
                floats[f"{name}.b"] = m.bias.detach().numpy()
            inv.append({"name": name, "op": "conv", "shape": list(w.shape), "stride": list(m.stride), "engine": "AIE CONV unit"})
        elif isinstance(m, nn.Linear):
            w = m.weight.detach().numpy()
            q, k = quantize_tensor(w)
            bundle.add(f"{name}.w", q.T)                  # (in, out): the GEMM unit's B matrix
            bundle.add(f"{name}.k", [k])
            if m.bias is not None:
                floats[f"{name}.b"] = m.bias.detach().numpy()
            inv.append({"name": name, "op": "linear", "shape": list(w.shape), "k": k, "engine": "AIE GEMM unit"})
    neck = model.neck
    bundle.add("neck.band_exp", neck.band_exp.numpy().astype(np.int64))
    bundle.add("neck.mix_gain_q8", np.round(neck.mix_gain.detach().numpy() * 256).astype(np.int64))
    floats["neck.gate_tau"] = neck.gate_tau.numpy()
    floats["neck.prior"] = neck.prior.numpy()
    floats["neck.score_bias"] = neck.score_bias.detach().numpy()
    lsb = cfg.transport_lsb
    k_u0 = pot_exponent(float(neck.u0.weight.detach().abs().max()))
    k_ua = pot_exponent(float(neck.u_alpha.weight.detach().abs().max()))
    mults = forecast_multipliers(lsb, lsb, cfg.key_dim, k_u0, k_ua)
    rz_k = max(pot_exponent(float(z.weight.detach().abs().max())) for z in neck.Z)
    ov, dec = calibrated_overrides(calib, cfg.to_dict(), mults, rz_k, calib.get("dup_edge_bound"))
    manifest = model_manifest(run, ov, dec, f"Trained SPLICE-B arm ({cfg.arm}), descriptor '{cfg.descriptor}'; non-deployable "
                              "until partition.catalog (G08) exists.", out_dir=Path(out))
    src = {"checkpoint": str(ckpt_path), "checkpoint_sha256": sha256_file(Path(ckpt_path)), "weights": ck.get("weights"),
           "epoch": ck.get("epoch"), "arm": cfg.arm, "descriptor": cfg.descriptor, "nc": ck.get("nc"),
           "seed": ck.get("seed"), "dataset": ck.get("dataset"),
           "pending_for_the_device": [
               "exposure discount log(1 - xi_g) of the forecast logit (SPLICE-V section 2): in the float model, "
               "not yet in reference/splicev_ref/forecast.py and the HEAD (FORECAST) unit",
               "full-detector unit programs with int32 biases and shifts (G10)"]}
    src["skipped_native_neck_layers"] = unused
    return write_outputs(Path(out), bundle, floats, inv, calib, manifest, src)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="run")
    a = ap.parse_args()
    rep = export_torch(a.ckpt, a.out, a.run)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
