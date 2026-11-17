"""Training on synthetic scenes, CPU only (numpy): what can be trained without a GPU or a dataset.

    python -m splice_det.toytrain --out artifacts/trained/synthetic_arm_e

This trains the learned maps of arm (e) that act on fragment states, against the exact
integer exchange of reference/splicev_ref, and calibrates the manifest's synthetic values on
held-out scenes.  It qualifies the training -> calibration -> export -> integer -> device
chain end to end; it is not a detector trained on images (the scenes have no image content:
features are prototypes plus noise, so box hypotheses are not learnable and stay synthetic).

  forecast maps  W_K, U, u0, u_alpha, O_F   minimise mean (dHhat - sg(dH))^2 (SPLICE-V, mu-term)
                 with dH the exact exchange output (its own maps O_T, W_S are fixed);
                 forward and gradients follow the integer pipeline's functional form
  class head     w_cls and the residual map Z: logistic loss on cell ownership with
                 l = (Pi N + M r Z) w_cls (the co-designed readout, Eq. (5))
  calibration    gate thresholds (Mondrian conformal, delta = 0.01), partner prior, band
                 exponents, on calibration scenes disjoint from training and test scenes
  export         integer heads (int8), calibration, manifest.model.json, report with the
                 before/after residual statistics and the six-boundary check of the slice
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import REPO
from splicev_ref import gen, toy
from splicev_ref.arith import div_rne
from splicev_ref.membership import UNASSIGNED, level_weights, restrict
from splicev_ref.walk import band_accumulators, export_taps

from . import calibrate as cal
from . import export as ex
from . import ops
from .backend import NumpyBackend

B = NumpyBackend()
TRAINED = ("W_K", "W_U", "u0", "ua", "O_F")
LOGIT_LO, LOGIT_HI = -8.0, 127.0 / 16.0
EPS = 2.0 ** -4


def shared_heads(seed: int = 7) -> Dict[str, np.ndarray]:
    """One model: the forecast maps (to be trained) and the exchange's own maps (fixed)."""
    h = dict(toy.make_scene(seed)["heads"])
    h["O_T"] = h["O_F"].copy()           # the exchange's translation map (fixed "pretrained" map)
    h["W_S"] = h["W_K"].copy()           # the symmetric evidence scorer (fixed)
    return h


def scene_with(seed: int, heads: Dict) -> Dict:
    sc = toy.make_scene(seed)
    sc["heads"] = {k: np.asarray(v).copy() for k, v in heads.items()}
    return sc


# --------------------------------------------------------------------------- forecast model

def forecast_forward(p: Dict, H, nu, adm):
    """Float form of the integer forecast of the toy pipeline (same scales and clamps)."""
    Kf = H @ p["W_K"] / 128.0
    Uf = nu @ p["W_U"] / 64.0
    VO = H @ p["O_F"] / 128.0
    ck = math.sqrt(8.0 * gen.FC_KK_MULT / 2.0 ** gen.FC_SHIFT / 16.0)
    cu = math.sqrt(gen.FC_UU_MULT / 2.0 ** gen.FC_SHIFT / 16.0)
    c0 = gen.FC_NULL_MULT / 2.0 ** gen.FC_NULL_SHIFT / 16.0
    ca = gen.FC_ALPHA_MULT / 2.0 ** gen.FC_ALPHA_SHIFT / 256.0
    l0 = nu @ p["u0"] * c0
    z = nu @ p["ua"] * ca
    Hs = H[:, :gen.C_TRANSPORT]
    Kq, Uq, VOc = np.clip(Kf, -127, 127), np.clip(Uf, -127, 127), np.clip(VO, -127, 127)
    L = (Kq * ck) @ (Kq * ck).T / 8.0 + (Uq * cu) @ (Uq * cu).T
    Lc = np.clip(L, LOGIT_LO, LOGIT_HI)
    l0c = np.clip(l0, LOGIT_LO, LOGIT_HI)
    dh, a, a0, alpha = ops.forecast(B, Kq * ck, Uq * cu, VOc, Hs.astype(np.float64), l0c, z, adm, d=64,
                                    clip_b=float(gen.FC_B), eps=EPS)
    cache = dict(Kf=Kf, Uf=Uf, VO=VO, Kq=Kq, Uq=Uq, VOc=VOc, L=L, l0=l0, z=z, a=a, a0=a0, alpha=alpha, Hs=Hs,
                 ck=ck, cu=cu, c0=c0, ca=ca, H=H, nu=nu, adm=adm)
    return dh, cache


def forecast_backward(p: Dict, G: np.ndarray, c: Dict) -> Dict[str, np.ndarray]:
    """Gradients of sum(G * dHhat) w.r.t. the five maps (checked by finite differences)."""
    a, a0, alpha, adm = c["a"], c["a0"], c["alpha"], c["adm"]
    innov_raw = c["VOc"][None, :, :] - c["Hs"][:, None, :]
    innov = np.clip(innov_raw, -gen.FC_B, gen.FC_B)
    S = np.einsum("fg,fgc->fc", a, innov)
    sig = alpha / (1.0 - EPS)
    dz = (G * S).sum(1) * (1.0 - EPS) * sig * (1.0 - sig)
    dS = alpha[:, None] * G
    A = np.einsum("fc,fgc->fg", dS, innov)
    dI = a[:, :, None] * dS[:, None, :] * (np.abs(innov_raw) < gen.FC_B)
    dVOc = dI.sum(0)
    t = (a * A).sum(1)
    dL = a * (A - t[:, None])
    dl0 = -a0 * t
    dL = dL * ((c["L"] > LOGIT_LO) & (c["L"] < LOGIT_HI)) * adm
    dl0 = dl0 * ((c["l0"] > LOGIT_LO) & (c["l0"] < LOGIT_HI))
    ck, cu = c["ck"], c["cu"]
    Ks, Us = c["Kq"] * ck, c["Uq"] * cu
    dKs = (dL + dL.T) @ Ks / 8.0
    dUs = (dL + dL.T) @ Us
    dKf = dKs * ck * (np.abs(c["Kf"]) < 127)
    dUf = dUs * cu * (np.abs(c["Uf"]) < 127)
    dVO = dVOc * (np.abs(c["VO"]) < 127)
    return {"W_K": c["H"].T @ dKf / 128.0, "W_U": c["nu"].T @ dUf / 64.0, "u0": c["nu"].T @ dl0 * c["c0"],
            "ua": c["nu"].T @ dz * c["ca"], "O_F": c["H"].T @ dVO / 128.0}


class Adam:
    def __init__(self, params: Dict, lr: float, b1: float = 0.9, b2: float = 0.999):
        self.m = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.v = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0

    def step(self, params: Dict, grads: Dict, lim: float = 127.0):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = np.clip(params[k] - self.lr * mh / (np.sqrt(vh) + 1e-8), -lim, lim)   # int8 range


# --------------------------------------------------------------------------- data

def traces(seeds: List[int], heads: Dict) -> List[Dict]:
    out = []
    for s in seeds:
        tr = toy.run(scene_with(s, heads))
        if tr["F"] < 2:
            continue
        out.append({"seed": s, "H": tr["H"].astype(np.float64), "nu": tr["nu"].astype(np.float64), "adm": tr["adm"],
                    "dH": tr["dV"].astype(np.float64), "F": tr["F"]})
    return out


def residual_stats(seeds: List[int], heads: Dict) -> Dict:
    """Residual r = dH - dHhat on rows with admissible donors (integer pipeline); mse_dH is
    the residual a zero forecast would leave."""
    rs, dhs, rows_adm, abst = [], [], 0, 0
    for s in seeds:
        tr = toy.run(scene_with(s, heads))
        if tr["F"] == 0:
            continue
        m = tr["adm"].any(1)
        rs.append(tr["r"][m])
        dhs.append(tr["dH"][m])
        rows_adm += int(m.sum())
        abst += int(tr["abstain"].sum())
    r = np.concatenate(rs) if rs else np.zeros((0, gen.C_TRANSPORT))
    dh = np.concatenate(dhs) if dhs else np.zeros((0, gen.C_TRANSPORT))
    return {"rows": rows_adm, "mean_abs_r": float(np.abs(r).mean()), "mse_r": float((r ** 2).mean()),
            "mse_dH_zero_forecast": float((dh ** 2).mean()),
            "frac_abs_r_ge_16": float((np.abs(r) >= 16).mean()), "max_abs_r": int(np.abs(r).max()) if r.size else 0,
            "abstentions": abst}


def train_forecast(train: List[Dict], val: List[Dict], init: Dict, epochs: int, lr: float, log) -> Dict:
    p = {k: np.asarray(init[k], dtype=np.float64).copy() for k in TRAINED}
    opt = Adam(p, lr)
    rng = np.random.default_rng(0)

    def mse(data):
        num, den = 0.0, 0
        for d in data:
            dh, _ = forecast_forward(p, d["H"], d["nu"], d["adm"])
            num += float(((dh - d["dH"]) ** 2).sum())
            den += d["dH"].size
        return num / max(den, 1)

    log({"stage": "forecast", "epoch": 0, "train_mse": mse(train), "val_mse": mse(val)})
    best, best_val = {k: v.copy() for k, v in p.items()}, mse(val)
    for ep in range(1, epochs + 1):
        opt.lr = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (ep - 1) / epochs)))
        for i in rng.permutation(len(train)):
            d = train[i]
            dh, c = forecast_forward(p, d["H"], d["nu"], d["adm"])
            G = 2.0 * (dh - d["dH"]) / d["dH"].size
            opt.step(p, forecast_backward(p, G, c))
        if ep % 10 == 0 or ep == epochs:
            v = mse(val)
            log({"stage": "forecast", "epoch": ep, "train_mse": mse(train), "val_mse": v, "lr": opt.lr})
            if v < best_val:
                best, best_val = {k: x.copy() for k, x in p.items()}, v
    log({"stage": "forecast_selected", "val_mse": best_val})
    return {k: np.round(v).astype(np.int64) for k, v in best.items()}


# --------------------------------------------------------------------------- class head

def readout_data(seeds: List[int], heads: Dict) -> List[Dict]:
    out = []
    for s in seeds:
        sc = scene_with(s, heads)
        tr = toy.run(sc)
        if tr["F"] == 0:
            continue
        geo = sc["geo"]
        L4 = sc["levels"]["P4"]
        n4 = geo.h4 * geo.w4
        x4 = L4["x"].reshape(n4, geo.channels)
        xs = np.clip(x4[:, :gen.C_TRANSPORT] + ops.fill(B, tr["dh_hat"].astype(float), np.where(tr["ids"]["P4"] == UNASSIGNED, -1, tr["ids"]["P4"])), -128, 127)
        n_feat = np.concatenate([xs, x4[:, gen.C_TRANSPORT:]], 1).astype(np.int64)
        ids4 = tr["ids"]["P4"]
        w4 = level_weights(L4["e"].ravel(), ids4, tr["F"])
        pin = restrict(n_feat, ids4, w4, tr["F"])
        out.append({"pin": pin.astype(np.float64), "ids": np.where(ids4 == UNASSIGNED, -1, ids4), "r": tr["r"].astype(np.float64),
                    "y": (L4["own"].ravel() >= 0).astype(np.float64), "F": tr["F"]})
    return out


def train_class(data: List[Dict], init_w: np.ndarray, init_Z: np.ndarray, epochs: int, lr: float, log) -> Tuple[np.ndarray, np.ndarray]:
    p = {"w": init_w.astype(np.float64).copy(), "Z": init_Z.astype(np.float64).copy()}
    opt = Adam(p, lr)
    rz_s = 2.0 ** gen.RZ_SHIFT

    def forward(d):
        rz = np.clip(d["r"] @ p["Z"] / rz_s, -128, 127)
        cin = np.clip(d["pin"] + rz, -128, 127)
        rows = ops.fill(B, cin, d["ids"])
        logit = rows @ p["w"] / 4.0 / 256.0                       # Q8.8 logit code -> natural units
        return logit, rows, rz

    def loss_acc(dd):
        tot, n, correct = 0.0, 0, 0
        for d in dd:
            lg, _, _ = forward(d)
            m = d["ids"] >= 0
            pr = 1 / (1 + np.exp(-lg[m]))
            y = d["y"][m]
            tot += float(-(y * np.log(pr + 1e-9) + (1 - y) * np.log(1 - pr + 1e-9)).sum())
            correct += int(((pr >= 0.5) == (y > 0.5)).sum())
            n += int(m.sum())
        return tot / max(n, 1), correct / max(n, 1)

    l, acc = loss_acc(data)
    log({"stage": "class", "epoch": 0, "bce": l, "acc_assigned": acc})
    for ep in range(1, epochs + 1):
        for d in data:
            lg, rows, rz = forward(d)
            m = (d["ids"] >= 0).astype(np.float64)
            g = (1 / (1 + np.exp(-lg)) - d["y"]) * m / max(m.sum(), 1)
            dw = rows.T @ g / 4.0 / 256.0
            drows = np.outer(g, p["w"]) / 4.0 / 256.0
            dcin = B.scatter_add(d["F"] + 1, np.where(d["ids"] >= 0, d["ids"], d["F"]), drows)[:d["F"]]
            cin = d["pin"] + rz
            drz = dcin * (np.abs(cin) < 127) * (np.abs(rz) < 127)
            dZ = d["r"].T @ drz / rz_s
            opt.step(p, {"w": dw, "Z": dZ})
        if ep % 10 == 0 or ep == epochs:
            l, acc = loss_acc(data)
            log({"stage": "class", "epoch": ep, "bce": l, "acc_assigned": acc})
    return np.round(p["w"]).astype(np.int64), np.round(p["Z"]).astype(np.int64)


# --------------------------------------------------------------------------- calibration

def calibrate_scenes(seeds: List[int], heads: Dict, delta: float) -> Dict:
    pairs: Dict[int, List[float]] = {}
    counts: List[int] = []
    band_abs = {name: [[] for _ in range(5)] for name in ("P3", "P4", "P5")}
    for s in seeds:
        sc = scene_with(s, heads)
        tr = toy.run(sc)
        for name in ("P3", "P4", "P5"):
            L = sc["levels"][name]
            acc = band_accumulators(L["x"], export_taps(L["w"], L["inside"]), L["betas"], gen.LEVELS[name]["steps"])
            for j in range(5):
                band_abs[name][j].append(float(np.quantile(np.abs(acc[j]), 0.9999)))
        F = tr["F"]
        if F < 2:
            continue
        geo = sc["geo"]
        own_anchor = np.concatenate([sc["levels"]["P4"]["obj"].ravel(), sc["levels"]["P3"]["obj"].ravel()[tr["reserve"]]])
        ids = np.where(tr["ids_anchor"] == UNASSIGNED, -1, tr["ids_anchor"])
        fo = cal.fragment_owner(ids, own_anchor, F)
        prec = tr["hyp_prec"].astype(np.float64) / 2.0 ** gen.GATE_PREC_FRAC      # natural units (1 / code^2)
        mu_f, var_f = ops.fuse_hypotheses(B, tr["hyp_mu"].astype(np.float64), prec, ids, F)
        d2 = ops.mahalanobis2(B, mu_f, var_f)                                      # the manifest's threshold units
        d_same, cnt = cal.pair_statistics(d2, fo, tr["adm"])
        pairs.setdefault(ops.gate_stratum(F, [gen.GATE_STRATUM_BOUND_0, gen.GATE_STRATUM_BOUND_1, gen.GATE_STRATUM_BOUND_2]), []).extend(d_same.tolist())
        counts.extend(cnt.tolist())
    taus = cal.gate_thresholds(pairs, gen.GATE_N_STRATA, delta, min_per_stratum=50)
    taus = [float(t) if t == t else float(gen.GATE_TAU_BF[i][0] * 2.0 ** gen.GATE_TAU_BF[i][1] / 2.0 ** gen.GATE_PREC_FRAC)
            for i, t in enumerate(taus)]
    exps = []
    for name in ("P3", "P4", "P5"):
        row = []
        for j in range(4):
            q = float(np.median(band_abs[name][j]))
            row.append(int(np.clip(math.floor(math.log2(127.0 * 2 ** 23 / max(q, 1.0))), -4, 7)))
        exps.append(row)
    q_low = float(np.median(band_abs["P4"][4]))
    return {"gate_tau": taus, "gate_pairs": {int(k): len(v) for k, v in pairs.items()}, "delta": delta,
            "prior": cal.partner_prior(counts, gen.KMAX), "partner_rows": len(counts), "band_exponents": exps,
            "lowpass_exponent": int(np.clip(math.floor(math.log2(127.0 * 2 ** 23 / max(q_low, 1.0))), -4, 7)),
            "scenes": len(seeds), "data": "synthetic scenes (reference/splicev_ref/toy.py)"}


# --------------------------------------------------------------------------- main

def slice_check(seeds: List[int], heads: Dict) -> Dict:
    import sys
    sys.path.insert(0, str(REPO / "ps"))
    from splicev_ps.slice_runner import RefBackend, compare, run_slice
    ok = []
    for s in seeds:
        want = toy.run(scene_with(s, heads))
        got = run_slice(scene_with(s, heads), RefBackend([0x03, 0x44, 0xB8]))
        ok.append(bool(compare(got, want)["all_zero_mismatch"]))
    return {"scenes": len(seeds), "bit_exact": int(sum(ok)), "all": all(ok)}


def phase_calibrate(out: Path, seeds: Dict, log) -> Dict:
    """Calibration on held-out scenes (independent of the trained maps: it uses the
    hypotheses, the features and the affinities), then the model manifest and its generated
    sources, so that training runs under the calibrated contract."""
    import subprocess
    import sys
    calib = calibrate_scenes(seeds["calib"], shared_heads(), 0.01)
    log({"stage": "calibrate", **{k: v for k, v in calib.items() if k != "data"}})
    mults = {"kk_mult": gen.FC_KK_MULT, "uu_mult": gen.FC_UU_MULT, "shift": gen.FC_SHIFT, "null_mult": gen.FC_NULL_MULT,
             "null_shift": gen.FC_NULL_SHIFT, "alpha_mult": gen.FC_ALPHA_MULT, "alpha_shift": gen.FC_ALPHA_SHIFT}
    cfg = {"a_r": gen.A_R_Q39 / 2.0 ** 39, "b_r": gen.B_R_Q39 / 2.0 ** 39, "c_r": gen.C_R_Q39 / 2.0 ** 39, "u_dim": gen.FC_UDIM,
           "gate_strata": (gen.GATE_STRATUM_BOUND_0, gen.GATE_STRATUM_BOUND_1, gen.GATE_STRATUM_BOUND_2)}
    ov, dec = ex.calibrated_overrides(calib, cfg, mults, gen.RZ_SHIFT, None, provenance="S")
    manifest = ex.model_manifest("synthetic-trained", ov, dec,
                                 "Calibrated and trained on synthetic scenes only (pipeline qualification); not a detector "
                                 "trained on images. Non-deployable.", deployable=False, out_dir=out)
    (out / "manifest.model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([sys.executable, str(REPO / "tools" / "generate_sources.py"), "--manifest", str(out / "manifest.model.json"),
                    "--out", str(out / "generated")], check=True, capture_output=True)
    return calib


def phase_train(out: Path, seeds: Dict, a, log) -> Dict:
    """Training and export under the calibrated contract (run with SPLICEV_GEN_DIR set)."""
    t0 = time.time()
    calib = json.loads((out / "calibration_phase1.json").read_text())
    base = shared_heads()
    tr_data, va_data = traces(seeds["train"], base), traces(seeds["val"], base)
    log({"stage": "data", "profile": gen.MANIFEST_ID, "train_scenes": len(tr_data), "val_scenes": len(va_data),
         "train_rows": int(sum(d["F"] for d in tr_data))})
    before = residual_stats(seeds["test"], base)
    log({"stage": "baseline_test", **before})
    fc = train_forecast(tr_data, va_data, base, a.epochs, a.lr, log)
    heads = dict(base)
    heads.update(fc)
    after = residual_stats(seeds["test"], heads)
    log({"stage": "trained_test", **after})
    rd = readout_data(seeds["train"], heads)
    w, Z = train_class(rd, base["w_cls"], base["Z"] * 0, a.class_epochs, 0.5, log)
    heads["w_cls"], heads["Z"] = w, Z
    check = slice_check(seeds["test"][:4], heads)
    log({"stage": "slice_check", **check})
    from splicev_ref import svb
    bundle = svb.Bundle()
    for k in sorted(heads):
        bundle.add(f"heads.{k}", np.asarray(heads[k], dtype=np.int64))
    manifest = json.loads((out / "manifest.model.json").read_text())
    inventory = [{"name": f"heads.{k}", "op": "gemm weights" if np.asarray(heads[k]).ndim == 2 else "vector",
                  "shape": list(np.asarray(heads[k]).shape), "trained": k in TRAINED + ("w_cls", "Z")} for k in sorted(heads)]
    floats = {f"init.{k}": np.asarray(base[k], dtype=np.float64) for k in TRAINED}
    rep = ex.write_outputs(out, bundle, floats, inventory, calib, manifest,
                           {"generator": "splice_det.toytrain", "profile": gen.MANIFEST_ID, "profile_sha256": gen.MANIFEST_SHA256,
                            "seeds": {k: [v[0], v[-1]] for k, v in seeds.items()}})
    (out / "heads.json").write_text(json.dumps({k: np.asarray(v).tolist() for k, v in heads.items()}) + "\n")
    return {"residual_test_before": before, "residual_test_after": after, "slice_check": check, "export": rep,
            "seconds_train": round(time.time() - t0, 1)}


def main() -> int:
    import os
    import subprocess
    import sys
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "trained" / "synthetic_arm_e"))
    ap.add_argument("--phase", choices=["all", "train"], default="all")
    ap.add_argument("--train", type=int, default=96)
    ap.add_argument("--val", type=int, default=24)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--test", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--class-epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = {"train": list(range(1000, 1000 + a.train)), "val": list(range(3000, 3000 + a.val)),
             "calib": list(range(5000, 5000 + a.calib)), "test": list(range(9000, 9000 + a.test))}
    log_path = out / "train_log.jsonl"

    def log(rec):
        rec["time"] = time.strftime("%H:%M:%S")
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    if a.phase == "all":
        if log_path.exists():
            log_path.unlink()
        t0 = time.time()
        calib = phase_calibrate(out, seeds, log)
        (out / "calibration_phase1.json").write_text(json.dumps(calib, indent=2) + "\n")
        env = dict(os.environ, SPLICEV_GEN_DIR=str(out / "generated" / "py"))
        argv = [sys.executable, "-m", "splice_det.toytrain", "--phase", "train", "--out", str(out)] + \
            [f"--{k.replace('_', '-')}={v}" for k, v in vars(a).items() if k not in ("out", "phase")]
        subprocess.run(argv, check=True, env=env, cwd=str(REPO / "detector"))
        rep = json.loads((out / "report_phase2.json").read_text())
        rep.update({"calibration": calib, "seconds": round(time.time() - t0, 1),
                    "note": "synthetic scenes; provenance S; the image-trained model comes from splice_det.train on the HPC"})
        (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
        (out / "report_phase2.json").unlink()
        (out / "calibration_phase1.json").unlink()
        print(json.dumps({"out": str(out), "mse_r_before": rep["residual_test_before"]["mse_r"],
                          "mse_r_after": rep["residual_test_after"]["mse_r"], "slice_bit_exact": rep["slice_check"]["all"]}))
        return 0
    rep = phase_train(out, seeds, a, log)
    (out / "report_phase2.json").write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
