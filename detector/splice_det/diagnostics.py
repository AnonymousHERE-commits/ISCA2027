"""Train-channel quantities of SPLICE-V (Tables 1, 5, 7 and section 3), measured on images
with ground truth (the calibration split or the validation set):

  forecast      median relative error ||dHhat - dH|| / ||dH|| ('err.'), share of the forecast's
                transport that crosses instances ('cross'), abstention and clip shares, the
                exposure budget xi
  admissibility false-rejection rate of the test per density stratum (target delta = 0.01)
  regime        fragment count F, candidates per row N_f and N_f / F, cap and budget exits
  Lemma 3       kappa_d (largest one-step probability of leaving one's class) and the exact
                lumpability defect ||Xi||_inf per level, on a subset of images
  leakage       the largest response of the typed low-pass and of the bands to another
                instance's indicator, per level, on the same subset
  deferral      measured sensitivity to fragment-constant probes (calibrate.sensitivity)

The lumpability and leakage functions are numpy and unit-tested; `run` drives the model.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import ops
from .backend import NumpyBackend
from .calibrate import fragment_owner, pair_statistics
from .targets import ownership


def lumpability(w: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Exact kappa_d and ||Xi||_inf of the lazy walk of one level (Lemma 3).

    w (8, H, W) edge weights, ids (H W,) fragment id per cell (-1 unassigned), weights (H W,)
    the restriction weights of the fragment cells.  Classes: the fragments and the set of
    unassigned cells (uniform weights there)."""
    w = np.asarray(w, dtype=np.float64)
    H, W = w.shape[1:]
    n = H * W
    ids = np.asarray(ids, dtype=np.int64).ravel()
    F = int(ids.max()) + 1 if ids.size and ids.max() >= 0 else 0
    cls = np.where(ids >= 0, ids, F)
    wsum = w.sum(0).ravel()
    T = np.zeros((n, F + 1))
    cells = np.arange(n)
    T[cells, cls] += 0.5 + 0.5 / (1.0 + wsum)
    grid = cells.reshape(H, W)
    for k, (dy, dx) in enumerate(ops.NEIGH):
        src = grid[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)].ravel()
        dst = grid[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)].ravel()
        np.add.at(T, (src, cls[dst]), 0.5 * w[k].ravel()[src] / (1.0 + wsum[src]))
    kappa = 1.0 - T[cells, cls]
    wt = np.where(ids >= 0, np.asarray(weights, dtype=np.float64).ravel(), 1.0)
    mass = np.zeros(F + 1)
    np.add.at(mass, cls, wt)
    zero = mass[cls] <= 0                                   # a class without mass: uniform weights
    wt = np.where(zero, 1.0, wt)
    mass = np.zeros(F + 1)
    np.add.at(mass, cls, wt)
    Pc = np.zeros((F + 1, F + 1))
    np.add.at(Pc, cls, T * wt[:, None])
    Pc = Pc / np.maximum(mass[:, None], 1e-300)
    xi = np.abs(T - Pc[cls]).sum(1)
    return {"kappa": float(kappa.max()) if n else 0.0, "kappa_mean": float(kappa.mean()) if n else 0.0,
            "xi_inf": float(xi.max()) if n else 0.0, "classes": F + 1}


def leakage(w: np.ndarray, betas: np.ndarray, owner: np.ndarray, steps: int, max_instances: int = 32) -> Dict[str, float]:
    """Largest response of the typed filters to another instance's indicator.

    w (8, H, W), betas (4, H, W) the per-cell chain, owner (H, W) instance of every cell (-1
    none).  For each instance U the indicator 1_U is filtered; the leakage is the largest
    value on cells owned by another instance: for the low-pass A(beta_4) and for the bands."""
    B = NumpyBackend()
    inst, cnt = np.unique(owner[owner >= 0], return_counts=True)
    inst = inst[np.argsort(-cnt)][:max_instances]
    if inst.size < 2:
        return {"lowpass": None, "band": None, "instances": int(inst.size)}
    ind = np.stack([(owner == u).astype(np.float64) for u in inst])[None]            # (1, K, H, W)
    powers = ops.walk_powers(B, ind, np.asarray(w, dtype=np.float64)[None], steps)
    Z, low = ops.bands(B, powers, np.asarray(betas, dtype=np.float64)[None])
    other = np.stack([(owner >= 0) & (owner != u) for u in inst])
    lo = float(np.where(other, low[0], 0.0).max())
    bd = float(max(np.where(other, np.abs(z[0]), 0.0).max() for z in Z))
    return {"lowpass": lo, "band": bd, "instances": int(inst.size)}


def _quantiles(x: Sequence[float]) -> Optional[Dict[str, float]]:
    a = np.asarray(list(x), dtype=np.float64)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
            "p99": float(np.quantile(a, 0.99)), "max": float(a.max())}


def run(model, loader, n_batches: int, exact_images: int = 200, log=print) -> Dict:
    """Drive the model over `n_batches` batches with ground-truth boxes."""
    import torch
    from . import calibrate, losses
    neck = model.neck
    cfg = neck.cfg
    strides = neck.strides
    dev = next(model.parameters()).device
    B = ops
    err, xi_all, part, part_ratio, Fs = [], [], [], [], []
    cross_num = cross_den = abst = n_frag = clip_num = clip_den = 0.0
    frr: Dict[int, List[int]] = {}
    cap = budget = frames = 0
    lump = {l: [] for l in range(3)}
    leak = {l: [] for l in range(3)}
    exact_done = 0
    neck.diag_exchange = True
    model.eval()
    try:
        with torch.no_grad():
            for b, batch in enumerate(loader):
                if b >= n_batches:
                    break
                img = batch["img"].to(dev).float() / 255.0
                boxes = losses.batch_boxes(batch)
                _, nk = model(img, boxes, False, True)
                for n, rec in enumerate(nk["frag"]):
                    frames += 1
                    Fn = rec["F"]
                    Fs.append(Fn)
                    cap += int(rec["link"]["overflow"] or Fn >= cfg.f_cap)
                    budget += int(rec["link"]["budget_hit"])
                    h4, w4 = nk["heads"][1]["e"].shape[-2:]
                    own4, _ = ownership(boxes[n], strides[1], h4, w4)
                    if Fn:
                        adm = rec["adm"].cpu().numpy()
                        cnt = adm.sum(1)
                        part.extend(cnt.tolist())
                        part_ratio.append(float(cnt.mean()) / Fn)
                        n_frag += Fn
                        fo = fragment_owner(rec["ids"]["P4"].cpu().numpy(), own4.ravel(), Fn)
                        if Fn >= 2:
                            mu_f, var_f = rec["mu_f"].cpu().numpy(), rec["var_f"].cpu().numpy()
                            diff = mu_f[:, None, :] - mu_f[None, :, :]
                            d2 = (diff ** 2 / (var_f[:, None, :] + var_f[None, :, :])).sum(-1)
                            d_same, _ = pair_statistics(d2, fo, adm)
                            tau = float(neck.gate_tau[rec["stratum"]])
                            acc = frr.setdefault(int(rec["stratum"]), [0, 0])
                            acc[0] += int((d_same > tau).sum())
                            acc[1] += int(d_same.size)
                        if rec.get("dh_hat") is not None and rec.get("dH") is not None:
                            dh, dH = rec["dh_hat"].double().cpu().numpy(), rec["dH"].double().cpu().numpy()
                            nrm = np.linalg.norm(dH, axis=1)
                            ok = nrm > cfg.transport_lsb
                            err.extend((np.linalg.norm(dh - dH, axis=1)[ok] / nrm[ok]).tolist())
                        if "transport_fg" in rec:
                            T = rec["transport_fg"].double().cpu().numpy()
                            crossing = (fo[:, None] >= 0) & (fo[None, :] >= 0) & (fo[:, None] != fo[None, :])
                            cross_num += float(T[crossing].sum())
                            cross_den += float(T.sum())
                        if "a0" in rec:
                            a0 = rec["a0"].cpu().numpy()
                            abst += float(((cnt == 0) | (a0 >= 0.5)).sum())
                            clip_num += float(rec["clip_fg"].sum())
                            clip_den += float(rec["y"].sum())
                        if "xi" in rec:
                            xi_all.extend(rec["xi"].cpu().numpy().tolist())
                    if exact_done < exact_images:
                        exact_done += 1
                        for l in range(3):
                            hl, wl = nk["heads"][l]["e"].shape[-2:]
                            ids = rec["ids"][f"P{l + 3}"].cpu().numpy()
                            wmap = nk["Wd"][l][n].double().cpu().numpy()
                            e = np.round(nk["heads"][l]["e"][n, 0].double().cpu().numpy().ravel() * 255.0)
                            lump[l].append(lumpability(wmap, ids, e))
                            own, _ = ownership(boxes[n], strides[l], hl, wl)
                            betas = B.beta_chain(NumpyBackend(), nk["heads"][l]["beta_u"][n:n + 1].double().cpu().numpy())[0]
                            leak[l].append(leakage(nk["W"][l][n].double().cpu().numpy(), betas, own, cfg.steps[l]))
                if b % 10 == 0:
                    log({"stage": "diagnostics", "batch": b, "frames": frames})
        sens = calibrate.sensitivity(model, loader, min(n_batches, 25))
    finally:
        neck.diag_exchange = False

    def agg(rows: List[Dict], key: str):
        v = [r[key] for r in rows if r.get(key) is not None]
        return {"max": float(np.max(v)), "mean": float(np.mean(v)), "min": float(np.min(v)), "n": len(v)} if v else None

    Fa = np.asarray(Fs, dtype=np.float64)
    pos = Fa[Fa > 0]
    return {
        "frames": frames, "arm": cfg.arm, "descriptor": cfg.descriptor,
        "forecast": {"err_median": float(np.median(err)) if err else None, "err": _quantiles(err),
                     "cross_share": cross_num / cross_den if cross_den > 0 else None,
                     "abstain_share": abst / n_frag if (n_frag and cfg.arm_spec["forecast"] == "confined") else None,
                     "clip_share": clip_num / clip_den if clip_den > 0 else None, "xi": _quantiles(xi_all)},
        "admissibility": {"delta": cfg.gate_delta, "false_rejection_rate": {
            str(s): {"rejected": v[0], "pairs": v[1], "rate": v[0] / v[1] if v[1] else None} for s, v in sorted(frr.items())}},
        "regime": {"F": _quantiles(Fs), "F_log_std": float(np.log(pos).std()) if pos.size > 1 else None,
                   "F_cv": float(pos.std() / pos.mean()) if pos.size > 1 else None,
                   "N_f": _quantiles(part), "N_over_F_mean": float(np.mean(part_ratio)) if part_ratio else None,
                   "cap_frames_share": cap / frames if frames else None, "budget_exit_share": budget / frames if frames else None},
        "lemma3": {f"P{l + 3}": {"kappa_d": agg(lump[l], "kappa"), "xi_inf": agg(lump[l], "xi_inf")} for l in range(3)},
        "leakage": {f"P{l + 3}": {"lowpass": agg(leak[l], "lowpass"), "band": agg(leak[l], "band")} for l in range(3)},
        "sensitivity": sens, "exact_images": exact_done,
    }
