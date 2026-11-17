"""Calibration on the declared calibration split (never on the validation set).

  band exponents    e_lj = floor(log2(127 / q_{0.9999} |Z_lj / lsb|)), clipped to [-4, 7]
  gate thresholds   Mondrian conformal quantile of d^2 over same-instance fragment pairs,
                    level ceil((n + 1)(1 - delta)) / n per density stratum (pooled when a
                    stratum has fewer than `min_per_stratum` pairs)
  partner prior     frequencies of min(K, Kmax) true partners among admissible partners
  activation scales per Conv block of the backbone/head: power-of-two lsb covering q_{0.9999}
The statistics functions are numpy-only so they are tested here; collect() runs the model.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


def band_exponent(q9999: float, lo: int = -4, hi: int = 7) -> int:
    if q9999 <= 0:
        return hi
    return int(np.clip(math.floor(math.log2(127.0 / q9999)), lo, hi))


def conformal_threshold(d2: np.ndarray, delta: float) -> float:
    """Smallest tau with at most delta false rejections (finite-sample conformal level)."""
    d2 = np.sort(np.asarray(d2, dtype=np.float64))
    n = d2.size
    if n == 0:
        return float("nan")
    k = int(math.ceil((n + 1) * (1.0 - delta)))
    return float(d2[min(k, n) - 1]) if k <= n else float(d2[-1] * 1.0 + 1e-9)


def gate_thresholds(pairs: Dict[int, List[float]], n_strata: int, delta: float, min_per_stratum: int = 200) -> List[float]:
    pooled = [v for s in pairs for v in pairs[s]]
    tau_pool = conformal_threshold(np.asarray(pooled), delta) if pooled else float("nan")
    out = []
    for s in range(n_strata):
        v = pairs.get(s, [])
        out.append(conformal_threshold(np.asarray(v), delta) if len(v) >= min_per_stratum else tau_pool)
    return out


def partner_prior(counts: Sequence[int], kmax: int = 3) -> List[float]:
    c = np.asarray([min(int(k), kmax) for k in counts if k >= 1])
    if c.size == 0:
        return [1.0 / kmax] * kmax
    h = np.array([(c == k).sum() for k in range(1, kmax + 1)], dtype=np.float64) + 0.5   # Jeffreys-style smoothing
    return list(h / h.sum())


def fragment_owner(ids: np.ndarray, owner: np.ndarray, F: int) -> np.ndarray:
    """Majority GT owner of each fragment's cells (-1 when none of its cells is owned)."""
    out = np.full(F, -1, dtype=np.int64)
    for f in range(F):
        o = owner[ids == f]
        o = o[o >= 0]
        if o.size:
            vals, cnt = np.unique(o, return_counts=True)
            out[f] = int(vals[np.argmax(cnt)])
    return out


def pair_statistics(d2: np.ndarray, frag_owner: np.ndarray, adm: np.ndarray):
    """Same-instance pair distances (upper triangle) and true-partner counts per receiver."""
    F = frag_owner.size
    same = (frag_owner[:, None] == frag_owner[None, :]) & (frag_owner[:, None] >= 0)
    np.fill_diagonal(same, False)
    iu = np.triu_indices(F, 1)
    d_same = d2[iu][same[iu]]
    counts = (same & adm).sum(axis=1)
    return d_same, counts


def _q9999(t, cap: int = 1_000_000) -> float:
    """q_{0.9999} of |t| on an evenly strided subsample (torch.quantile has a size limit)."""
    import torch
    a = t.detach().abs().flatten().float()
    a = a[::max(1, a.numel() // cap)]
    return float(torch.quantile(a, 0.9999)) if a.numel() else 0.0


def collect(model, loader, n_batches: int, strides=(8, 16, 32)) -> Dict:
    """Run the model on calibration batches and gather the statistics (torch)."""
    import torch
    from . import losses
    from .targets import ownership
    model.eval()
    pairs: Dict[int, List[float]] = {}
    counts: List[int] = []
    act: Dict[str, List[float]] = {}
    hooks = []
    for name, m in model.yolo.model.named_modules():
        if m.__class__.__name__ == "Conv":
            hooks.append(m.register_forward_hook(lambda _m, _i, o, name=name: act.setdefault(name, []).append(
                _q9999(o))))
    with torch.no_grad():
        for b, batch in enumerate(loader):
            if b >= n_batches:
                break
            img = batch["img"].to(next(model.parameters()).device).float() / 255.0
            boxes = losses.batch_boxes(batch)
            _, nk = model(img, boxes)
            for n, rec in enumerate(nk["frag"]):
                Fn = rec["F"]
                if Fn < 2:
                    continue
                h4, w4 = nk["heads"][1]["e"].shape[-2:]
                own4, _ = ownership(boxes[n], strides[1], h4, w4)
                ids4 = rec["ids"]["P4"].cpu().numpy()
                fo = fragment_owner(ids4, own4.ravel(), Fn)
                mu_f, var_f = rec["mu_f"].cpu().numpy(), rec["var_f"].cpu().numpy()
                diff = mu_f[:, None, :] - mu_f[None, :, :]
                d2 = (diff ** 2 / (var_f[:, None, :] + var_f[None, :, :])).sum(-1)
                d_same, cnt = pair_statistics(d2, fo, rec["adm"].cpu().numpy())
                pairs.setdefault(rec["stratum"], []).extend(d_same.tolist())
                counts.extend(cnt.tolist())
    for h in hooks:
        h.remove()
    return {"pairs": pairs, "counts": counts, "act_q9999": {k: float(np.median(v)) for k, v in act.items()}}


def band_statistics(model, loader, n_batches: int) -> List[List[float]]:
    """q_{0.9999} of |Z_lj| / transport_lsb per level and band (code units before 2^e)."""
    import torch
    from . import ops
    from .backend import TorchBackend
    neck = model.neck
    B = TorchBackend()
    vals = [[[] for _ in range(4)] for _ in range(3)]
    model.eval()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            if b >= n_batches:
                break
            img = batch["img"].to(next(model.parameters()).device).float() / 255.0
            X = [p(f) for p, f in zip(neck.proj, model.backbone(img))]
            for l in range(3):
                h = neck.heads[l](X[l])
                Z, _ = ops.bands(B, ops.walk_powers(B, X[l], ops.edge_weights(B, h["q"], h["pi"]), neck.cfg.steps[l]),
                                 ops.beta_chain(B, h["beta_u"]))
                for j in range(4):
                    vals[l][j].append(_q9999(Z[j] / neck.cfg.transport_lsb))
    return [[float(np.median(v)) if v else 0.0 for v in row] for row in vals]


def sensitivity(model, loader, n_batches: int) -> Dict:
    """Measured sensitivity of the neck after (4) to fragment-constant probes (section 3, 'The
    deferral'): the within-object part of the response, relative to the response and to the
    probe, sqrt(sum ||Q dN||^2 / sum ||dN||^2) and sqrt(sum ||Q dN||^2 / sum ||M v||^2)."""
    import torch
    neck = model.neck
    q = full = probe = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            if b >= n_batches:
                break
            img = batch["img"].to(next(model.parameters()).device).float() / 255.0
            _, nk = model(img, None, False, True)
            tot, fl, pr, _cnt = neck.probe_response(nk["Xp"], nk["W"], nk["frag"], nk["heads"], nk["N"])
            q, full, probe, n = q + float(tot), full + float(fl), probe + float(pr), n + 1
    return {"q_over_response": math.sqrt(q / full) if full > 0 else None,
            "q_over_probe": math.sqrt(q / probe) if probe > 0 else None, "batches": n,
            "probe_sigma_lsb": neck.cfg.probe_sigma_lsb}


def apply(model, stats: Dict, bands: List[List[float]], delta: float, strata: int = 4) -> Dict:
    """Write the calibrated values into the model buffers; returns the record for export."""
    import torch
    neck = model.neck
    exps = [[band_exponent(q) for q in row] for row in bands]
    neck.band_exp.copy_(torch.tensor(exps, dtype=torch.float32))
    taus = gate_thresholds(stats["pairs"], strata, delta)
    taus = [t if t == t else float(neck.gate_tau[i]) for i, t in enumerate(taus)]     # keep defaults for empty strata
    neck.gate_tau.copy_(torch.tensor(taus, dtype=torch.float32))
    prior = partner_prior(stats["counts"], neck.cfg.kmax)
    neck.prior.copy_(torch.tensor(prior, dtype=torch.float32))
    act = {k: 2.0 ** -int(math.floor(math.log2(127.0 / max(v, 1e-6)))) for k, v in stats.get("act_q9999", {}).items()}
    return {"band_exponents": exps, "band_q9999": bands, "gate_tau": taus, "gate_pairs": {int(k): len(v) for k, v in stats["pairs"].items()},
            "prior": prior, "partner_rows": len(stats["counts"]), "act_lsb": act, "delta": delta}
