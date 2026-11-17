"""Autograd self-checks, run with real PyTorch by the suite's semantic preflight (CPU, seconds).

  exchange   gradients of the differentiable exchange (Eq. (1) marginals, optimality system
             of Eq. (2) on fixed active sets) against central finite differences in float64
  arms       one training step of the SPLICE neck in every readout arm: finite gradients, and
             the gradient reaches exactly the parameters the arm trains (the exchange's own
             maps in arms a-e, not in f and g; the forecast's maps in d, e, f)

The unit tests exercise the same code forward-only on a numpy stand-in when torch is
missing; these checks are the part that needs autograd.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def check_exchange(seed: int = 0, tol: float = 1e-5) -> Dict:
    import torch
    from . import exchange_float as xf, exchange_torch as xt
    rng = np.random.default_rng(seed)
    F, Ct = 14, 5
    X = rng.normal(0, 1, (F, 2)) * 2
    adm = (((X[:, None] - X[None]) ** 2).sum(-1) < 6.0) & ~np.eye(F, dtype=bool)
    S = rng.normal(0, 1, (F, 6))
    val = {"ev": (S @ S.T) / np.sqrt(6), "need": rng.uniform(0.2, 0.9, F), "rho": rng.uniform(0.3, 2.0, F),
           "phi": rng.normal(0, 0.15, (F, 4)), "vo_t": rng.normal(0, 1, (F, Ct)), "hs": rng.normal(0, 1, (F, Ct))}
    prior = [0.4, 0.45, 0.15]
    ex = xf.exchange(adm, val["ev"], val["need"], val["rho"], val["phi"], val["vo_t"], val["hs"], prior)
    dec = xf.decisions(adm, ex["lam"])
    G = torch.tensor(rng.normal(0, 1, (F, Ct)))

    def loss_of(v: Dict[str, np.ndarray], grad: bool = False):
        item = {k: torch.tensor(a, dtype=torch.float64, requires_grad=grad) for k, a in v.items()}
        item["dec"] = dec
        out = xt.exchange_batch([item], prior, 0.25, 1.0, 0.5, 3, 8)[0]
        return (out["dV"] * G).sum(), item, out

    loss, item, out = loss_of(val, True)
    loss.backward()
    forward = float(np.abs(out["dV"].detach().numpy() - ex["dV"]).max())
    worst, checked, active = 0.0, 0, int(out["smask"].sum())
    for name in ("ev", "need", "rho", "phi", "vo_t", "hs"):
        g = item[name].grad.numpy()
        idxs = [tuple(i) for i in np.argwhere(np.abs(g) > 1e-9)]
        for idx in [idxs[i] for i in rng.choice(len(idxs), size=min(6, len(idxs)), replace=False)] if idxs else []:
            h = 1e-6
            up = {k: a.copy() for k, a in val.items()}
            dn = {k: a.copy() for k, a in val.items()}
            up[name][idx] += h
            dn[name][idx] -= h
            num = (float(loss_of(up)[0]) - float(loss_of(dn)[0])) / (2 * h)
            worst = max(worst, abs(num - g[idx]) / max(1.0, abs(num)))
            checked += 1
    reach = {name: bool(np.abs(item[name].grad.numpy()).max() > 0) for name in ("ev", "need", "rho", "phi", "vo_t", "hs")}
    ok = forward < 1e-9 and worst < tol and checked >= 12 and active > 0 and all(reach.values())
    return {"passed": bool(ok), "forward_abs_error": forward, "worst_relative_gradient_error": worst, "entries_checked": checked,
            "active_sources": active, "gradient_reaches": reach}


def check_arms(seed: int = 0) -> Dict:
    import torch
    from .config import ARMS, SpliceConfig
    from .losses import aux_losses, total_loss
    from .neck import SpliceNeck
    in_ch, out_ch, strides, size = (24, 40, 56), (32, 48, 64), (8, 16, 32), 128
    rng = np.random.default_rng(seed)
    feats = [torch.tensor(rng.standard_normal((2, c, size // s, size // s)).astype(np.float32)) for c, s in zip(in_ch, strides)]
    boxes = []
    for _ in range(2):
        xy = rng.uniform(0, size * 0.6, (4, 2))
        boxes.append(np.concatenate([xy, np.minimum(xy + rng.uniform(16, 56, (4, 2)), size - 1)], 1))
    res, ok = {}, True
    for arm, spec in ARMS.items():
        torch.manual_seed(seed)
        neck = SpliceNeck(in_ch, out_ch, SpliceConfig(arm=arm), workers=0, strides=strides)
        with torch.no_grad():
            for h in neck.heads:                      # eligible cells and linked pairs, so that fragments form
                h.out.bias[9] = 3.0
            neck.score_bias.fill_(2.0)
            neck.gate_tau.fill_(1e9)                  # every pair admissible
            neck.E_F.weight.copy_(0.1 * torch.eye(neck.cfg.transport))
            for z in neck.Z:
                z.weight.normal_(0, 0.05)
            if spec["offset"]:
                for z in neck.score_off:
                    z.weight.normal_(0, 0.05)
        neck.train()
        out = neck(feats, boxes, probe=True)
        det = sum((c ** 2).mean() + (b ** 2).mean() for c, b in zip(out["cls_feats"], out["box_feats"]))
        if out["score_offset"] is not None:
            det = det + sum((o ** 2).mean() for o in out["score_offset"])
        loss, extra = total_loss(det, out, aux_losses(out, boxes, strides, neck.cfg.hyp_weight), neck.cfg, 2)
        loss.backward()
        gmax = lambda p: float(p.grad.abs().max()) if p.grad is not None else 0.0
        finite = all(bool(torch.isfinite(p.grad).all()) for p in neck.parameters() if p.grad is not None)
        got = {"exchange": max(gmax(neck.W_S.weight), gmax(neck.trust.weight), gmax(neck.O_T.weight)) > 0,
               "forecast": max(gmax(neck.W_K.weight), gmax(neck.O_F.weight)) > 0}
        want = {"exchange": spec["exchange"] == "always", "forecast": spec["forecast"] is not None}
        fragments = [r["F"] for r in out["frag"]]
        good = finite and got == want and min(fragments) > 1 and bool(torch.isfinite(loss))
        res[arm] = {"passed": bool(good), "loss": float(loss), "fragments": fragments, "gradient_reaches": got,
                    "expected": want, "finite": finite, **extra}
        ok &= good
    return {"passed": bool(ok), "arms": res}


def run(log: Callable = print) -> Dict:
    import torch
    out = {"torch": torch.__version__, "exchange": check_exchange(), "arms": check_arms()}
    out["passed"] = bool(out["exchange"]["passed"] and out["arms"]["passed"])
    log(f"[SELFCHECK] exchange: {out['exchange']['passed']} (gradient error {out['exchange']['worst_relative_gradient_error']:.2e}, "
        f"{out['exchange']['entries_checked']} entries); arms: "
        + ", ".join(f"{a}={'ok' if r['passed'] else 'FAIL'}" for a, r in out["arms"]["arms"].items()))
    return out
