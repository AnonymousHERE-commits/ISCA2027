"""Training objective (SPLICE-V section 2):

  L = L_det + mu * mean (dHhat - sg(dH))^2 + mu_s || Q [G(X' + M v) - G(X')] ||^2 + L_aux

L_det is YOLO26's own criterion on the arm's readout.  The stop-gradient keeps the exchange
from being pulled toward its forecast (the term exists for the arms with a forecast: d, e,
f); the sensitivity term comes from SpliceNeck(probe=True) and keeps fragment-constant
perturbations out of the detail part.  L_aux supervises the neck's cell heads (the gate by
cell-in-box targets), the linkage pair scorer and the box hypotheses with the targets of
targets.py (design decisions, README).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import targets as tg


def batch_boxes(batch: Dict) -> List[np.ndarray]:
    """Per-image GT boxes (xyxy, pixels of the network input) from an Ultralytics batch."""
    n_img = batch["img"].shape[0]
    h, w = batch["img"].shape[-2:]
    bi = batch["batch_idx"].detach().cpu().numpy().astype(np.int64).ravel()
    bb = batch["bboxes"].detach().cpu().numpy().reshape(-1, 4).astype(np.float64)
    out = []
    for n in range(n_img):
        b = bb[bi == n]
        xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                         (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1) if len(b) else np.zeros((0, 4))
        out.append(xyxy)
    return out


def _bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if mask is None:
        return loss.mean()
    m = mask.to(loss.dtype)
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def aux_losses(nk: Dict, boxes: Sequence[np.ndarray], strides: Sequence[int], hyp_weight: float = 0.05) -> Dict[str, torch.Tensor]:
    hd = nk["heads"]
    dev = hd[0]["e"].device
    parts: Dict[str, torch.Tensor] = {}
    owners = []
    for l, s in enumerate(strides):
        h, w = hd[l]["e"].shape[-2:]
        tl = [tg.level_targets(b, s, h, w) for b in boxes]
        owners.append([t["owner"] for t in tl])
        T = lambda k: torch.as_tensor(np.stack([t[k] for t in tl]), device=dev, dtype=hd[l]["e"].dtype)
        e, q, t, g, pi = T("e")[:, None], T("q")[:, None], T("t")[:, None], T("g")[:, None], T("pi")
        inside = torch.ones_like(pi)
        for k, (dy, dx) in enumerate(((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))):
            if dy < 0:
                inside[:, k, :1] = 0
            if dy > 0:
                inside[:, k, -1:] = 0
            if dx < 0:
                inside[:, k, :, :1] = 0
            if dx > 0:
                inside[:, k, :, -1:] = 0
        parts[f"e{l}"] = _bce(hd[l]["e_logit"], e)
        parts[f"q{l}"] = _bce(hd[l]["q_logit"], q)
        parts[f"t{l}"] = _bce(hd[l]["t_logit"], t, e > 0.5)
        parts[f"g{l}"] = _bce(hd[l]["g_logit"], g)
        parts[f"pi{l}"] = _bce(hd[l]["pi_logit"], pi, inside > 0.5)
    # linkage pair scorer: P4 forward window, reserve windows and reserve pairs
    win = nk["window_logits"]
    wt = torch.as_tensor(np.stack([tg.window_targets(o).reshape(win.shape[2], win.shape[3], 12).transpose(2, 0, 1)
                                   for o in owners[1]]), device=dev, dtype=win.dtype)
    parts["pair_p4"] = _bce(win, wt.clamp(min=0), wt >= 0)
    rs_l, rs_t, hyp_l = [], [], []
    for n, rec in enumerate(nk["frag"]):
        rs_tgt, rr_tgt = tg.reserve_targets(owners[0][n], owners[1][n], rec["reserve"], rec["rr_p"])
        rs_l += [rec["res_logits"][0].flatten(), rec["res_logits"][1].flatten()]
        rs_t += [torch.as_tensor(rs_tgt.ravel(), device=dev), torch.as_tensor(rr_tgt.ravel(), device=dev)]
        own_a = np.concatenate([owners[1][n].ravel(), owners[0][n].ravel()[np.asarray(rec["reserve"], dtype=np.int64)]])
        tgt, valid = tg.hypothesis_targets(boxes[n], own_a)
        if valid.any():
            v = torch.as_tensor(valid, device=dev)
            y = torch.as_tensor(tgt, device=dev, dtype=rec["mu"].dtype)
            lp = rec["logprec"].clamp(-12.0, 2.0)
            z = (rec["mu"] - y) / rec["stride_a"][:, None]          # error in strides of the anchor's level
            nll = 0.5 * (torch.exp(lp) * z ** 2 - lp)
            hyp_l.append(nll[v].mean())
    if rs_l:
        lg = torch.cat(rs_l)
        tt = torch.cat(rs_t).to(lg.dtype)
        if lg.numel():
            parts["pair_reserve"] = _bce(lg, tt.clamp(min=0), tt >= 0)
    if hyp_l:
        parts["hyp"] = hyp_weight * torch.stack(hyp_l).mean()
    return parts


def total_loss(det_loss: torch.Tensor, nk: Dict, aux: Dict[str, torch.Tensor], cfg, batch_size: int):
    """Combine the terms; Ultralytics returns L_det already multiplied by the batch size, so
    the added terms are scaled the same way."""
    extra = cfg.mu_forecast * nk["fc_loss"]
    if "sens_loss" in nk:
        extra = extra + cfg.lambda_sens * nk["sens_loss"]
    if aux:
        extra = extra + cfg.aux_weight * torch.stack(list(aux.values())).sum()
    return det_loss + batch_size * extra, {"fc": float(nk["fc_loss"]), "sens": float(nk.get("sens_loss", 0.0)),
                                           "aux": float(sum(float(v) for v in aux.values())) if aux else 0.0}
