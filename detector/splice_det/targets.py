"""Supervision targets of the SPLICE heads from ground-truth boxes (numpy, per image).

SPLICE-V names the supervised heads (edge affinity pi, purity q, eligibility e, thin
structure t, gate g, pair scores of the linkage window, box hypotheses) but not their
targets; the rules below are design decisions of this implementation (documented in
detector/README.md):

  ownership   a cell is owned by the smallest-area box that contains its centre; else -1
  e (elig.)   1 on owned cells
  q (purity)  1 / (number of boxes containing the centre), 0 on background
  t (thin)    1 when the owner box's shorter side is below two cells of the level
  g (gate)    1 on owned cells (the class score is sigma(l) * g, Prop. 3)
  pi_k        1 when the cell and its k-th neighbour have the same owner (both owned)
  pair score  same rule on the linkage window (forward half window, reserve windows)
  hypothesis  the owner box (x0, y0, x1, y1) in pixels, only on owned anchors
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .ops import NEIGH

LINK_FWD = tuple((dy, dx) for dy in range(0, 3) for dx in range(-2, 3) if dy > 0 or dx > 0)


def ownership(boxes: np.ndarray, stride: int, h: int, w: int):
    """boxes (M, 4) xyxy pixels -> owner (h, w) int, count (h, w) number of containing boxes."""
    owner = np.full((h, w), -1, dtype=np.int64)
    count = np.zeros((h, w), dtype=np.int64)
    if boxes is None or len(boxes) == 0:
        return owner, count
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cy = (np.arange(h) + 0.5) * stride
    cx = (np.arange(w) + 0.5) * stride
    inside = (cx[None, None, :] >= b[:, 0, None, None]) & (cx[None, None, :] < b[:, 2, None, None]) & \
             (cy[None, :, None] >= b[:, 1, None, None]) & (cy[None, :, None] < b[:, 3, None, None])
    count = inside.sum(axis=0)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    big = np.where(inside, area[:, None, None], np.inf)
    best = np.argmin(big, axis=0)
    owner = np.where(count > 0, best, -1)
    return owner, count


def level_targets(boxes: np.ndarray, stride: int, h: int, w: int) -> Dict[str, np.ndarray]:
    owner, count = ownership(boxes, stride, h, w)
    owned = owner >= 0
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) if boxes is not None and len(boxes) else np.zeros((0, 4))
    short = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)
    thin = np.zeros((h, w))
    if len(b):
        thin = np.where(owned, (short[np.maximum(owner, 0)] < 2 * stride).astype(np.float64), 0.0)
    pi = np.zeros((8, h, w))
    for k, (dy, dx) in enumerate(NEIGH):
        nb = np.full((h, w), -2, dtype=np.int64)
        ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
        xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
        nb[yd, xd] = owner[ys, xs]
        pi[k] = (owned & (nb == owner)).astype(np.float64)
    return {
        "owner": owner,
        "e": owned.astype(np.float64),
        "q": np.where(owned, 1.0 / np.maximum(count, 1), 0.0),
        "t": thin,
        "g": owned.astype(np.float64),
        "pi": pi,
    }


def window_targets(owner4: np.ndarray) -> np.ndarray:
    """(n4, 12) same-owner indicator on the forward half window (LINK_FWD order); -1 = no pair."""
    h, w = owner4.shape
    out = np.full((h, w, 12), -1.0)
    for k, (dy, dx) in enumerate(LINK_FWD):
        y1, x0, x1 = h - dy, max(0, -dx), min(w, w - dx)
        if y1 <= 0 or x1 <= x0:
            continue
        a = owner4[0:y1, x0:x1]
        b = owner4[dy:y1 + dy, x0 + dx:x1 + dx]
        out[0:y1, x0:x1, k] = ((a >= 0) & (a == b)).astype(np.float64)
    return out.reshape(h * w, 12)


def reserve_targets(owner3: np.ndarray, owner4: np.ndarray, reserve: Sequence[int], rr_partner: np.ndarray):
    """Reserve window targets (R, 25) against the P4 window of each reserve's parent, and
    reserve-pair targets (R, 8) aligned with rr_partner (0xFFFF = no pair -> -1)."""
    h3, w3 = owner3.shape
    h4, w4 = owner4.shape
    res = np.asarray(reserve, dtype=np.int64).ravel()
    o3 = owner3.ravel()
    own = o3[res][:, None]
    k = np.arange(25)
    yy = ((res // w3) >> 1)[:, None] + (k // 5 - 2)[None]
    xx = ((res % w3) >> 1)[:, None] + (k % 5 - 2)[None]
    inside = (yy >= 0) & (yy < h4) & (xx >= 0) & (xx < w4)
    o4 = owner4[np.clip(yy, 0, h4 - 1), np.clip(xx, 0, w4 - 1)]
    rs = np.where(inside, ((own >= 0) & (own == o4)).astype(np.float64), -1.0)
    rp = np.asarray(rr_partner, dtype=np.int64).reshape(res.size, -1)
    valid = rp != 0xFFFF
    op = o3[res[np.where(valid, rp, 0)]] if res.size else np.zeros(rp.shape, dtype=np.int64)
    rr = np.where(valid, ((own >= 0) & (own == op)).astype(np.float64), -1.0)
    return rs, rr


def hypothesis_targets(boxes: np.ndarray, owner_anchor: np.ndarray):
    """Per-anchor box targets (n, 4) in pixels and a validity mask (n,)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) if boxes is not None and len(boxes) else np.zeros((0, 4))
    valid = owner_anchor >= 0
    tgt = np.zeros((owner_anchor.size, 4))
    if len(b):
        tgt[valid] = b[owner_anchor[valid]]
    return tgt, valid
