"""Fragment membership operations (SPLICE-V §2 Coarsen, Remark 1).

M is stored as one u16 id per cell (UNASSIGNED = 0xFFFF).  Restriction
H = R X with R = (M^T W M)^-1 M^T W is an exact int64 weighted sum per fragment
followed by one half-even integer division; fills and details are exact.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import gen
from .arith import Counters, div_rne_np, sat_np

UNASSIGNED = gen.UNASSIGNED


def assigned(ids: np.ndarray) -> np.ndarray:
    return np.asarray(ids) != UNASSIGNED


def fragment_sums(x: np.ndarray, ids: np.ndarray, weights: np.ndarray, n_frag: int) -> Tuple[np.ndarray, np.ndarray]:
    """sums (F, C) = sum_i w_i x_i over each fragment; wsum (F,) = sum_i w_i."""
    x = np.asarray(x, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    sums = np.zeros((n_frag, x.shape[1]), dtype=np.int64)
    wsum = np.zeros(n_frag, dtype=np.int64)
    m = ids != UNASSIGNED
    if np.any(ids[m] >= n_frag):
        raise ValueError("fragment id out of range")
    np.add.at(sums, ids[m], w[m, None] * x[m])
    np.add.at(wsum, ids[m], w[m])
    return sums, wsum


def fragment_means(sums: np.ndarray, wsum: np.ndarray, lo: int = -128, hi: int = 127,
                   cnt: Counters | None = None) -> np.ndarray:
    """H = sat(div_rne(sum, wsum)); fragments with zero weight get 0."""
    sums = np.asarray(sums, dtype=np.int64)
    wsum = np.asarray(wsum, dtype=np.int64)
    safe = np.where(wsum > 0, wsum, 1)
    h = div_rne_np(sums, safe[:, None])
    h = np.where((wsum > 0)[:, None], h, 0)
    return sat_np(h, lo, hi, cnt, "sat_mean")


def restrict(x, ids, weights, n_frag, cnt=None) -> np.ndarray:
    s, ws = fragment_sums(x, ids, weights, n_frag)
    return fragment_means(s, ws, cnt=cnt)


def fill(h: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """(M h)_i = h[id_i], 0 for unassigned cells."""
    h = np.asarray(h, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    out = np.zeros((ids.shape[0], h.shape[1]), dtype=np.int64)
    m = ids != UNASSIGNED
    out[m] = h[ids[m]]
    return out


def detail(x: np.ndarray, h: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """D = X - M H (exact, int16 container)."""
    return np.asarray(x, dtype=np.int64) - fill(h, ids)


def project_out(rw: np.ndarray, ids: np.ndarray, weights: np.ndarray, n_frag: int) -> np.ndarray:
    """R_w^perp = (I - Pi) R_w with the manifest's rounding; unassigned rows unchanged."""
    mean = restrict(rw, ids, weights, n_frag)
    return np.asarray(rw, dtype=np.int64) - fill(mean, ids)


def synthesize(xs: np.ndarray, ids: np.ndarray, dh_hat: np.ndarray, r_perp: np.ndarray | None,
               cnt: Counters | None = None) -> np.ndarray:
    """X' = sat8(X + M dH_hat + R_perp) on the transported channels (Eq. 4)."""
    out = np.asarray(xs, dtype=np.int64) + fill(dh_hat, ids)
    if r_perp is not None:
        out = out + np.asarray(r_perp, dtype=np.int64)
    return sat_np(out, -128, 127, cnt, "sat_synth")


def class_band(n_feat: np.ndarray, ids: np.ndarray, weights: np.ndarray, n_frag: int,
               rz: np.ndarray | None, cnt: Counters | None = None) -> np.ndarray:
    """Pi N + M (r Z): fragment-constant class-band input, int8; unassigned rows are 0."""
    pin = restrict(n_feat, ids, weights, n_frag, cnt)
    tot = pin if rz is None else pin + np.asarray(rz, dtype=np.int64)
    out = fill(sat_np(tot, -128, 127, cnt, "sat_clsband"), ids)
    return out


def level_weights(e_level: np.ndarray, ids: np.ndarray, n_frag: int) -> np.ndarray:
    """DD-08: eligibility weights; a fragment with zero mass at the level uses uniform weights."""
    e = np.asarray(e_level, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    m = ids != UNASSIGNED
    mass = np.zeros(n_frag, dtype=np.int64)
    np.add.at(mass, ids[m], e[m])
    w = e.copy()
    zero_mass = np.zeros_like(m)
    zero_mass[m] = mass[ids[m]] == 0
    w[zero_mass] = 1
    return w


def extend_levels(anchor_ids_p4: np.ndarray, reserve_cells_p3: np.ndarray, reserve_ids: np.ndarray,
                  p3_hw=(gen.P3_H, gen.P3_W), p4_hw=(gen.P4_H, gen.P4_W), p5_hw=(gen.P5_H, gen.P5_W)):
    """DD-07: hard memberships of every level from the anchor memberships."""
    h3, w3 = p3_hw
    h4, w4 = p4_hw
    h5, w5 = p5_hw
    ids4 = np.asarray(anchor_ids_p4, dtype=np.int64).reshape(h4, w4)
    ys, xs = np.mgrid[0:h3, 0:w3]
    ids3 = ids4[np.minimum(ys >> 1, h4 - 1), np.minimum(xs >> 1, w4 - 1)].copy()
    for cell, fid in zip(np.asarray(reserve_cells_p3, dtype=np.int64), np.asarray(reserve_ids, dtype=np.int64)):
        ids3.flat[cell] = fid
    ids5 = np.full((h5, w5), UNASSIGNED, dtype=np.int64)
    for y in range(h5):
        for x in range(w5):
            kids = ids4[2 * y:2 * y + 2, 2 * x:2 * x + 2].ravel()
            kids = kids[kids != UNASSIGNED]
            if kids.size:
                vals, counts = np.unique(kids, return_counts=True)
                best = counts.max()
                ids5[y, x] = int(vals[counts == best].min())
    return ids3.ravel(), ids4.ravel(), ids5.ravel()
