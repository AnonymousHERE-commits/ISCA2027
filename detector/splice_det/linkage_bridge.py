"""Reserve selection and certified linkage through the integer reference (CPU).

The fragment ids are discrete decisions: the model's float scores are quantized to the
manifest's u8 codes and the integer reference (reference/splicev_ref) computes exactly
what the PL engines compute on the board (Lemma 2 makes the certified MNN rounds equal to
the sequential linkage, so DFP32, Db and M share the same fragments for the same codes).
Images are processed in parallel by a process pool.
"""
from __future__ import annotations

from multiprocessing import get_context
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import REPO  # noqa: F401  (path setup)
from splicev_ref import gen
from splicev_ref.linkage import anchor_order, certified, select_fragments, window_edges
from splicev_ref.membership import extend_levels
from splicev_ref.reserve import select_reserve

UNASSIGNED_REF = gen.UNASSIGNED


def to_u8(p: np.ndarray) -> np.ndarray:
    """Probability in [0, 1] -> u8 code (round half even)."""
    return np.clip(np.round(np.asarray(p, dtype=np.float64) * 255.0), 0, 255).astype(np.int64)


def reserve_job(args) -> np.ndarray:
    e3, t3, count, w3 = args
    return select_reserve(np.asarray(e3), np.asarray(t3), count, w3)


def rr_pairs(reserve: np.ndarray, w3: int, slots: int = 8, window: int = gen.LINK_WINDOW) -> np.ndarray:
    """Up to `slots` later reserves within the Chebyshev window of each reserve's P4 parent."""
    r = np.asarray(reserve, dtype=np.int64).ravel()
    rx, ry = (r % w3) >> 1, (r // w3) >> 1
    out = np.full((r.size, slots), 0xFFFF, dtype=np.int64)
    if r.size == 0:
        return out
    near = np.maximum(np.abs(rx[None, :] - rx[:, None]), np.abs(ry[None, :] - ry[:, None])) <= window
    ok = near & (np.arange(r.size)[None, :] > np.arange(r.size)[:, None])
    for i in range(r.size):
        js = np.flatnonzero(ok[i])[:slots]
        out[i, :js.size] = js
    return out


def linkage_job(args) -> Dict:
    """One image: u8 codes in, fragment ids of every level out (-1 = unassigned)."""
    (scores_p4, reserve, rs, rr_p, rr_s, e4, e3, h3, w3, h4, w4, h5, w5, budget, f_cap) = args
    edges = window_edges(scores_p4, reserve, rs, rr_p, rr_s, h4, w4, w3)
    e_anchor = np.concatenate([np.asarray(e4).ravel(), np.asarray(e3).ravel()[reserve]]).astype(np.int64)
    active = e_anchor >= gen.RECV_MIN_E
    n_anchor = e_anchor.size
    link = certified(n_anchor, edges, active.tolist(), budget)
    ids, F = select_fragments(link, n_anchor, e_anchor, f_cap)
    n4 = h4 * w4
    ids3, ids4, ids5 = extend_levels(ids[:n4], reserve, ids[n4:], (h3, w3), (h4, w4), (h5, w5))
    conv = lambda a: np.where(np.asarray(a) == UNASSIGNED_REF, -1, np.asarray(a)).astype(np.int64).ravel()
    return {"ids_anchor": conv(ids), "F": int(F), "order": np.asarray(anchor_order(ids), dtype=np.int64),
            "ids": {"P3": conv(ids3), "P4": conv(ids4), "P5": conv(ids5)}, "e_anchor": e_anchor,
            "rounds": link.rounds, "work": link.work, "budget_hit": bool(link.budget_hit), "overflow": bool(link.overflow)}


class Pool:
    """Process pool for the per-image CPU steps (falls back to serial with workers=0)."""

    def __init__(self, workers: int = 0):
        self.workers = workers
        self.pool = get_context("spawn").Pool(workers) if workers > 0 else None

    def map(self, fn, jobs: Sequence) -> List:
        if self.pool is None:
            return [fn(j) for j in jobs]
        return self.pool.map(fn, jobs)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
