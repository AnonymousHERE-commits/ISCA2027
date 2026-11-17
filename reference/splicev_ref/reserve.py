"""P3 reserve anchors by thin-aware farthest-point sampling (SPLICE-V §2 Coarsen; DD-09).

Candidates are P3 cells with eligibility e >= RESERVE_MIN_E.  The first pick is
the candidate with the largest thin cue t; every later pick maximises
dmin2 * (256 + t), dmin2 the squared grid distance to the picks so far.  All
ties go to the smallest cell index.  Work is counted in candidate scans.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import gen
from .arith import Counters

BIG = 1 << 30


def select_reserve(e: np.ndarray, t: np.ndarray, count: int = gen.N_RESERVE, width: int = gen.P3_W,
                   cnt: Counters | None = None) -> np.ndarray:
    """Return up to 'count' P3 cell indices (in selection order)."""
    e = np.asarray(e, dtype=np.int64).ravel()
    t = np.asarray(t, dtype=np.int64).ravel()
    cand = np.flatnonzero(e >= gen.RESERVE_MIN_E)
    if cand.size == 0:
        return np.zeros(0, dtype=np.int64)
    cy, cx = cand // width, cand % width
    tc = t[cand]
    dmin = np.full(cand.size, BIG, dtype=np.int64)
    picked = np.zeros(cand.size, dtype=bool)
    out = []
    for k in range(min(count, cand.size)):
        if k == 0:
            score = tc.copy()
        else:
            score = dmin * (256 + tc)
        score = np.where(picked, -1, score)
        best = int(np.argmax(score))          # argmax returns the first maximum = smallest index
        out.append(int(cand[best]))
        picked[best] = True
        d2 = (cy - cy[best]) ** 2 + (cx - cx[best]) ** 2
        dmin = np.minimum(dmin, d2)
        if cnt is not None:
            cnt.add("reserve_scans", int(cand.size))
    return np.asarray(out, dtype=np.int64)


def reserve_window_position(cells_p3: np.ndarray, width_p3: int = gen.P3_W) -> Tuple[np.ndarray, np.ndarray]:
    """P4-parent window convention: (x, y) on the P4 lattice."""
    cells = np.asarray(cells_p3, dtype=np.int64)
    return (cells % width_p3) >> 1, (cells // width_p3) >> 1
