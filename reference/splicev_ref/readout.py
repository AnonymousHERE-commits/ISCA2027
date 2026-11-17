"""Co-designed readout (SPLICE-V Eq. (5), Prop. 3, Prop. 5).

Class score   p_hat = floor(sigma_hat(l) * g_hat >> 15) in Q0.15, sigma_hat < 1.
Candidate     p_hat >= tau.
Exact skip    a block whose gates are all below tau produces no candidate (Prop. 3).
Residual      r = dH - dHhat (int16, |r| <= 191); r Z is an int16 x int8 product rounded
              once and joins only the class band (Prop. 5).
"""
from __future__ import annotations


import numpy as np

from . import gen
from .arith import Counters, rshift_rne_np, sat_np
from .forecast import sig_index


def class_scores(logits_q88: np.ndarray, gate_q15: np.ndarray) -> np.ndarray:
    lg = np.asarray(logits_q88, dtype=np.int64).ravel()
    g = np.asarray(gate_q15, dtype=np.int64).ravel()
    if np.any(g < 0) or np.any(g >= (1 << 15)):
        raise ValueError("gate codes must lie in [0, 2^15)")
    sig = np.asarray([gen.SIG_TABLE[sig_index(int(v))] for v in lg], dtype=np.int64)
    return (sig * g) >> gen.SCORE_FRAC


def candidates(p: np.ndarray, tau: int = gen.TAU_DET_Q15) -> np.ndarray:
    return np.flatnonzero(np.asarray(p) >= tau)


def skippable_blocks(gate_hw: np.ndarray, tau: int = gen.TAU_DET_Q15, block: int = gen.SKIP_BLOCK) -> np.ndarray:
    """(H/b, W/b) bool: every gate in the block is below tau."""
    g = np.asarray(gate_hw, dtype=np.int64)
    h, w = g.shape
    hb, wb = -(-h // block), -(-w // block)
    out = np.zeros((hb, wb), dtype=bool)
    for by in range(hb):
        for bx in range(wb):
            out[by, bx] = bool(np.all(g[by * block:(by + 1) * block, bx * block:(bx + 1) * block] < tau))
    return out


def residual(dh: np.ndarray, dh_hat: np.ndarray, cnt: Counters | None = None) -> np.ndarray:
    r = np.asarray(dh, dtype=np.int64) - np.asarray(dh_hat, dtype=np.int64)
    if np.any(np.abs(r) > gen.RES_BOUND):
        if cnt is not None:
            cnt.add("residual_bound_violation", int(np.count_nonzero(np.abs(r) > gen.RES_BOUND)))
    return r


def rz_product(r: np.ndarray, z: np.ndarray, cnt: Counters | None = None) -> np.ndarray:
    """(F, 48) int16 x (48, C) int8 -> (F, C) int8, rounded once at RZ_SHIFT, saturation counted."""
    acc = np.asarray(r, dtype=np.int64) @ np.asarray(z, dtype=np.int64)
    return sat_np(rshift_rne_np(acc, gen.RZ_SHIFT), -128, 127, cnt, "sat_rz")
