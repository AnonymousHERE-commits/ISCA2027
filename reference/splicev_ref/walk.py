"""Ownership-typed lazy walk and signed wavelet bands (SPLICE-V §2, Definition 1, Prop. 1).

Walk export:  P = I/2 + S^-1 (I + W) / 2,  S = I + diag(W 1).  Lateral taps
p_ij = rne(2^14 P_ij) = div_rne(w_ij << 13, S_code) with w in Q0.15 and
S_code = 2^15 + sum_j w_ij; the self tap is 2^14 - sum_j p_ij, so every row
sums to exactly 2^14 and constants are reproduced exactly (Prop. 1(i)).

Walk step:    X(t+1)_i = rne((sum_k tap_ik X(t)_{i+o_k}) >> 14) on int16 Q0.15 states.

Bands:        per cell, cumulative sums R_t(b) = 2^15 - rne(b^(t+1) >> 1) (Q0.15,
              b in Q0.16, powers by repeated RNE products) are monotone; the
              coefficients a_t = R_t - R_(t-1) are nonnegative and sum to 2^15;
              band j accumulates sum_t (a_t(b_j) - a_t(b_(j+1))) X(t) in int32 and
              is rounded once at its own exponent: z_j = sat8(rne(acc_j >> (23 - e_j))).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from . import gen
from .arith import Counters, div_rne_np, rshift_rne_np, sat_np

# 8-neighbour directions (dy, dx) in the manifest's fixed order and their 3x3 slots.
NEIGH = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
TAP_SLOT = (0, 1, 2, 3, 5, 6, 7, 8)
OFFSETS9 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))
ONE_W = 1 << gen.AFF_FRAC        # affinity 1.0 in Q0.15
ONE_TAP = 1 << gen.TAP_Q         # 2^14
ONE_COEF = 1 << gen.COEF_R       # 2^15
ONE_BETA = 1 << gen.BETA_FRAC    # 2^16


def inside_mask(h: int, w: int) -> np.ndarray:
    """(h, w, 8) bool: does direction d point inside the lattice."""
    m = np.zeros((h, w, 8), dtype=bool)
    ys, xs = np.mgrid[0:h, 0:w]
    for d, (dy, dx) in enumerate(NEIGH):
        m[..., d] = (ys + dy >= 0) & (ys + dy < h) & (xs + dx >= 0) & (xs + dx < w)
    return m


def export_taps(w_codes: np.ndarray, lattice_inside: np.ndarray | None = None) -> np.ndarray:
    """Affinity codes (h, w, 8) in [0, 2^15] -> integer taps (h, w, 9), rows sum to 2^14.

    Directions pointing outside the lattice (lattice_inside False) get weight 0.
    """
    w = np.asarray(w_codes, dtype=np.int64)
    if w.ndim != 3 or w.shape[2] != 8:
        raise ValueError("w_codes must be (h, w, 8)")
    if np.any(w < 0) or np.any(w > ONE_W):
        raise ValueError("affinity codes must lie in [0, 2^15]")
    if lattice_inside is not None:
        w = np.where(lattice_inside, w, 0)
    s_code = ONE_W + w.sum(axis=2)                       # Q0.15 of S = 1 + sum w
    lat = div_rne_np(w << 13, s_code[..., None])          # rne(2^14 * w / (2 S))
    taps = np.zeros(w.shape[:2] + (9,), dtype=np.int64)
    for d, slot in enumerate(TAP_SLOT):
        taps[..., slot] = lat[..., d]
    taps[..., 4] = ONE_TAP - lat.sum(axis=2)
    if np.any(taps < 0):
        raise AssertionError("negative tap after export")
    return taps


def walk_step(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """One step on int16 Q0.15 states (h, w, c); neighbours outside the array contribute 0."""
    x = np.asarray(x, dtype=np.int64)
    h, w = x.shape[:2]
    xp = np.zeros((h + 2, w + 2) + x.shape[2:], dtype=np.int64)
    xp[1:-1, 1:-1] = x
    acc = np.zeros_like(x)
    for k, (dy, dx) in enumerate(OFFSETS9):
        acc += taps[..., k][..., None] * xp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return rshift_rne_np(acc, gen.TAP_Q)


def cumulative_sums(beta: np.ndarray, steps: int) -> np.ndarray:
    """beta (...,) Q0.16 -> R (..., steps+1) Q0.15, monotone, R[..., steps] = 2^15."""
    beta = np.asarray(beta, dtype=np.int64)
    if np.any(beta < 0) or np.any(beta >= ONE_BETA):
        raise ValueError("beta codes must lie in [0, 2^16)")
    out = np.zeros(beta.shape + (steps + 1,), dtype=np.int64)
    pw = np.full(beta.shape, ONE_BETA, dtype=np.int64)    # beta^0 in Q0.16
    for t in range(steps):
        pw = rshift_rne_np(pw * beta, gen.BETA_FRAC)       # beta^(t+1)
        out[..., t] = ONE_COEF - rshift_rne_np(pw, 1)
    out[..., steps] = ONE_COEF
    return out


def bank_coefficients(beta: np.ndarray, steps: int) -> np.ndarray:
    """a_t(beta) = R_t - R_(t-1), nonnegative, summing to 2^15."""
    r = cumulative_sums(beta, steps)
    a = r.copy()
    a[..., 1:] = r[..., 1:] - r[..., :-1]
    return a


def band_coefficients(betas: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """betas (..., 4) Q0.16 strictly increasing chain b_1..b_4 (b_0 = 0 implicit).

    Returns d (..., 4, steps+1) band coefficients and low (..., steps+1) for A(b_4).
    """
    betas = np.asarray(betas, dtype=np.int64)
    if betas.shape[-1] != 4:
        raise ValueError("betas must carry the four nontrivial chain points")
    if np.any(np.diff(betas, axis=-1) <= 0) or np.any(betas[..., 0] <= 0):
        raise ValueError("beta chain must satisfy 0 < b_1 < b_2 < b_3 < b_4 < 1")
    chain = np.concatenate([np.zeros(betas.shape[:-1] + (1,), dtype=np.int64), betas], axis=-1)
    a = bank_coefficients(chain, steps)                  # (..., 5, steps+1)
    d = a[..., :4, :] - a[..., 1:, :]
    bound = np.abs(d).sum(axis=-1)
    if np.any(bound >= (1 << 16)):
        raise AssertionError("band accumulator certificate violated: sum_t |d_jt| must stay below 2^16")
    return d, a[..., 4, :]


def walk_powers(x0_i8: np.ndarray, taps: np.ndarray, steps: int, state_in_q15: bool = False) -> np.ndarray:
    """X(0..steps) as int16 Q0.15; x0 int8 codes are promoted by << 8 (exact)."""
    x = np.asarray(x0_i8, dtype=np.int64)
    x = x if state_in_q15 else (x << 8)
    out = [x]
    for _ in range(steps):
        x = walk_step(x, taps)
        out.append(x)
    return np.stack(out, axis=0)


def walk_bands(x0_i8: np.ndarray, taps: np.ndarray, betas: np.ndarray, steps: int,
               exps: Sequence[int], e_low: int, halo: int = 0,
               cnt: Counters | None = None) -> np.ndarray:
    """Signed bands Z_0..Z_3 and the low-pass A(b_4)X as int8 codes.

    x0_i8 (h, w, c), taps (h, w, 9), betas (h, w, 4).  With halo > 0 the
    arrays are a tile and only the interior [halo:h-halo, halo:w-halo] is
    returned (it is exact after 'steps' <= halo steps).  Output (5, h', w', c).
    """
    pw = walk_powers(x0_i8, taps, steps)                 # (T+1, h, w, c)
    d, low = band_coefficients(betas, steps)             # (h, w, 4, T+1), (h, w, T+1)
    h, w = pw.shape[1:3]
    sl = (slice(halo, h - halo), slice(halo, w - halo))
    pw = pw[:, sl[0], sl[1]]
    d = d[sl[0], sl[1]]
    low = low[sl[0], sl[1]]
    outs = []
    for j in range(4):
        acc = np.einsum("hwt,thwc->hwc", d[:, :, j, :], pw)
        if np.any(np.abs(acc) >= (1 << 31)):
            raise AssertionError("band accumulator left int32")
        outs.append(sat_np(rshift_rne_np(acc, 23 - int(exps[j])), -128, 127, cnt, f"sat_band{j}"))
    acc = np.einsum("hwt,thwc->hwc", low, pw)
    outs.append(sat_np(rshift_rne_np(acc, 23 - int(e_low)), -128, 127, cnt, "sat_band_low"))
    return np.stack(outs, axis=0)


def band_accumulators(x0_i8, taps, betas, steps, halo: int = 0) -> np.ndarray:
    """Unrounded int32 accumulators (5, h', w', c): exact telescoping check uses these."""
    pw = walk_powers(x0_i8, taps, steps)
    d, low = band_coefficients(betas, steps)
    h, w = pw.shape[1:3]
    sl = (slice(halo, h - halo), slice(halo, w - halo))
    pw = pw[:, sl[0], sl[1]]
    d = d[sl[0], sl[1]]
    low = low[sl[0], sl[1]]
    accs = [np.einsum("hwt,thwc->hwc", d[:, :, j, :], pw) for j in range(4)]
    accs.append(np.einsum("hwt,thwc->hwc", low, pw))
    return np.stack(accs, axis=0)


def level_exponents(level: str) -> Tuple[Tuple[int, ...], int]:
    return tuple(gen.BAND_EXP[level]), gen.BAND_EXP_LOW
