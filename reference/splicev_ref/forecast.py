"""Gated innovation forecast of Eq. (3) in the manifest's arithmetic.

For receiver f with admissible set C_f (mask G) and a null option 0:
  l_fg  = clamp(rne((<K_f,K_g> * kk_mult + <U nu_f, U nu_g> * uu_mult) >> shift))      Q?.4 code
  l_f0  = clamp(rne((u0^T nu_f) * null_mult >> null_shift))
  w     = FC_EXP[l - L_MIN]              (table exponential, no row maximum: DD-16)
  num_c = sum_{g in C_f} w_g * clip_B(VO_g,c - Hs_f,c),   den = w_0 + sum_{g in C_f} w_g
  alpha = (sigma_hat(u_alpha^T nu_f) * 15) >> 4          ((1 - 2^-4) sigma, Q0.15)
  scale = (alpha << 30) // den           (one integer division per row)
  dHhat_f,c = sat_sym(rne((num_c * scale) >> 45))        (rounded once)
Masked donors contribute exactly zero; the null weight enters only the
denominator; an empty C_f gives zero (abstention).  |dHhat| <= alpha (1 - a_0) B.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import gen
from .arith import Counters, clamp, rshift_rne

L_MIN, L_MAX = gen.FC_LOGIT_MIN, gen.FC_LOGIT_MAX


def logit_code(raw: int, mult: int, shift: int, cnt: Counters | None = None, name: str = "fc_logit_clamp") -> int:
    v = rshift_rne(raw * mult, shift)
    if (v < L_MIN or v > L_MAX) and cnt is not None:
        cnt.add(name)
    return clamp(v, L_MIN, L_MAX)


def sig_index(code_q88: int) -> int:
    c = clamp(code_q88, gen.SIG_IN_MIN, gen.SIG_IN_MAX)
    return (c - gen.SIG_IN_MIN) >> gen.SIG_IDX_SHIFT


def sigmoid_q15(code_q88: int) -> int:
    return gen.SIG_TABLE[sig_index(code_q88)]


def alpha_q15(alpha_raw: int, cnt: Counters | None = None) -> int:
    code = rshift_rne(alpha_raw * gen.FC_ALPHA_MULT, gen.FC_ALPHA_SHIFT)
    if (code < gen.SIG_IN_MIN or code > gen.SIG_IN_MAX) and cnt is not None:
        cnt.add("fc_alpha_clamp")
    return (sigmoid_q15(code) * gen.FC_SHRINK_NUM) >> gen.FC_SHRINK_SHIFT


def forecast_row(f: int, K: np.ndarray, Uv: np.ndarray, VO: np.ndarray, Hs: np.ndarray,
                 null_raw: int, alpha_raw: int, adm_row: Sequence[bool], cnt: Counters | None = None):
    """Returns (dHhat (48,) ints, abstained bool)."""
    K = np.asarray(K, dtype=np.int64)
    Uv = np.asarray(Uv, dtype=np.int64)
    VO = np.asarray(VO, dtype=np.int64)
    Hs = np.asarray(Hs, dtype=np.int64)
    width = VO.shape[1]
    l0 = logit_code(int(null_raw), gen.FC_NULL_MULT, gen.FC_NULL_SHIFT, cnt, "fc_null_clamp")
    w0 = gen.FC_EXP[l0 - L_MIN]
    den = w0
    num = [0] * width
    n_adm = 0
    for g in range(len(adm_row)):
        if not adm_row[g] or g == f:
            continue
        n_adm += 1
        kk = int(np.dot(K[f], K[g]))
        uu = int(np.dot(Uv[f], Uv[g]))
        l = logit_code(kk * gen.FC_KK_MULT + uu * gen.FC_UU_MULT, 1, gen.FC_SHIFT, cnt)
        w = gen.FC_EXP[l - L_MIN]
        den += w
        diff = VO[g] - Hs[f]
        for c in range(width):
            d = int(diff[c])
            d = -gen.FC_B if d < -gen.FC_B else gen.FC_B if d > gen.FC_B else d
            num[c] += w * d
    alpha = alpha_q15(int(alpha_raw), cnt)
    scale = (alpha << gen.FC_SCALE_SHIFT) // den
    out = []
    for c in range(width):
        v = rshift_rne(num[c] * scale, gen.FC_OUT_SHIFT)
        out.append(max(-127, min(127, v)))
    abstained = n_adm == 0 or w0 >= den - w0
    if cnt is not None and abstained:
        cnt.add("fc_abstain")
    return out, abstained


def forecast(K, Uv, VO, Hs, null_raw, alpha_raw, adm: np.ndarray, rows=None, cnt: Counters | None = None):
    n = adm.shape[0]
    rows = range(n) if rows is None else rows
    out = np.zeros((n, np.asarray(VO).shape[1]), dtype=np.int64)
    abst = np.zeros(n, dtype=bool)
    for f in rows:
        out[f], abst[f] = forecast_row(f, K, Uv, VO, Hs, int(null_raw[f]), int(alpha_raw[f]), adm[f], cnt)
    return out, abst
