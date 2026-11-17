"""Early conformal gate: fused hypotheses and the shared admissible mask G (SPLICE-V §2, §3).

Fusion (per fragment, per dimension d):  P_f = sum_a p_a,  mu_f = div_rne(sum_a p_a mu_a, P_f),
variance v_f = 1/P_f by the BF24 reciprocal unit.  Pair statistic:
  d2_fg = sum_d BF((mu_f - mu_g)^2) * recip(v_f + v_g)    (BF24, fixed order d = 0..3)
admissible iff d2_fg <= tau[stratum(F)] (Mondrian threshold, BF24).  The mask is
symmetric with a zero diagonal and is stored once (DD-12): F_cap x F_cap bits,
row-major, LSB-first in 32-bit words.  Precision codes carry 'gate.prec_frac'
fractional bits, so tau is stored as tau * 2^prec_frac.
"""
from __future__ import annotations

from typing import List

import numpy as np

from . import bf24 as bf
from . import gen
from .arith import Counters, div_rne

ROW_WORDS = gen.F_CAP // 32


def stratum(n_frag: int) -> int:
    bounds = [getattr(gen, f"GATE_STRATUM_BOUND_{i}") for i in range(gen.GATE_N_STRATA - 1)]
    s = 0
    for b in bounds:
        if n_frag >= b:
            s += 1
    return s


def tau_bf(n_frag: int) -> bf.BF:
    s = stratum(n_frag)
    return tuple(gen.GATE_TAU_BF[s])


def fuse(mu: np.ndarray, prec: np.ndarray, ids: np.ndarray, n_frag: int):
    """mu (n, 4) int16 codes, prec (n, 4) u16 precision codes, ids (n,) -> (mu_f, var_f)."""
    mu = np.asarray(mu, dtype=np.int64)
    prec = np.asarray(prec, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    psum = np.zeros((n_frag, 4), dtype=np.int64)
    msum = np.zeros((n_frag, 4), dtype=np.int64)
    m = ids != gen.UNASSIGNED
    np.add.at(psum, ids[m], prec[m])
    np.add.at(msum, ids[m], prec[m] * mu[m])
    mu_f = np.zeros((n_frag, 4), dtype=np.int64)
    var_f: List[List[bf.BF]] = []
    for f in range(n_frag):
        row = []
        for d in range(4):
            if psum[f, d] > 0:
                mu_f[f, d] = div_rne(int(msum[f, d]), int(psum[f, d]))
                row.append(bf.recip(bf.from_int(int(psum[f, d]))))
            else:
                row.append(bf.ZERO)
        var_f.append(row)
    return mu_f, var_f


def pair_d2(mu_f, var_f, f: int, g: int) -> bf.BF | None:
    """BF24 statistic, or None when a fragment carries no hypothesis (never admissible)."""
    acc = bf.ZERO
    for d in range(4):
        vs = bf.add(var_f[f][d], var_f[g][d])
        if bf.is_zero(var_f[f][d]) or bf.is_zero(var_f[g][d]):
            return None
        diff = int(mu_f[f, d]) - int(mu_f[g, d])
        acc = bf.add(acc, bf.mul(bf.from_int(diff * diff), bf.recip(vs)))
    return acc


def admissible_mask(mu_f, var_f, n_frag: int, cnt: Counters | None = None) -> np.ndarray:
    """Boolean (F, F) admissibility, symmetric, zero diagonal."""
    tau = tau_bf(n_frag)
    adm = np.zeros((n_frag, n_frag), dtype=bool)
    for f in range(n_frag):
        for g in range(f + 1, n_frag):
            d2 = pair_d2(mu_f, var_f, f, g)
            ok = d2 is not None and bf.cmp(d2, tau) <= 0
            adm[f, g] = adm[g, f] = ok
    if cnt is not None:
        cnt.add("gate_pairs", n_frag * (n_frag - 1) // 2)
        cnt.add("gate_admissible_pairs", int(adm.sum()) // 2)
    return adm


def pack_mask(adm: np.ndarray) -> np.ndarray:
    """(F, F) bool -> (F_cap * F_cap / 32,) uint32 words (DD-12)."""
    n = adm.shape[0]
    if n > gen.F_CAP:
        raise ValueError("F exceeds F_cap")
    words = np.zeros(gen.F_CAP * ROW_WORDS, dtype=np.uint64)
    for f in range(n):
        for g in np.flatnonzero(adm[f]):
            words[f * ROW_WORDS + (g >> 5)] |= np.uint64(1) << np.uint64(g & 31)
    return words.astype(np.uint32)


def unpack_mask(words: np.ndarray, n_frag: int) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint64)
    adm = np.zeros((n_frag, n_frag), dtype=bool)
    for f in range(n_frag):
        for g in range(n_frag):
            adm[f, g] = bool((int(words[f * ROW_WORDS + (g >> 5)]) >> (g & 31)) & 1)
    return adm


def gate_cycles(n_frag: int, lanes: int) -> int:
    """Modelled C_G(n) = (5 F^2 + 12 * 1856) / n + 2 c_f (SPLICE-V §3)."""
    return (5 * n_frag * n_frag + 12 * gen.N_ANCHOR) // lanes + 2 * 32
