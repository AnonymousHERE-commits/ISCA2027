"""Word-level ABI of the PL HLS kernels (m_axi buffers of 32-bit words).

Shared by the PS host (ps/splicev_ps), the HLS C-simulation testbenches (via
tools/make_vectors.py) and the reference, so a board run, a C-sim and the
reference compare the same bytes.  One element per word unless stated.

sv_reserve   cells[n] = e | t << 8                -> picks[count], stats[4] = n_picks, scans_lo, scans_hi, 0
sv_linkage   edges[3E] = a, b, q; anchors[n] = active | elig << 8
             -> ids[n], order[n], stats[8] = F, rounds, work_lo, work_hi, budget_hit, overflow, dup_error, cap_hit
sv_gate      anchors[5n] = mu0|mu1<<16, mu2|mu3<<16, p0|p1<<16, p2|p3<<16, id
             -> mask[F_cap * F_cap/32], fused[12 F_cap] = mu_f[4], var_m[4], var_e[4], stats[2] = pairs, F
sv_exchange1 partners/evidence[F_cap N_cap], npart/need/rho[F_cap], phi[4 F_cap]
             -> sources[MS F_cap], lam[2 MS F_cap] (int64 lo, hi), status[F_cap]
sv_exchange2 sources, lam, tv[F_cap MS 12] (48 int8 packed), h[F_cap 12] -> dv[F_cap 12], stats[2] = sat, 0
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import gen
from .units import pack_u8, unpack


def u32(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint32)


def s32(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64) & 0xFFFFFFFF
    return np.where(a >= (1 << 31), a - (1 << 32), a)


def reserve_in(e, t) -> np.ndarray:
    return u32(np.asarray(e, dtype=np.int64) | (np.asarray(t, dtype=np.int64) << 8))


def linkage_in(edges, active, elig):
    return u32(np.asarray(edges, dtype=np.int64).reshape(-1)), u32(np.asarray(active, dtype=np.int64) | (np.asarray(elig, dtype=np.int64) << 8))


def gate_in(mu, prec, ids) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.int64) & 0xFFFF
    prec = np.asarray(prec, dtype=np.int64) & 0xFFFF
    n = mu.shape[0]
    w = np.zeros((n, 5), dtype=np.int64)
    w[:, 0] = mu[:, 0] | (mu[:, 1] << 16)
    w[:, 1] = mu[:, 2] | (mu[:, 3] << 16)
    w[:, 2] = prec[:, 0] | (prec[:, 1] << 16)
    w[:, 3] = prec[:, 2] | (prec[:, 3] << 16)
    w[:, 4] = np.asarray(ids, dtype=np.int64)
    return u32(w.ravel())


def lam_words(lam_slot) -> np.ndarray:
    lam = np.asarray(lam_slot, dtype=np.int64).ravel()
    out = np.zeros(2 * lam.size, dtype=np.int64)
    out[0::2] = lam & 0xFFFFFFFF
    out[1::2] = (lam >> 32) & 0xFFFFFFFF
    return u32(out)


def lam_from_words(words) -> np.ndarray:
    """(lo, hi) word pairs -> int64 (two's complement, exact)."""
    w = (np.asarray(words, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
    v = w[0::2] | (w[1::2] << np.uint64(32))
    return v.view(np.int64)


def pad_rows(a, rows: int, fill=0) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    out = np.full((rows,) + a.shape[1:], fill, dtype=np.int64)
    out[:a.shape[0]] = a
    return out


def i8_rows(a, rows: int) -> np.ndarray:
    """(n, 48) int8 -> (rows, 12) packed words."""
    a = pad_rows(a, rows)
    return u32(np.asarray([pack_u8(r) for r in a], dtype=np.int64))


def i8_rows_from(words, n: int, width: int = 48) -> np.ndarray:
    w = np.asarray(words, dtype=np.int64).reshape(-1, width // 4)
    return np.asarray([unpack(r, width, "8", True) for r in w[:n]], dtype=np.int64)


def exchange2_in(sources, lam_slot, tv, h) -> Dict[str, np.ndarray]:
    F, MS = gen.F_CAP, gen.MS
    src = pad_rows(sources, F, 0xFFFF)
    lam = pad_rows(lam_slot, F)
    tvp = pad_rows(np.asarray(tv, dtype=np.int64), F)
    tvw = u32(np.asarray([pack_u8(tvp[f, s]) for f in range(F) for s in range(MS)], dtype=np.int64))
    return {"sources": u32(src.ravel()), "lam": lam_words(lam), "tv": tvw.ravel(), "h": i8_rows(h, F).ravel()}
