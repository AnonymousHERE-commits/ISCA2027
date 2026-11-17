"""Exchange decisions in float64 numpy (model DFP32): Eq. (1) partner posterior, Eq. (2) trust
solve and the innovation dV, per image.

SPLICE-V section 2: memberships, partner lists, active sets and admissibility decisions are
constants of the backward pass.  This module makes those decisions (and gives the forward
values); exchange_torch.py recomputes the marginals and solves the optimality system of
Eq. (2) on the fixed active sets inside autograd, so gradients reach the exchange.  The
definitions follow the integer reference (reference/splicev_ref/posterior.py, trust.py,
exchange.py) without the integer arithmetic: ESP marginals with the partner-count prior,
mutual posterior min(p_fg, p_gf), D clamp, Phi scaling, the Proposition 1(ii) screen, the Ms
cap by mutual posterior, an exact solution of the box-free QP over at most Ms sources
(enumeration of active sets, unique by strict convexity), the [0, 2] iterate clamp, and
dV_f = sum_g lam_fg (VO_g - Hs_f).

Eq. (2), U_f(lam) = zeta_f (1 - sum_g lam_g)^2 + lam^T Sigma_f lam + 2 c sum_g lam_g (1 - p_fg)
with Sigma_f = D_f + Phi_f Phi_f^T, is solved as
  min 1/2 lam^T A lam + b^T lam, lam >= 0,  A = zeta 1 1^T + D + Phi Phi^T,  b = -zeta 1 + c (1 - p),
which is U_f / 2 up to the constant zeta / 2 (the terminal system zeta 1 1^T + Sigma_f of
Proposition 1(iv)).
"""
from __future__ import annotations

from itertools import product
from math import comb
from typing import Dict, List, Sequence

import numpy as np

from . import REPO  # noqa: F401  (path setup)
from splicev_ref import gen

_SUBSETS = {}


def _subsets(m: int) -> np.ndarray:
    if m not in _SUBSETS:
        _SUBSETS[m] = np.array(list(product([0, 1], repeat=m)), dtype=bool)[1:]   # non-empty active sets
    return _SUBSETS[m]


def esp_marginals(s: np.ndarray, prior: Sequence[float], kmax: int = 3) -> np.ndarray:
    """Marginals p_u of one receiver row, s (n,) evidence in natural-log units."""
    n = s.size
    if n == 0:
        return np.zeros(0)
    kh = min(kmax, n)
    m = s.max()
    w = np.exp(s - m)                                   # w_u e^{-m}; factor e^{-Km} folded into coef
    pre = np.zeros((n + 1, kh + 1)); pre[0, 0] = 1.0
    for u in range(1, n + 1):
        pre[u, 0] = 1.0
        pre[u, 1:] = pre[u - 1, 1:] + w[u - 1] * pre[u - 1, :-1]
    suf = np.zeros((n + 2, kh + 1)); suf[n + 1, 0] = 1.0
    for u in range(n, 0, -1):
        suf[u, 0] = 1.0
        suf[u, 1:] = suf[u + 1, 1:] + w[u - 1] * suf[u + 1, :-1]
    logc = np.array([np.log(max(prior[k - 1], 1e-300)) - np.log(comb(n, k)) + k * m for k in range(1, kh + 1)])
    c = np.exp(logc - logc.max())
    z = sum(c[k - 1] * pre[n, k] for k in range(1, kh + 1))
    p = np.zeros(n)
    for u in range(1, n + 1):
        acc = 0.0
        for k in range(1, kh + 1):
            e = sum(pre[u - 1, i] * suf[u + 1, k - 1 - i] for i in range(k))
            acc += c[k - 1] * e
        p[u - 1] = w[u - 1] * acc / z
    return p


def esp_marginals_batch(S: np.ndarray, mask: np.ndarray, prior: Sequence[float], kmax: int = 3) -> np.ndarray:
    """esp_marginals for every receiver row of an image at once.

    S (F, n) evidence, mask (F, n) valid partner slots (a prefix of each row).  Padding has
    weight zero, so it changes no elementary symmetric polynomial; the per-row partner count
    enters only the prior coefficients.  Returns (F, n) marginals, zero on padding."""
    S = np.asarray(S, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    F, n = S.shape
    out = np.zeros((F, n))
    if F == 0 or n == 0:
        return out
    nv = mask.sum(axis=1)
    m = np.where(mask, S, -np.inf).max(axis=1)
    m = np.where(nv > 0, m, 0.0)
    w = np.where(mask, np.exp(np.where(mask, S - m[:, None], 0.0)), 0.0)
    kh = min(kmax, n)
    pre = np.zeros((F, n + 1, kh + 1)); pre[:, :, 0] = 1.0
    for u in range(1, n + 1):
        pre[:, u, 1:] = pre[:, u - 1, 1:] + w[:, u - 1, None] * pre[:, u - 1, :-1]
    suf = np.zeros((F, n + 2, kh + 1)); suf[:, :, 0] = 1.0
    for u in range(n, 0, -1):
        suf[:, u, 1:] = suf[:, u + 1, 1:] + w[:, u - 1, None] * suf[:, u + 1, :-1]
    ks = np.arange(1, kh + 1)
    log_comb = np.array([[np.log(comb(int(v), int(k))) if k <= v else 0.0 for k in ks] for v in nv]).reshape(F, kh)
    pri = np.log(np.maximum(np.asarray(prior, dtype=np.float64)[:kh], 1e-300))
    logc = pri[None, :] - log_comb + ks[None, :] * m[:, None]
    valid_k = ks[None, :] <= np.minimum(kmax, nv)[:, None]
    logc = np.where(valid_k, logc, -np.inf)
    top = np.where(nv > 0, logc.max(axis=1), 0.0)
    c = np.where(valid_k, np.exp(logc - top[:, None]), 0.0)                       # (F, kh)
    z = (c * pre[:, n, 1:]).sum(axis=1)
    E = np.zeros((F, n, kh))
    for k in range(1, kh + 1):
        for i in range(k):
            E[:, :, k - 1] += pre[:, 0:n, i] * suf[:, 2:n + 2, k - 1 - i]
    acc = (E * c[:, None, :]).sum(axis=2)
    out = np.where(mask & (z[:, None] > 0), w * acc / np.where(z > 0, z, 1.0)[:, None], 0.0)
    return out


def solve_trust_row(need: float, p: np.ndarray, rho: np.ndarray, phi: np.ndarray, src: Sequence[int],
                    a_r: float, b_r: float, c_r: float, ms: int, d_clamp=(2.0 ** -4, 2.0 ** 6),
                    phi_fro2_max: float = 16.0, lam_max: float = 2.0) -> Dict:
    """One receiver row.  Returns {'sources': [...], 'lam': [...], 'screened': n}."""
    n = p.size
    out = {"sources": [], "lam": [], "screened": 0}
    if n == 0:
        return out
    need = max(0.0, float(need))
    d = np.where(rho > 0, a_r / np.maximum(rho, 1e-300) + b_r * (1.0 - p), d_clamp[1])
    d = np.clip(d, *d_clamp)
    ph = np.asarray(phi, dtype=np.float64).copy()
    while (ph * ph).sum() > phi_fro2_max:
        ph *= 0.5
    norms = np.sqrt((ph * ph).sum(axis=1))
    nmax = norms.max() if n else 0.0
    keep = [k for k in range(n) if not (c_r * (1.0 - p[k]) > need + norms[k] * nmax)]
    out["screened"] = n - len(keep)
    keep.sort(key=lambda k: (-p[k], src[k]))
    keep = keep[:ms]
    if not keep:
        return out
    kk = np.array(keep)
    m = kk.size
    A = need + ph[kk] @ ph[kk].T + np.diag(d[kk])
    b = -need + c_r * (1.0 - p[kk])
    best, best_obj = np.zeros(m), 0.0                   # lam = 0 is feasible with objective 0
    S = _subsets(m)
    M = np.where(S[:, :, None] & S[:, None, :], A[None], 0.0) + np.where(np.eye(m, dtype=bool)[None] & ~S[:, :, None], 1.0, 0.0)
    rhs = np.where(S, -b[None], 0.0)
    lam = np.linalg.solve(M, rhs[..., None])[..., 0]
    grad = lam @ A + b
    ok = np.all(np.where(S, lam >= -1e-12, True), axis=1) & np.all(np.where(~S, grad >= -1e-10, True), axis=1)
    if ok.any():
        obj = 0.5 * np.einsum("si,ij,sj->s", lam, A, lam) + lam @ b
        obj = np.where(ok, obj, np.inf)
        i = int(np.argmin(obj))
        if obj[i] < best_obj:
            best = np.clip(lam[i], 0.0, None)
    best = np.clip(best, 0.0, lam_max)
    for j in range(m):
        if best[j] > 0:
            out["sources"].append(int(src[kk[j]]))
            out["lam"].append(float(best[j]))
    order = np.argsort(out["sources"])
    out["sources"] = [out["sources"][i] for i in order]
    out["lam"] = [out["lam"][i] for i in order]
    return out


def exchange(adm: np.ndarray, evidence: np.ndarray, need: np.ndarray, rho: np.ndarray, phi: np.ndarray,
             vo_t: np.ndarray, hs: np.ndarray, prior: Sequence[float], a_r: float = 0.25, b_r: float = 1.0,
             c_r: float = 0.5, ms: int = gen.MS, n_cap: int = gen.N_CAP, kmax: int = gen.KMAX) -> Dict:
    """Exchange of one image.

    adm (F, F) bool symmetric, evidence (F, F) natural-log pair evidence, need/rho (F,),
    phi (F, 4), vo_t (F, Ct) translated values, hs (F, Ct) receiver states.
    Returns dV (F, Ct), lam (F, F), mutual (F, F), partners, marginals."""
    adm = np.asarray(adm, dtype=bool)
    F = adm.shape[0]
    a = adm & ~np.eye(F, dtype=bool)
    rank = np.cumsum(a, axis=1) - 1                       # slot of g among f's admissible partners
    a &= rank < n_cap                                     # partners: the first n_cap in id order
    n = int(a.sum(axis=1).max()) if F else 0
    partners = [[int(g) for g in np.flatnonzero(a[f])] for f in range(F)]
    slot = np.where(a, rank, -1)
    S = np.zeros((F, n)); mask = np.zeros((F, n), dtype=bool)
    fi, gi = np.nonzero(a)
    S[fi, slot[fi, gi]] = np.asarray(evidence, dtype=np.float64)[fi, gi]
    mask[fi, slot[fi, gi]] = True
    pm = esp_marginals_batch(S, mask, prior, kmax)
    P = np.zeros((F, F))
    P[fi, gi] = pm[fi, slot[fi, gi]]
    mutual = np.where(a & a.T, np.minimum(P, P.T), 0.0)
    marg = [pm[f, :len(partners[f])] for f in range(F)]
    lam = np.zeros((F, F))
    screened = 0
    for f in range(F):
        cand = partners[f]
        if not cand:
            continue
        cand_a = np.asarray(cand)
        row = solve_trust_row(need[f], mutual[f, cand_a], np.asarray(rho)[cand_a], np.asarray(phi)[cand_a].reshape(-1, 4),
                              cand, a_r, b_r, c_r, ms)
        screened += row["screened"]
        for g, l in zip(row["sources"], row["lam"]):
            lam[f, g] = l
    dv = lam @ vo_t - lam.sum(axis=1, keepdims=True) * hs
    return {"dV": dv, "lam": lam, "mutual": mutual, "partners": partners, "marginals": marg, "screened": screened}


def exchange_job(args) -> Dict:
    """Process-pool entry: exchange(*args) without the bulky per-row lists."""
    ex = exchange(*args)
    return {"dV": ex["dV"], "lam": ex["lam"], "mutual": ex["mutual"], "screened": ex["screened"]}


def decisions(adm: np.ndarray, lam: np.ndarray, n_cap: int = gen.N_CAP, ms: int = gen.MS) -> Dict:
    """Index form of one image's exchange decisions, for the differentiable recomputation.

    pidx (F, n)  partner ids of every receiver (the first n_cap admissible ids), zero padded
    pmask (F, n) valid partner slots
    back (F, n)  slot of the receiver in that partner's own list, -1 when it is not there
    sidx (F, ms) slot (in the receiver's partner list) of every source with lam > 0
    smask (F, ms) valid source slots"""
    adm = np.asarray(adm, dtype=bool)
    F = adm.shape[0]
    a = adm & ~np.eye(F, dtype=bool)
    rank = np.cumsum(a, axis=1) - 1
    a &= rank < n_cap
    n = int(a.sum(axis=1).max()) if F else 0
    slot = np.where(a, rank, -1)
    pidx = np.zeros((F, n), dtype=np.int64)
    pmask = np.zeros((F, n), dtype=bool)
    fi, gi = np.nonzero(a)
    pidx[fi, slot[fi, gi]] = gi
    pmask[fi, slot[fi, gi]] = True
    rows = np.arange(F)[:, None]
    back = np.where(pmask, slot[pidx, rows], -1) if n else np.zeros((F, 0), dtype=np.int64)
    act = (np.asarray(lam) > 0) & a
    ra = np.cumsum(act, axis=1) - 1
    sidx = np.zeros((F, ms), dtype=np.int64)
    smask = np.zeros((F, ms), dtype=bool)
    fa, ga = np.nonzero(act & (ra < ms))
    sidx[fa, ra[fa, ga]] = slot[fa, ga]
    smask[fa, ra[fa, ga]] = True
    return {"pidx": pidx, "pmask": pmask, "back": back, "sidx": sidx, "smask": smask, "n": n}
