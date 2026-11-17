"""Differentiable exchange (SPLICE-V section 2, 'Training and arms').

"No gradient passes through the linkage: memberships, active sets and admissibility
decisions are constants of the backward pass, and gradients reach the exchange through (1)
and the optimality system of (2)."  exchange_float.py makes the decisions per image; this
module recomputes, for all fragments of a batch at once and in float64,

  p^(f)_u   the partner marginals of Eq. (1) by the prefix/suffix recursion (differentiable
            in the pair evidence),
  p_fg      the mutual posterior min(p^(f)_g, p^(g)_f),
  lam_f     the solution of the stationarity system of Eq. (2) on the receiver's fixed
            active set, (zeta 1 1^T + D + Phi Phi^T)_SS lam_S = zeta 1 - c (1 - p_S),
  dH_f      sum_g lam_fg (V_g O_T - H_f^s),

so that the forward values equal exchange_float.exchange and the backward pass reaches the
evidence scorer, the need / reliability / moment heads and the translation map.
"""
from __future__ import annotations

from math import comb, log
from typing import Dict, List, Sequence

import numpy as np
import torch


def esp_marginals(S: torch.Tensor, mask: torch.Tensor, prior: Sequence[float], kmax: int = 3) -> torch.Tensor:
    """Eq. (1) for every row: S (R, n) evidence at the partner slots, mask (R, n) valid slots.
    Padding has weight zero.  The row maximum used for scaling is a constant of the graph."""
    R, n = S.shape
    if R == 0 or n == 0:
        return S.new_zeros((R, n))
    zero = torch.zeros_like(S)
    nv = mask.sum(1)
    m = torch.where(mask, S, torch.full_like(S, -1e30)).amax(1)
    m = torch.where(nv > 0, m, torch.zeros_like(m)).detach()
    w = torch.where(mask, torch.exp(torch.where(mask, S - m[:, None], zero)), zero)
    kh = min(kmax, n)
    first = torch.cat([S.new_ones((R, 1)), S.new_zeros((R, kh))], 1)            # e_0 = 1, e_k = 0
    lead = S.new_zeros((R, 1))
    pre = [first]
    for u in range(n):                                                         # e_k of the first u + 1 weights
        prev = pre[-1]
        pre.append(prev + w[:, u:u + 1] * torch.cat([lead, prev[:, :-1]], 1))
    suf = [first]
    for u in range(n - 1, -1, -1):                                             # e_k of the weights after u
        nxt = suf[-1]
        suf.append(nxt + w[:, u:u + 1] * torch.cat([lead, nxt[:, :-1]], 1))
    suf = suf[::-1]                                                            # suf[u] = e_k(w_u, ..., w_{n-1})
    P = torch.stack(pre[:n], 1)                                                # (R, n, kh + 1): before slot u
    Q = torch.stack(suf[1:], 1)                                                # (R, n, kh + 1): after slot u
    # coefficients c_K = prior_K / C(N_f, K) e^{K m}, scaled by their largest (constants)
    nv_np = np.asarray(nv.detach().cpu().numpy(), dtype=np.int64)
    m_np = np.asarray(m.detach().cpu().numpy(), dtype=np.float64)
    ks = np.arange(1, kh + 1)
    logc = np.full((R, kh), -np.inf)
    for k in ks:
        ok = nv_np >= k
        lc = np.array([log(comb(int(v), int(k))) if v >= k else 0.0 for v in nv_np])
        logc[:, k - 1] = np.where(ok, log(max(float(prior[k - 1]), 1e-300)) - lc + k * m_np, -np.inf)
    top = np.where(nv_np > 0, logc.max(axis=1), 0.0)
    c = torch.as_tensor(np.where(np.isfinite(logc), np.exp(logc - top[:, None]), 0.0), dtype=S.dtype, device=S.device)
    z = (c * pre[n][:, 1:]).sum(1)
    acc = torch.zeros_like(S)
    for k in range(1, kh + 1):
        ek = torch.zeros_like(S)
        for i in range(k):
            ek = ek + P[:, :, i] * Q[:, :, k - 1 - i]                          # e_{k-1} without slot u
        acc = acc + c[:, k - 1:k] * ek
    zs = torch.where(z > 0, z, torch.ones_like(z))
    return torch.where(mask & (z[:, None] > 0), w * acc / zs[:, None], zero)


def trust_kkt(need: torch.Tensor, p: torch.Tensor, rho: torch.Tensor, phi: torch.Tensor, smask: torch.Tensor,
              a_r: float, b_r: float, c_r: float, d_clamp=(2.0 ** -4, 2.0 ** 6), lam_max: float = 2.0) -> torch.Tensor:
    """Stationarity system of Eq. (2) on fixed active sets.

    need (R,), p / rho (R, Ms) of the active sources, phi (R, Ms, 4) already scaled,
    smask (R, Ms).  Returns lam (R, Ms), zero on inactive slots."""
    ms = p.shape[1]
    need = need.clamp(min=0.0)
    D = (a_r / rho + b_r * (1.0 - p)).clamp(d_clamp[0], d_clamp[1])
    eye = torch.eye(ms, dtype=p.dtype, device=p.device)[None]
    A = need[:, None, None] + phi @ phi.permute(0, 2, 1) + eye * D[:, :, None]
    both = smask[:, :, None] & smask[:, None, :]
    A = torch.where(both, A, eye + torch.zeros_like(A))
    rhs = torch.where(smask, need[:, None] - c_r * (1.0 - p), torch.zeros_like(p))
    lam = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
    return lam.clamp(0.0, lam_max)


def exchange_batch(items: List[Dict], prior: Sequence[float], a_r: float, b_r: float, c_r: float, kmax: int,
                   ms: int, phi_fro2_max: float = 16.0) -> List[Dict]:
    """All images of a batch at once.

    Each item: {'ev' (F, F), 'need' (F,), 'rho' (F,), 'phi' (F, 4), 'vo_t' (F, Ct), 'hs' (F, Ct)}
    torch tensors plus 'dec', the numpy decisions of exchange_float.decisions.
    Returns per image {'dV' (F, Ct), 'lam' (F, Ms), 'src' (F, Ms) source ids, 'smask',
    'mutual' (F, n), 'pmask'} with gradients to every tensor input."""
    if not items:
        return []
    dev = items[0]["ev"].device
    n_max = max(it["dec"]["n"] for it in items)
    offs, tot = [], 0
    for it in items:
        offs.append(tot)
        tot += it["ev"].shape[0]
    S_rows, pm_rows, grow_rows, back_rows, sidx_rows, sm_rows = [], [], [], [], [], []
    for it, off in zip(items, offs):
        d = it["dec"]
        F, n = d["pidx"].shape
        pad = n_max - n
        pidx = np.pad(d["pidx"], ((0, 0), (0, pad)))
        pmask = np.pad(d["pmask"], ((0, 0), (0, pad)))
        back = np.pad(d["back"], ((0, 0), (0, pad)), constant_values=-1)
        rows = torch.arange(F, device=dev)[:, None]
        cols = torch.as_tensor(pidx, device=dev)
        S_rows.append(it["ev"].double()[rows, cols] if n_max else it["ev"].new_zeros((F, 0)).double())
        pm_rows.append(pmask)
        grow_rows.append(pidx + off)
        back_rows.append(back)
        sidx_rows.append(d["sidx"])
        sm_rows.append(d["smask"])
    S = torch.cat(S_rows, 0)
    pmask = torch.as_tensor(np.concatenate(pm_rows, 0), device=dev)
    grow = torch.as_tensor(np.concatenate(grow_rows, 0), device=dev)
    back = torch.as_tensor(np.concatenate(back_rows, 0), device=dev)
    sidx = torch.as_tensor(np.concatenate(sidx_rows, 0), device=dev)
    smask = torch.as_tensor(np.concatenate(sm_rows, 0), device=dev)
    cat = lambda k: torch.cat([it[k].double() for it in items], 0)
    need, rho, phi, vo_t, hs = cat("need"), cat("rho"), cat("phi"), cat("vo_t"), cat("hs")

    pm = esp_marginals(S, pmask, prior, kmax)                                   # (R, n)
    R = S.shape[0]
    rows = torch.arange(R, device=dev)[:, None]
    if n_max:
        has_back = pmask & (back >= 0)
        p_back = pm[grow, back.clamp(min=0)]
        mutual = torch.where(has_back, torch.minimum(pm, p_back), torch.zeros_like(pm))
        g_act = grow[rows, sidx]                                                # (R, Ms) global source rows
        p_act = torch.where(smask, mutual[rows, sidx], torch.zeros_like(mutual[rows, sidx]))
        # Phi scaling of the row: halve until the candidates' squared Frobenius norm is <= 2^4
        fro2 = (torch.where(pmask, (phi ** 2).sum(1)[grow], torch.zeros_like(pm))).sum(1).detach()
        halv = torch.ceil(torch.log(torch.clamp(fro2 / phi_fro2_max, min=1.0)) / log(4.0) - 1e-12).clamp(min=0.0)
        scale = 0.5 ** halv
    else:
        mutual = pm
        g_act = sidx
        p_act = S.new_zeros((R, ms))
        scale = S.new_ones((R,))
    rho_act = torch.where(smask, rho[g_act], torch.ones_like(p_act))
    phi_act = torch.where(smask[:, :, None], phi[g_act] * scale[:, None, None], torch.zeros_like(phi[g_act]))
    lam = trust_kkt(need, p_act, rho_act, phi_act, smask, a_r, b_r, c_r)
    lam = torch.where(smask, lam, torch.zeros_like(lam))
    dV = (lam[:, :, None] * vo_t[g_act]).sum(1) - lam.sum(1, keepdim=True) * hs
    out = []
    for it, off in zip(items, offs):
        F = it["ev"].shape[0]
        sl = slice(off, off + F)
        out.append({"dV": dV[sl].to(it["hs"].dtype), "lam": lam[sl], "src": g_act[sl] - off, "smask": smask[sl],
                    "mutual": mutual[sl], "pmask": pmask[sl]})
    return out
