"""Screened trust solve of Eq. (2) in Q8.39 (SPLICE-V Eq. (2), Definition 1, Prop. 2).

Per receiver f:  minimise  1/2 l^T A l + b^T l  over l >= 0, with
  A = g~ 1 1^T + D + Phi Phi^T,   b = -g~ 1 + c_r (1 - p),   D_g = a_r/rho_g + b_r (1 - p_g)
(which is U_f/2 up to a constant).  Steps, all exact integer arithmetic:
  1. D clamped to [2^-4, 2^6]; Phi scaled by 2^-s (RNE) until ||Phi||_F^2 <= 2^4 (counted)
  2. screen (Prop. 2(ii)): drop g when c_r (1 - p_g) > g~ + ceil||phi_g|| * max_h ceil||phi_h||,
     the right side inflated by its rounding bound so every screened unit satisfies the exact test
  3. keep at most Ms survivors by mutual posterior (descending, ties to the smaller source id)
  4. primal active set from l = 0, at most KQP steps; Newton directions solve
     A_FF p = -grad_F by LDL^T with the right side scaled into [2^46, 2^47)
     (48-bit block-floating direction); ratio test by exact cross-multiplication
  5. stored iterates clamped to [0, 2]; termination on KKT residual <= 2^-23;
     a capped row keeps its last (feasible) iterate, counted.
The C++ twin is pl/include/sv_trust.h.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt
from typing import List, Sequence

from . import gen
from .arith import Counters, div_rne, rshift_rne

Q = gen.TRUST_FRAC
ONE = 1 << Q
QMAX = gen.Q39_MAX
QMIN = gen.Q39_MIN


def qmul(a: int, b: int) -> int:
    return rshift_rne(a * b, Q)


def qdiv(a: int, b: int) -> int:
    return div_rne(a << Q, b)


def qsat(x: int, cnt: Counters | None = None) -> int:
    if x > QMAX:
        if cnt is not None:
            cnt.add("trust_q_sat")
        return QMAX
    if x < QMIN:
        if cnt is not None:
            cnt.add("trust_q_sat")
        return QMIN
    return x


def isqrt_ceil(n: int) -> int:
    r = isqrt(n)
    return r if r * r == n else r + 1


@dataclass
class TrustRow:
    sources: List[int]            # survivor source ids, in survivor order
    lam: List[int]                # Q8.39 trusts of the survivors
    status: str = "empty"         # empty | converged | capped | singular
    iterations: int = 0
    screened: int = 0
    counters: Counters = field(default_factory=Counters)

    def source_list(self, ms: int = gen.MS) -> List[int]:
        """Selected sources with positive trust, ascending id, padded with 0xFFFF."""
        sel = sorted(s for s, l in zip(self.sources, self.lam) if l > 0)
        return sel + [0xFFFF] * (ms - len(sel))


def ldl_solve(a: List[List[int]], rhs: List[int], cnt: Counters | None = None):
    """Solve A x = rhs (A SPD in Q8.39, rhs in Q8.39 * 2^s units). None if a pivot is not positive."""
    m = len(rhs)
    lmat = [[0] * m for _ in range(m)]
    dvec = [0] * m
    tmp = [[0] * m for _ in range(m)]       # tmp[j][k] = L_jk * D_k (Q8.39)
    for j in range(m):
        s = a[j][j] << Q
        for k in range(j):
            s -= lmat[j][k] * tmp[j][k]
        dj = rshift_rne(s, Q)
        if dj <= 0:
            if cnt is not None:
                cnt.add("trust_singular")
            return None
        dvec[j] = dj
        for i in range(j + 1, m):
            s = 0
            for k in range(j):
                s += lmat[i][k] * tmp[j][k]
            lmat[i][j] = qdiv(a[i][j] - rshift_rne(s, Q), dj)
            tmp[i][j] = qmul(lmat[i][j], dj)
    y = [0] * m
    for i in range(m):
        s = 0
        for k in range(i):
            s += lmat[i][k] * y[k]
        y[i] = rhs[i] - rshift_rne(s, Q)
    z = [qdiv(y[i], dvec[i]) for i in range(m)]
    x = [0] * m
    for i in range(m - 1, -1, -1):
        s = 0
        for k in range(i + 1, m):
            s += lmat[k][i] * x[k]
        x[i] = z[i] - rshift_rne(s, Q)
    return x


def solve_row(need: int, cand: Sequence[int], p: Sequence[int], rho: Sequence[int], phi: Sequence[Sequence[int]],
              a_r: int = gen.A_R_Q39, b_r: int = gen.B_R_Q39, c_r: int = gen.C_R_Q39,
              ms: int = gen.MS, kqp: int = gen.KQP) -> TrustRow:
    """need, p, rho, phi in Q8.39; cand are source ids aligned with p, rho, phi."""
    cnt = Counters()
    row = TrustRow(sources=[], lam=[], counters=cnt)
    n = len(cand)
    if n == 0:
        return row
    if need < 0:
        need = 0
        cnt.add("trust_need_clamp")
    dvals = []
    for k in range(n):
        if rho[k] > 0:
            dk = qdiv(a_r, rho[k]) + qmul(b_r, ONE - p[k])
        else:
            dk = gen.D_MAX_Q39
            cnt.add("trust_rho_nonpositive")
        if dk < gen.D_MIN_Q39:
            dk = gen.D_MIN_Q39
            cnt.add("trust_D_clamp")
        elif dk > gen.D_MAX_Q39:
            dk = gen.D_MAX_Q39
            cnt.add("trust_D_clamp")
        dvals.append(dk)
    ph = [list(map(int, v)) for v in phi]
    limit = 1 << (gen.PHI_FRO2_MAX_LOG2 + 2 * Q)
    halvings = 0
    while sum(x * x for v in ph for x in v) > limit:
        ph = [[rshift_rne(x, 1) for x in v] for v in ph]
        halvings += 1
    if halvings:
        cnt.add("trust_phi_scaled")
    norms = [isqrt_ceil(sum(x * x for x in v)) for v in ph]
    nmax = max(norms) if norms else 0
    keep = []
    for k in range(n):
        lhs = c_r * (ONE - p[k])
        rhs = need * ONE + norms[k] * nmax
        if lhs > rhs:
            row.screened += 1
        else:
            keep.append(k)
    cnt.add("trust_screened", row.screened)
    keep.sort(key=lambda k: (-p[k], cand[k]))
    if len(keep) > ms:
        cnt.add("trust_ms_cap")
    keep = keep[:ms]
    if not keep:
        row.status = "converged"
        return row
    m = len(keep)
    amat = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            dot = sum(ph[keep[i]][t] * ph[keep[j]][t] for t in range(4))
            v = need + rshift_rne(dot, Q)
            if i == j:
                v += dvals[keep[i]]
            amat[i][j] = qsat(v, cnt)
    bvec = [qsat(-need + qmul(c_r, ONE - p[keep[i]]), cnt) for i in range(m)]
    lam = [0] * m
    active = set(range(m))
    status = "capped"
    steps = 0
    tol = gen.KKT_TOL_Q39
    while True:
        grad = [rshift_rne(sum(amat[i][j] * lam[j] for j in range(m)), Q) + bvec[i] for i in range(m)]
        resid = 0
        for i in range(m):
            r = max(0, -grad[i]) if i in active else abs(grad[i])
            resid = max(resid, r)
        if resid <= tol:
            status = "converged"
            break
        if steps >= kqp:
            status = "capped"
            cnt.add("trust_iteration_cap")
            break
        steps += 1
        free = [i for i in range(m) if i not in active]
        if not free or max(abs(grad[i]) for i in free) <= tol:
            j = min((i for i in active), key=lambda i: (grad[i], i))
            active.discard(j)
            continue
        gm = max(abs(grad[i]) for i in free)
        s = max(0, min(40, 46 - gm.bit_length()))
        sub = [[amat[i][j] for j in free] for i in free]
        ps = ldl_solve(sub, [(-grad[i]) << s for i in free], cnt)
        if ps is None:
            status = "singular"
            break
        block = None
        for t, i in enumerate(free):
            if ps[t] < 0:
                # alpha_i = lam_i * 2^s / (-ps_t); compare with the current minimum exactly
                if block is None:
                    block = (t, i)
                else:
                    bt, bi = block
                    lhs = lam[i] * (-ps[bt])
                    rhs = lam[bi] * (-ps[t])
                    if lhs < rhs or (lhs == rhs and i < bi):
                        block = (t, i)
        full = block is None or (lam[block[1]] << s) >= -ps[block[0]]
        if full:
            for t, i in enumerate(free):
                lam[i] += rshift_rne(ps[t], s)
        else:
            bt, bi = block
            lj = lam[bi]
            for t, i in enumerate(free):
                if i != bi:
                    lam[i] += div_rne(lj * ps[t], -ps[bt])
            lam[bi] = 0
            active.add(bi)
        for i in range(m):
            if lam[i] < 0:
                lam[i] = 0
                cnt.add("trust_clamp_lo")
            elif lam[i] > gen.LAMBDA_MAX_Q39:
                lam[i] = gen.LAMBDA_MAX_Q39
                cnt.add("trust_clamp_hi")
    row.sources = [int(cand[k]) for k in keep]
    row.lam = lam
    row.status = status
    row.iterations = steps
    cnt.add(f"trust_{status}")
    return row


def objective(lam: Sequence[int], amat, bvec) -> float:
    """Diagnostic only (float): 1/2 l^T A l + b^T l in real units."""
    m = len(lam)
    x = [l / ONE for l in lam]
    a = [[amat[i][j] / ONE for j in range(m)] for i in range(m)]
    b = [v / ONE for v in bvec]
    return 0.5 * sum(x[i] * a[i][j] * x[j] for i in range(m) for j in range(m)) + sum(b[i] * x[i] for i in range(m))


def innovation(lam_row: TrustRow, tv_by_slot: Sequence[Sequence[int]], h: Sequence[int],
               cnt: Counters | None = None) -> List[int]:
    """Toy one-hop innovation (dev profile only):
    dv_f,c = sat_sym(rne(sum_slot l_slot (tv_slot,c - h_f,c) >> 39)) over the source list."""
    lam_of = {s: l for s, l in zip(lam_row.sources, lam_row.lam)}
    srcs = lam_row.source_list()
    out = []
    for c in range(len(h)):
        acc = 0
        for slot, sid in enumerate(srcs):
            if sid != 0xFFFF:
                acc += lam_of[sid] * (int(tv_by_slot[slot][c]) - int(h[c]))
        v = rshift_rne(acc, Q)
        if v > 127 or v < -127:
            if cnt is not None:
                cnt.add("sat_dv")
            v = max(-127, min(127, v))
        out.append(v)
    return out
