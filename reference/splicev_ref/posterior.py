"""Partner posterior by elementary symmetric polynomials (SPLICE-V Eq. (1), Prop. 2).

For a receiver row with units u = 1..N, weights w_u = e^(s_u) (BF24, s in Q7.8)
and prior pi(K), K = 1..min(Kmax, N):
  p_u = (1/Z) sum_K pi(K)/C(N,K) * w_u * e_(K-1)(w_-u),   Z = sum_K pi(K)/C(N,K) e_K(w)
computed by prefix and suffix recursions on nonnegative BF24 numbers in a
fixed operation order (the C++ twin is pl/include/sv_posterior.h).
"""
from __future__ import annotations

from math import comb
from typing import List, Sequence, Tuple

from . import bf24 as bf
from . import gen

ONE = bf.from_int(1)


def prior_row(stratum: int = 0) -> List[bf.BF]:
    base = stratum * gen.KMAX
    return [(gen.PRIOR_BF[base + k][0], gen.PRIOR_BF[base + k][1]) for k in range(gen.KMAX)]


def esp_marginals(w: Sequence[bf.BF], prior: Sequence[bf.BF], kmax: int = gen.KMAX) -> Tuple[List[bf.BF], bf.BF]:
    n = len(w)
    if n == 0:
        return [], bf.ZERO
    kh = min(kmax, n)
    pre = [[ONE] + [bf.ZERO] * kh]
    for u in range(1, n + 1):
        prev = pre[-1]
        row = [ONE]
        for k in range(1, kh + 1):
            row.append(bf.add(prev[k], bf.mul(w[u - 1], prev[k - 1])))
        pre.append(row)
    suf = [None] * (n + 2)
    suf[n + 1] = [ONE] + [bf.ZERO] * kh
    for u in range(n, 0, -1):
        nxt = suf[u + 1]
        row = [ONE]
        for k in range(1, kh + 1):
            row.append(bf.add(nxt[k], bf.mul(w[u - 1], nxt[k - 1])))
        suf[u] = row
    coef = [None] + [bf.mul(prior[k - 1], bf.recip(bf.from_int(comb(n, k)))) for k in range(1, kh + 1)]
    z = bf.ZERO
    for k in range(1, kh + 1):
        z = bf.add(z, bf.mul(coef[k], pre[n][k]))
    rz = bf.recip(z)
    out = []
    for u in range(1, n + 1):
        acc = bf.ZERO
        for k in range(1, kh + 1):
            e = bf.ZERO
            for i in range(k):
                e = bf.add(e, bf.mul(pre[u - 1][i], suf[u + 1][k - 1 - i]))
            acc = bf.add(acc, bf.mul(coef[k], e))
        out.append(bf.mul(bf.mul(w[u - 1], acc), rz))
    return out, z


def row_marginals(evidence_q78: Sequence[int], stratum: int = 0) -> List[bf.BF]:
    w = [bf.exp_q78(int(s)) for s in evidence_q78]
    p, _z = esp_marginals(w, prior_row(stratum))
    return p


def mutual_posteriors(partners: Sequence[Sequence[int]], marginals: Sequence[Sequence[bf.BF]]):
    """p_fg = min(p^(f)_fg, p^(g)_fg); partners[f] lists row f's units (ascending ids)."""
    index = [{g: k for k, g in enumerate(row)} for row in partners]
    out = []
    for f, row in enumerate(partners):
        r = []
        for k, g in enumerate(row):
            a = marginals[f][k]
            kk = index[g].get(f) if g < len(index) else None
            b = marginals[g][kk] if kk is not None else bf.ZERO
            r.append(a if bf.cmp(a, b) <= 0 else b)
        out.append(r)
    return out
