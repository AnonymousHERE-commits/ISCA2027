"""Rational oracles for the walk, the posterior, the trust QP and the lumpability defect."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
from typing import List, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------ walk / bank


def exact_walk_matrix(w: np.ndarray) -> List[List[Fraction]]:
    """Dense row-stochastic P = I/2 + S^-1 (I + W) / 2 for a small symmetric W (n x n, w in [0,1])."""
    n = len(w)
    P = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        S = 1 + sum(Fraction(w[i][j]) for j in range(n) if j != i)
        for j in range(n):
            if i == j:
                P[i][j] = Fraction(1, 2) + Fraction(1, 2) / S
            else:
                P[i][j] = Fraction(w[i][j]) / (2 * S)
    return P


def matvec(P, x):
    return [sum(P[i][j] * x[j] for j in range(len(x))) for i in range(len(P))]


def bank_apply(P, x, beta: Fraction, steps: int):
    """A(beta) x = sum_t a_t(beta) P^t x with a_t = (1-beta) beta^t (t < s), a_s = beta^s."""
    out = [Fraction(0)] * len(x)
    cur = list(x)
    for t in range(steps + 1):
        a = beta ** steps if t == steps else (1 - beta) * beta ** t
        out = [o + a * c for o, c in zip(out, cur)]
        cur = matvec(P, cur)
    return out


# ------------------------------------------------------------------ posterior

def exact_marginals(w: Sequence[Fraction], prior: Sequence[Fraction], kmax: int = 3) -> List[Fraction]:
    """Eq. (1) by enumeration of partner sets (small N only)."""
    n = len(w)
    kh = min(kmax, n)
    num = [Fraction(0)] * n
    z = Fraction(0)
    for K in range(1, kh + 1):
        c = prior[K - 1] / comb(n, K)
        for S in combinations(range(n), K):
            prod = Fraction(1)
            for u in S:
                prod *= w[u]
            z += c * prod
            for u in S:
                num[u] += c * prod
    return [x / z for x in num]


# ------------------------------------------------------------------ trust QP

def _solve(A, b):
    """Exact Gaussian elimination, A x = b (A SPD, Fractions)."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = M[c][c]
        if piv == 0:
            raise ZeroDivisionError("singular")
        for r in range(c + 1, n):
            f = M[r][c] / piv
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][k] * x[k] for k in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def exact_nnqp(A: Sequence[Sequence[Fraction]], b: Sequence[Fraction]) -> List[Fraction]:
    """argmin 1/2 x^T A x + b^T x over x >= 0 for SPD A, by enumerating free sets."""
    m = len(b)
    for size in range(0, m + 1):
        for free in combinations(range(m), size):
            x = [Fraction(0)] * m
            if free:
                sub = [[A[i][j] for j in free] for i in free]
                xf = _solve(sub, [-b[i] for i in free])
                if any(v < 0 for v in xf):
                    continue
                for i, v in zip(free, xf):
                    x[i] = v
            g = [sum(A[i][j] * x[j] for j in range(m)) + b[i] for i in range(m)]
            if all(g[i] >= 0 for i in range(m) if i not in free):
                return x
    raise AssertionError("no KKT point found")


# ------------------------------------------------------------------ lumpability (Lemma 3)

def lumpability(P: np.ndarray, labels: Sequence[int], weights: Optional[Sequence[float]] = None):
    """E = P M - M P_c with P_c = R P M, and kappa_boundary = max_i P(leave class)."""
    P = np.asarray(P, dtype=float)
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    n, C = len(labels), len(classes)
    M = np.zeros((n, C))
    for i, l in enumerate(labels):
        M[i, classes.index(l)] = 1.0
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    R = np.zeros((C, n))
    for c in range(C):
        idx = labels == classes[c]
        R[c, idx] = w[idx] / w[idx].sum()
    Pc = R @ P @ M
    E = P @ M - M @ Pc
    kappa = max(sum(P[i, j] for j in range(n) if labels[j] != labels[i]) for i in range(n))
    return {"E": E, "E_inf": np.abs(E).sum(axis=1).max(), "kappa": kappa, "Pc": Pc, "M": M, "R": R}
