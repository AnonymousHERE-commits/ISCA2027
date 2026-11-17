"""Closed-form model quantities of the manuscript (all provenance 'A': analytical).

These reproduce the paper's arithmetic so that a stated number can be audited;
they are hypotheses to test on the board (P3-P8), never status values.
"""
from __future__ import annotations

import math
from typing import Dict


def gate_cycles(F: int = 256, n: int = 128, anchors: int = 1856, c_f: int = 32) -> float:
    """C_G(n) = (5 F^2 + 12 * anchors) / n + 2 c_f."""
    return (5 * F * F + 12 * anchors) / n + 2 * c_f


def exchange_cycles(F: int = 256, N: int = 128, n: int = 128, kmax: int = 3, ms: int = 8, kqp: int = 16,
                    c_r: int = 8, c_w: int = 10, c_f: int = 32, e_dup: int = 0) -> float:
    """Eq. (6), Proposition 4(i)."""
    ceil_fn = math.ceil(F / n)
    return (67 * F * N / n + ceil_fn * N * kmax ** 2 + 144 * F * ms / n + 6 * c_f
            + (F / 32) * min(e_dup, F * N / 8)
            + kqp * max(ceil_fn * (5 + 4 * ms + c_r) * ms, ms * (c_r + 2 * c_w)))


def exchange_stage_ms(cx: float, f_pl: float = 300e6, macs_priority: float = 7360 * 256 * 8,
                      tiles: int = 34, gemm_rate: float = 154.0, f_aie: float = 1e9,
                      h_us: float = 12.5, w_us: float = 50.0) -> Dict[str, float]:
    """X = C_X / f_PL + t_U + 2h + w and g2 = X + 2h (Proposition 4(ii))."""
    t_u = macs_priority / (tiles * gemm_rate * f_aie)
    x = cx / f_pl + t_u + 2 * h_us * 1e-6 + w_us * 1e-6
    return {"t_U_us": t_u * 1e6, "X_ms": x * 1e3, "g2_ms": (x + 2 * h_us * 1e-6) * 1e3}


def lemma1_row_slice_threshold(R: int = 256, eta_k: float = 0.5, B: int = 8, tiles: int = 34) -> float:
    """Row slicing is intake-bound whenever n < tiles R eta_k / B."""
    return tiles * R * eta_k / B


def lemma1_row_slice_ceiling(n: int = 400, R: int = 256, B: int = 8, tiles: int = 34) -> float:
    """eta <= (B / R) (n / tiles)."""
    return (B / R) * (n / tiles)


def lemma1_output_ceiling(k: int = 3, c: int = 1, Bo: int = 8, R: int = 256) -> float:
    return k * k * c * Bo / R


def lemma1_time(a: int, b: int, k: int, c: int, a_in: int, sigma: int, Mb: int,
                R: int, eta_k: float, B: int = 8, Bo: int = 8) -> Dict[str, float]:
    """Lower bound on one tile's cycles and its useful efficiency (Lemma 1)."""
    W = b * k * k * c
    X = a_in * c / sigma
    t_c = a * b * k * k * c / (R * eta_k)
    t_in = min(math.ceil(X / Mb) * W + X, W + math.ceil(W / Mb) * X) / B
    t_out = a * b / Bo
    T = max(t_c, t_in, t_out)
    return {"T": T, "eta": a * b * k * k * c / (R * T), "compute": t_c, "intake": t_in, "output": t_out}


def prop1_walk_term(T: float = 12, q: int = 14, osc9: float = 2.0, bs: int = 16, s: int = 12, r: int = 15,
                    i_minus_p: float = 2.0) -> Dict[str, float]:
    """Executed walk error budget at one level (Proposition 1(ii)-(iii))."""
    tap = T * 2.0 ** (2 - q) * osc9
    state = T * 2.0 ** (-bs)
    coef = s * 2.0 ** (-r - 1) * i_minus_p
    return {"tap": tap, "state": state, "coef": coef, "total": tap + state + coef}


def prop2_marginal_bound(b: int = 24, N: int = 128, kmax: int = 3, ue: float = 2.0 ** -20) -> float:
    """|p_hat - p| / p <= 2a / (1 - a)  (Proposition 2(i)); returns log2 of the bound."""
    u = 2.0 ** -b
    n = 2 * N + 2 * kmax
    gam = n * u / (1 - n * u)
    theta = gam + kmax * ue * (1 + gam)
    a = (1 + theta) * (1 + u) ** 2 - 1
    return math.log2(2 * a / (1 - a))


def p1_rate_bound(n_images: int = 9370, alpha: float = 0.05, family: int = 12) -> float:
    """One-sided upper bound on the per-image mismatch rate after zero mismatches."""
    return 1.0 - (alpha / family) ** (1.0 / n_images)


def prop6_latency(A1: float, A2a: float, A2b: float, A3a: float, A3b: float, g1: float, g2: float, g3: float) -> float:
    """L1 = A1 + max(A2a, g1) + A2b + max(A3a, g2) + A3b + g3."""
    return A1 + max(A2a, g1) + A2b + max(A3a, g2) + A3b + g3


def lane_energy_rule(t_pl_s: Dict[int, float], P_f: float = 0.35, p_l: float = 0.0066) -> int:
    """n* = argmin_n (P_f + n p_l) t_PL(n) over the enabled-lane options."""
    return min(t_pl_s, key=lambda n: (P_f + n * p_l) * t_pl_s[n])
