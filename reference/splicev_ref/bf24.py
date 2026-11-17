"""Nonnegative block-floating arithmetic with 24-bit mantissas (posterior, gate).

A value is a pair (m, e) meaning m * 2^e with m = 0 (and e = 0) or
2^23 <= m < 2^24.  Every operation returns the exact result rounded half to
even to 24 bits, except the reciprocal, which is the manifest's seed-table +
two-Newton-step unit (Fig. 3(d)) followed by one RNE normalization.  The C++
twin is pl/include/sv_bf24.h; both must agree bit for bit.
"""
from __future__ import annotations

from typing import Tuple

from . import gen
from .arith import rshift_rne

BF = Tuple[int, int]
MB = 24
M_LO = 1 << (MB - 1)
M_HI = 1 << MB
ZERO: BF = (0, 0)


def norm(x: int, e: int) -> BF:
    """Exact nonnegative integer x * 2^e -> nearest BF (ties to even)."""
    if x < 0:
        raise ValueError("bf24 values are nonnegative")
    if x == 0:
        return ZERO
    n = x.bit_length()
    if n > MB:
        s = n - MB
        m = rshift_rne(x, s)
        e += s
        if m == M_HI:
            m >>= 1
            e += 1
        return (m, e)
    if n < MB:
        return (x << (MB - n), e - (MB - n))
    return (x, e)


def from_int(x: int) -> BF:
    return norm(x, 0)


def from_fixed(x: int, frac: int) -> BF:
    return norm(x, -frac)


def is_zero(a: BF) -> bool:
    return a[0] == 0


def mul(a: BF, b: BF) -> BF:
    if a[0] == 0 or b[0] == 0:
        return ZERO
    return norm(a[0] * b[0], a[1] + b[1])


def add(a: BF, b: BF) -> BF:
    if a[0] == 0:
        return b
    if b[0] == 0:
        return a
    if a[1] < b[1]:
        a, b = b, a
    d = a[1] - b[1]
    if d >= MB + 2:
        return a  # b < 1/4 ulp(a): the exact RNE result is a
    return norm((a[0] << d) + b[0], b[1])


def recip(a: BF) -> BF:
    """1/a by the manifest reciprocal unit; a must be nonzero."""
    m, e = a
    if m == 0:
        raise ZeroDivisionError("bf24 reciprocal of zero")
    idx = (m >> 12) - 2048
    y = gen.RECIP_SEED[idx]
    for _ in range(2):
        t = m * y                    # Q2.54
        err = (1 << 55) - t          # 2 - x*y, Q2.54
        y = (y * (err >> 24)) >> 30  # Q2.30
    return norm(y, -54 - e)


def div(a: BF, b: BF) -> BF:
    return mul(a, recip(b))


def cmp(a: BF, b: BF) -> int:
    if a[0] == 0 or b[0] == 0:
        return (a[0] > 0) - (b[0] > 0)
    if a[1] != b[1]:
        return 1 if a[1] > b[1] else -1
    return (a[0] > b[0]) - (a[0] < b[0])


def to_fixed(a: BF, frac: int) -> int:
    """a * 2^frac rounded half-even to an integer."""
    m, e = a
    s = e + frac
    return m << s if s >= 0 else rshift_rne(m, -s)


def to_float(a: BF) -> float:
    return float(a[0]) * 2.0 ** a[1]


def exp_q78(s_code: int) -> BF:
    """e^(s_code / 256) with relative error below 2^-20 (two-level 2^f table)."""
    t = s_code * gen.LOG2E_Q30              # units 2^-38
    k = t >> 38
    f = t - (k << 38)                       # [0, 2^38)
    hi = f >> 28                            # top 10 bits of the fraction
    lo = f & ((1 << 28) - 1)
    c = (lo * gen.LN2_Q30) >> 44            # Q0.24 of lo * 2^-38 * ln 2
    m = rshift_rne(gen.EXP2_HI[hi] * ((1 << 24) + c), 24)
    return norm(m, k - 23)
