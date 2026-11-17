"""Lookup tables of the manifest, generated from formulas with exact decimal arithmetic.

No table is typed by hand: each one is a pure function of the manifest, computed
at 60 significant digits and rounded half-even once, so every consumer
(AIE C++, HLS C++, PS Python, reference Python) receives identical integers.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN, ROUND_FLOOR, getcontext
from typing import Dict, List, Tuple

getcontext().prec = 60

_LN2 = Decimal(2).ln()


def _rne(x: Decimal) -> int:
    return int(x.to_integral_value(rounding=ROUND_HALF_EVEN))


def _floor(x: Decimal) -> int:
    return int(x.to_integral_value(rounding=ROUND_FLOOR))


def forecast_exp_table(entries: int, out_frac: int, offset: int, logit_frac: int, logit_min: int) -> List[int]:
    """w[i] = rne(2^out_frac * exp((i + logit_min) / 2^logit_frac + offset))."""
    scale = Decimal(2) ** out_frac
    out = []
    for i in range(entries):
        x = Decimal(i + logit_min) / (Decimal(2) ** logit_frac) + Decimal(offset)
        v = _rne(scale * x.exp())
        if not (0 < v < 2 ** 32):
            raise ValueError("forecast exp table entry out of uint32 range")
        out.append(v)
    return out


def sigmoid_table(entries: int, in_frac: int, clamp_lo: int, index_shift: int) -> List[int]:
    """s[i] = min(32767, rne(2^15 * sigmoid(x_mid))), x_mid the bucket centre."""
    out = []
    half_bucket = 1 << (index_shift - 1)
    for i in range(entries):
        code_mid = (i << index_shift) + clamp_lo + half_bucket
        x = Decimal(code_mid) / (Decimal(2) ** in_frac)
        s = Decimal(1) / (Decimal(1) + (-x).exp())
        out.append(min(32767, _rne(Decimal(32768) * s)))
    return out


def recip_seed_table(entries: int = 2048) -> List[int]:
    """T[i] = floor(2^43 / (2 i + 4097)): Q2.30 seed of 1/x at the bucket centre."""
    return [(1 << 43) // (2 * i + 4097) for i in range(entries)]


def exp2_hi_table(entries: int = 1024) -> List[int]:
    """E[i] = rne(2^23 * 2^(i/1024)): 24-bit mantissas in [2^23, 2^24)."""
    out = []
    for i in range(entries):
        v = _rne((Decimal(2) ** 23) * ((Decimal(i) / Decimal(entries)) * _LN2).exp())
        if not ((1 << 23) <= v < (1 << 24)):
            raise ValueError("exp2 table entry outside [2^23, 2^24)")
        out.append(v)
    return out


def log2e_q30() -> int:
    return _rne((Decimal(2) ** 30) / _LN2)


def ln2_q30() -> int:
    return _rne((Decimal(2) ** 30) * _LN2)


def to_q(x: float | int | str, frac: int) -> int:
    """Exact decimal value -> fixed point code, rounded half-even."""
    return _rne(Decimal(str(x)) * (Decimal(2) ** frac))


def bf24_from_decimal(x: Decimal) -> Tuple[int, int]:
    """Nonnegative decimal -> (m, e) with m in [2^23, 2^24) or (0, 0), value m*2^e, RNE."""
    if x < 0:
        raise ValueError("bf24 values are nonnegative")
    if x == 0:
        return (0, 0)
    e = 0
    y = x
    while y >= (1 << 24):
        y /= 2
        e += 1
    while y < (1 << 23):
        y *= 2
        e -= 1
    m = _rne(y)
    if m == (1 << 24):
        m >>= 1
        e += 1
    return (m, e)


def bf24_from_value(x: float | int | str) -> Tuple[int, int]:
    return bf24_from_decimal(Decimal(str(x)))


def all_tables(m: Dict) -> Dict[str, List[int]]:
    fc = m["forecast"]
    lf = fc["logit_format"]
    et = fc["exp_table"]
    st = m["readout"]["sigmoid_table"]
    return {
        "FC_EXP": forecast_exp_table(et["entries"], et["out_frac"], et["offset"], lf["frac"], lf["clamp"][0]),
        "SIG_TABLE": sigmoid_table(st["entries"], st["in_frac"], st["clamp"][0], st["index_shift"]),
        "RECIP_SEED": recip_seed_table(2048),
        "EXP2_HI": exp2_hi_table(1024),
    }
