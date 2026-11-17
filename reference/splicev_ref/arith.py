"""Integer arithmetic of the manifest (Definition 1): rounding, saturation, counters.

Scalar functions take Python ints (arbitrary precision, exact).  The ``*_np``
variants take numpy int64 arrays and are exact as long as inputs stay within
int64, which every caller asserts through the manifest's bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


@dataclass
class Counters:
    """Counted events of one frame or unit (saturations, budget exits, caps...)."""
    c: Dict[str, int] = field(default_factory=dict)

    def add(self, name: str, n: int = 1) -> None:
        if n:
            self.c[name] = self.c.get(name, 0) + int(n)

    def get(self, name: str) -> int:
        return self.c.get(name, 0)

    def merge(self, other: "Counters") -> None:
        for k, v in other.c.items():
            self.add(k, v)


def rshift_rne(x: int, s: int) -> int:
    """x * 2^-s rounded half to even; s <= 0 is an exact left shift."""
    if s <= 0:
        return x << (-s)
    q = x >> s
    r = x - (q << s)
    half = 1 << (s - 1)
    if r > half or (r == half and (q & 1)):
        q += 1
    return q


def rshift_floor(x: int, s: int) -> int:
    return x >> s if s >= 0 else x << (-s)


def div_rne(n: int, d: int) -> int:
    """n / d rounded half to even (d > 0)."""
    if d <= 0:
        raise ZeroDivisionError("div_rne requires a positive divisor")
    q, r = divmod(n, d)
    two_r = 2 * r
    if two_r > d or (two_r == d and (q & 1)):
        q += 1
    return q


def div_floor(n: int, d: int) -> int:
    if d <= 0:
        raise ZeroDivisionError("div_floor requires a positive divisor")
    return n // d


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def sat(x: int, lo: int, hi: int, cnt: Counters | None = None, name: str = "sat") -> int:
    if x < lo:
        if cnt is not None:
            cnt.add(name)
        return lo
    if x > hi:
        if cnt is not None:
            cnt.add(name)
        return hi
    return x


def sat8(x: int, cnt: Counters | None = None, name: str = "sat8") -> int:
    return sat(x, INT8_MIN, INT8_MAX, cnt, name)


def sat8_sym(x: int, cnt: Counters | None = None, name: str = "sat8_sym") -> int:
    return sat(x, -127, 127, cnt, name)


def sat16(x: int, cnt: Counters | None = None, name: str = "sat16") -> int:
    return sat(x, INT16_MIN, INT16_MAX, cnt, name)


def sat32(x: int, cnt: Counters | None = None, name: str = "sat32") -> int:
    return sat(x, INT32_MIN, INT32_MAX, cnt, name)


def rotl32(x: int, r: int) -> int:
    x &= 0xFFFFFFFF
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def check_word(words, seed: int = 0x9E3779B9) -> int:
    c = seed
    for w in words:
        c = rotl32(c, 5) ^ (int(w) & 0xFFFFFFFF)
    return c


# ---------------------------------------------------------------- numpy forms

def rshift_rne_np(x: np.ndarray, s: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    if s <= 0:
        return x << (-s)
    q = x >> s
    r = x - (q << s)
    half = np.int64(1) << (s - 1)
    up = (r > half) | ((r == half) & ((q & 1) == 1))
    return q + up.astype(np.int64)


def rshift_rne_np_var(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-element shift amounts (s >= 1)."""
    x = np.asarray(x, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    if np.any(s < 1):
        raise ValueError("per-element shifts must be >= 1")
    q = x >> s
    r = x - (q << s)
    half = np.int64(1) << (s - 1)
    up = (r > half) | ((r == half) & ((q & 1) == 1))
    return q + up.astype(np.int64)


def div_rne_np(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    if np.any(d <= 0):
        raise ZeroDivisionError("div_rne_np requires positive divisors")
    q = np.floor_divide(n, d)
    r = n - q * d
    two_r = 2 * r
    up = (two_r > d) | ((two_r == d) & ((q & 1) == 1))
    return q + up.astype(np.int64)


def sat_np(x: np.ndarray, lo: int, hi: int, cnt: Counters | None = None, name: str = "sat") -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    if cnt is not None:
        cnt.add(name, int(np.count_nonzero((x < lo) | (x > hi))))
    return np.clip(x, lo, hi)


def as_i64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)
