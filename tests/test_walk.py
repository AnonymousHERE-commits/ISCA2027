import unittest
from fractions import Fraction

import numpy as np

import _path  # noqa: F401
from splicev_ref import walk
from splicev_ref.exact import oracles


def random_lattice(rng, h, w):
    inside = walk.inside_mask(h, w)
    wc = rng.integers(0, 1 << 15, size=(h, w, 8))
    return np.where(inside, wc, 0), inside


class WalkTests(unittest.TestCase):
    def test_taps_are_row_stochastic_and_nonnegative(self):
        rng = np.random.default_rng(1)
        wc, inside = random_lattice(rng, 9, 7)
        taps = walk.export_taps(wc, inside)
        self.assertTrue(np.all(taps >= 0))
        self.assertTrue(np.all(taps.sum(axis=2) == 1 << 14))
        self.assertTrue(np.all(taps[..., 4] >= (1 << 13) - 8))

    def test_constants_are_reproduced_exactly(self):   # Prop. 1(i)
        rng = np.random.default_rng(2)
        wc, inside = random_lattice(rng, 10, 10)
        taps = walk.export_taps(wc, inside)
        for c in (-32768, -12345, 0, 777, 32767):
            x = np.full((10, 10, 1), c, dtype=np.int64)
            for _ in range(12):
                x = walk.walk_step(x, taps)
            self.assertTrue(np.all(x == c), c)

    def test_bank_coefficients(self):
        rng = np.random.default_rng(3)
        beta = rng.integers(1, (1 << 16) - 1, size=200)
        for s in (4, 8, 12):
            a = walk.bank_coefficients(beta, s)
            self.assertTrue(np.all(a >= 0))
            self.assertTrue(np.all(a.sum(axis=-1) == 1 << 15))

    def test_telescoping_holds_in_the_accumulator(self):
        rng = np.random.default_rng(4)
        h, w, c = 8, 9, 3
        wc, inside = random_lattice(rng, h, w)
        taps = walk.export_taps(wc, inside)
        betas = np.sort(rng.integers(1000, 64000, size=(h, w, 4)), axis=-1) + np.arange(4)
        x = rng.integers(-128, 128, size=(h, w, c))
        for s in (4, 8, 12):
            acc = walk.band_accumulators(x, taps, betas, s)
            self.assertTrue(np.array_equal(acc.sum(axis=0), (x << 8) << 15), s)

    def test_executed_walk_within_prop1_budget(self):
        """A(P_hat) X in the manifest against A(P) X exactly on a small graph."""
        rng = np.random.default_rng(5)
        h, w = 5, 5
        wc, inside = random_lattice(rng, h, w)
        taps = walk.export_taps(wc, inside)
        n = h * w
        W = [[Fraction(0)] * n for _ in range(n)]
        for y in range(h):
            for x in range(w):
                for d, (dy, dx) in enumerate(walk.NEIGH):
                    if inside[y, x, d]:
                        W[y * w + x][(y + dy) * w + x + dx] = Fraction(int(wc[y, x, d]), 1 << 15)
        P = oracles.exact_walk_matrix(W)
        xs = rng.integers(-128, 128, size=(h, w, 1))
        s = 8
        beta = 45000
        X = [Fraction(int(v) << 8, 1 << 15) for v in xs.ravel()]
        exact = oracles.bank_apply(P, X, Fraction(beta, 1 << 16), s)
        pw = walk.walk_powers(xs, taps, s)
        a = walk.bank_coefficients(np.full((h, w), beta), s)
        got = np.einsum("hwt,thwc->hwc", a, pw)[..., 0].ravel()
        err = max(abs(Fraction(int(g), 1 << 30) - e) for g, e in zip(got, exact))
        T = sum(t * (1 - Fraction(beta, 1 << 16)) * Fraction(beta, 1 << 16) ** t for t in range(s)) + s * Fraction(beta, 1 << 16) ** s
        bound = float(T) * (2.0 ** (2 - 14) * 2 + 2.0 ** -16) + s * 2.0 ** -16 * 2
        self.assertLess(float(err), bound)

    def test_band_certificate_holds_for_extreme_valid_chains(self):
        """sum_t |d_jt| < 2^16 keeps every band accumulator inside int32 (|X| <= 2^15)."""
        chains = np.array([[1, 2, 3, 4], [65529, 65530, 65531, 65532], [1, 2, 65531, 65532], [100, 30000, 30001, 65000]])
        for s in (4, 8, 12):
            d, low = walk.band_coefficients(chains, s)
            self.assertTrue(np.all(np.abs(d).sum(axis=-1) < (1 << 16)))
            self.assertTrue(np.all(low.sum(axis=-1) == 1 << 15))
        with self.assertRaises(ValueError):
            walk.band_coefficients(np.array([[5, 4, 6, 7]]), 4)   # not increasing

if __name__ == "__main__":
    unittest.main()
