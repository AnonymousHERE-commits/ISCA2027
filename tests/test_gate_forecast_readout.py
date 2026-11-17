import unittest

import numpy as np

import _path  # noqa: F401
import splicev_gen as gen
from splicev_ref import forecast as fc
from splicev_ref import gate, readout


class GateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 40
        self.ids = rng.integers(0, 6, size=self.n)
        self.mu = rng.integers(0, 300, size=(self.n, 4))
        self.prec = rng.integers(1, 30, size=(self.n, 4))

    def test_mask_symmetric_zero_diagonal_roundtrip(self):
        mu_f, var_f = gate.fuse(self.mu, self.prec, self.ids, 6)
        adm = gate.admissible_mask(mu_f, var_f, 6)
        self.assertTrue(np.array_equal(adm, adm.T))
        self.assertFalse(np.any(np.diag(adm)))
        words = gate.pack_mask(adm)
        self.assertEqual(words.size * 4, 8192)
        self.assertTrue(np.array_equal(gate.unpack_mask(words, 6), adm))

    def test_identical_hypotheses_are_admissible(self):
        mu = np.tile(np.array([[50, 60, 70, 80]]), (4, 1))
        prec = np.full((4, 4), 20)
        mu_f, var_f = gate.fuse(mu, prec, np.array([0, 0, 1, 1]), 2)
        self.assertTrue(gate.admissible_mask(mu_f, var_f, 2)[0, 1])

    def test_gate_cycles_at_the_cap(self):
        self.assertEqual(gate.gate_cycles(256, 128), 2798)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.F = 7
        self.K = rng.integers(-60, 60, size=(self.F, 64))
        self.Uv = rng.integers(-60, 60, size=(self.F, gen.FC_UDIM))
        self.VO = rng.integers(-128, 128, size=(self.F, 48))
        self.Hs = rng.integers(-128, 128, size=(self.F, 48))
        self.null = rng.integers(-3000, 3000, size=self.F)
        self.alpha = rng.integers(-3000, 3000, size=self.F)
        adm = rng.random((self.F, self.F)) < 0.5
        adm = np.triu(adm, 1)
        self.adm = adm | adm.T

    def test_masked_donors_contribute_nothing(self):
        out, _ = fc.forecast(self.K, self.Uv, self.VO, self.Hs, self.null, self.alpha, self.adm)
        VO2 = self.VO.copy()
        for f in range(self.F):
            for g in range(self.F):
                if not self.adm[0, g]:
                    VO2[g] = -VO2[g]
        out2 = fc.forecast_row(0, self.K, self.Uv, VO2, self.Hs, int(self.null[0]), int(self.alpha[0]), self.adm[0])[0]
        self.assertEqual(list(out[0]), list(out2))

    def test_empty_admissible_set_gives_zero(self):
        empty = np.zeros_like(self.adm)
        out, ab = fc.forecast(self.K, self.Uv, self.VO, self.Hs, self.null, self.alpha, empty)
        self.assertTrue(np.all(out == 0))
        self.assertTrue(np.all(ab))

    def test_bounded_by_alpha_times_clip(self):
        out, _ = fc.forecast(self.K, self.Uv, self.VO, self.Hs, self.null, self.alpha, self.adm)
        self.assertLessEqual(int(np.abs(out).max()), gen.FC_B)

    def test_null_option_shrinks_and_is_not_renormalised_away(self):
        f = int(np.flatnonzero(self.adm.sum(axis=1) > 0)[0])
        lo, _ = fc.forecast_row(f, self.K, self.Uv, self.VO, self.Hs, -100000, int(self.alpha[f]), self.adm[f])
        hi, ab = fc.forecast_row(f, self.K, self.Uv, self.VO, self.Hs, 100000, int(self.alpha[f]), self.adm[f])
        self.assertGreaterEqual(sum(abs(v) for v in lo), sum(abs(v) for v in hi))
        self.assertTrue(ab)


class ReadoutTests(unittest.TestCase):
    def test_score_never_exceeds_gate_and_skip_is_exact(self):   # Prop. 3
        rng = np.random.default_rng(2)
        h = w = 32
        g = rng.integers(0, 1 << 15, size=(h, w))
        g[:8, :8] = rng.integers(0, gen.TAU_DET_Q15, size=(8, 8))
        logits = rng.integers(-32768, 32768, size=h * w)
        p = readout.class_scores(logits, g.ravel())
        self.assertTrue(np.all(p <= g.ravel()))
        skip = readout.skippable_blocks(g)
        self.assertTrue(skip[0, 0])
        cand = set(readout.candidates(p).tolist())
        for by, bx in zip(*np.nonzero(skip)):
            for y in range(by * 8, by * 8 + 8):
                for x in range(bx * 8, bx * 8 + 8):
                    self.assertNotIn(y * w + x, cand)

    def test_residual_fits_nine_bits(self):
        dh = np.full((3, 48), 127)
        dh_hat = np.full((3, 48), -64)
        r = readout.residual(dh, dh_hat)
        self.assertEqual(int(r.max()), gen.RES_BOUND)
        z = np.ones((48, 8), dtype=np.int64)
        self.assertEqual(readout.rz_product(r, z).shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
