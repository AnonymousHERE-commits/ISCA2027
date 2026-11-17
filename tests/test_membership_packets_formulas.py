import unittest

import numpy as np

import _path  # noqa: F401
import splicev_gen as gen
from splicev_ref import formulas, membership as mb, packets


class MembershipTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.n, self.F, self.C = 300, 9, 16
        self.ids = rng.integers(0, self.F, size=self.n)
        self.ids[rng.random(self.n) < 0.1] = mb.UNASSIGNED
        self.w = rng.integers(1, 256, size=self.n)
        self.x = rng.integers(-128, 128, size=(self.n, self.C))

    def test_update_moves_coarse_coordinates_exactly(self):   # Remark 1
        H = mb.restrict(self.x, self.ids, self.w, self.F)
        dH = np.random.default_rng(5).integers(-5, 6, size=(self.F, self.C))
        x2 = self.x + mb.fill(dH, self.ids)
        D = mb.detail(self.x, H, self.ids)
        D2 = mb.detail(x2, H + dH, self.ids)
        self.assertTrue(np.array_equal(D, D2))
        H2 = mb.restrict(x2, self.ids, self.w, self.F, cnt=None)
        self.assertTrue(np.array_equal(np.clip(H + dH, -128, 127), H2))

    def test_projected_correction_has_half_lsb_mean(self):
        rw = np.random.default_rng(6).integers(-50, 50, size=(self.n, self.C))
        rp = mb.project_out(rw, self.ids, self.w, self.F)
        s, ws = mb.fragment_sums(rp, self.ids, self.w, self.F)
        ok = ws > 0
        self.assertTrue(np.all(np.abs(s[ok] / ws[ok, None]) <= 0.5 + 1e-12))

    def test_fill_is_fragment_constant_and_zero_outside(self):
        H = mb.restrict(self.x, self.ids, self.w, self.F)
        f = mb.fill(H, self.ids)
        self.assertTrue(np.all(f[self.ids == mb.UNASSIGNED] == 0))
        for k in range(self.F):
            rows = f[self.ids == k]
            self.assertTrue(np.all(rows == rows[0]))

    def test_level_extension(self):
        ids4 = np.arange(16).reshape(4, 4) % 3
        ids3, i4, ids5 = mb.extend_levels(ids4.ravel(), np.array([0]), np.array([2]), (8, 8), (4, 4), (2, 2))
        self.assertEqual(ids3[0], 2)
        self.assertEqual(ids3[1], ids4[0, 0])
        self.assertEqual(ids5.size, 4)


class PacketTests(unittest.TestCase):
    def test_roundtrip_and_stale_detection(self):
        rng = np.random.default_rng(7)
        f = {"e": rng.integers(0, 256, 6400), "t": rng.integers(0, 256, 6400)}
        blob = packets.pack("H0", 41, 3, f)
        meta, out = packets.unpack(blob, "H0", 41, 3)
        self.assertTrue(np.array_equal(out["e"], f["e"]))
        with self.assertRaises(packets.StalePacket):
            packets.unpack(blob, "H0", 42, 3)
        bad = bytearray(blob)
        bad[40] ^= 1
        with self.assertRaises(ValueError):
            packets.unpack(bytes(bad), "H0", 41, 3)

    def test_mask_packet_is_8192_bytes(self):
        self.assertEqual(gen.PACKET_LAYOUTS["G"]["payload_bytes"], 8192)
        self.assertEqual(gen.PACKET_LAYOUTS["H3p"]["payload_bytes"], 4096)
        self.assertEqual(gen.PACKET_LAYOUTS["H3pp"]["payload_bytes"], 106496)


class PaperNumberTests(unittest.TestCase):
    def test_gate_and_exchange_cycles(self):
        self.assertEqual(formulas.gate_cycles(), 2798)
        self.assertEqual(formulas.exchange_cycles(e_dup=0), 33472)
        self.assertEqual(formulas.exchange_cycles(e_dup=1024), 41664)
        self.assertEqual(formulas.exchange_cycles(e_dup=4096), 66240)
        st = formulas.exchange_stage_ms(41664)
        self.assertAlmostEqual(st["X_ms"], 0.217, places=3)
        self.assertAlmostEqual(st["g2_ms"], 0.242, places=3)
        self.assertAlmostEqual(st["t_U_us"], 2.9, places=1)

    def test_lemma1(self):
        self.assertEqual(formulas.lemma1_row_slice_threshold(), 544)
        self.assertAlmostEqual(formulas.lemma1_row_slice_ceiling(), 0.37, places=2)
        self.assertAlmostEqual(formulas.lemma1_output_ceiling(), 0.28, places=2)

    def test_prop1_and_prop2(self):
        self.assertAlmostEqual(formulas.prop1_walk_term()["total"], 0.0064, places=4)
        self.assertAlmostEqual(formulas.prop1_walk_term(q=13)["total"], 0.012, places=3)
        self.assertAlmostEqual(formulas.prop2_marginal_bound(24), -14.7, places=1)
        self.assertAlmostEqual(formulas.prop2_marginal_bound(16), -6.9, places=1)

    def test_p1_bound_and_boundary(self):
        import math
        # 1 - (alpha/K)^(1/n) = 5.85e-4; the manuscript states the bound rounded up, 5.9e-4
        self.assertEqual(math.ceil(formulas.p1_rate_bound() * 1e5) / 10, 5.9)
        total = sum(t["figure_kb"] for t in [
            {"figure_kb": v} for v in (12.8, 139.7, 19.2, 25.0, 8.2, 100.0, 4.1, 106.5, 20.0, 100.0)])
        self.assertAlmostEqual(total, 535.5, places=1)

    def test_prop6(self):
        L = formulas.prop6_latency(1.0, 0.405, 0.1, 0.745, 0.2, 0.185, 0.242, 0.05)
        self.assertAlmostEqual(L, 1.0 + 0.405 + 0.1 + 0.745 + 0.2 + 0.05)


if __name__ == "__main__":
    unittest.main()
