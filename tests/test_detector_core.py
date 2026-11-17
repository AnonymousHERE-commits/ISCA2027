"""The detector's float neck operations against the integer reference (model M).

The float operations (detector/splice_det/ops.py, exchange_float.py) are what the
PyTorch model trains with; the integer reference is what the hardware executes.  On the
synthetic scenes both are evaluated on the same inputs: the float results must agree with
the integer codes to within the rounding the manifest introduces.
"""
import sys
import unittest

import numpy as np

import _path  # noqa: F401
from _path import REPO

sys.path.insert(0, str(REPO / "detector"))

import splicev_gen as gen  # noqa: E402
from splice_det import exchange_float, ops  # noqa: E402
from splice_det.backend import NumpyBackend  # noqa: E402
from splice_det.targets import level_targets, ownership, window_targets  # noqa: E402
from splicev_ref import forecast as fc_ref, gate as gate_ref, toy  # noqa: E402
from splicev_ref.membership import UNASSIGNED, restrict as restrict_ref  # noqa: E402
from splicev_ref.walk import export_taps, walk_bands  # noqa: E402

B = NumpyBackend()


def ids_float(ids):
    ids = np.asarray(ids, dtype=np.int64)
    return np.where(ids == UNASSIGNED, -1, ids)


class DetectorCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = toy.make_scene(3)
        cls.tr = toy.run(cls.scene)

    def test_bands_match_the_integer_walk(self):
        for name in ("P3", "P4", "P5"):
            L = self.scene["levels"][name]
            steps = gen.LEVELS[name]["steps"]
            exps = gen.BAND_EXP[name]
            ref = walk_bands(L["x"], export_taps(L["w"], L["inside"]), L["betas"], steps, exps, gen.BAND_EXP_LOW, 0)
            x = np.transpose(L["x"], (2, 0, 1))[None].astype(np.float64)
            w = np.transpose(np.where(L["inside"], L["w"], 0), (2, 0, 1))[None] / 2.0 ** 15
            betas = np.transpose(L["betas"], (2, 0, 1))[None] / 2.0 ** 16
            Z, low = ops.bands(B, ops.walk_powers(B, x, w, steps), betas)
            for j in range(4):
                code = np.clip(ops.band_codes(B, Z[j], 1.0, exps[j]), -128, 127)[0].transpose(1, 2, 0)
                diff = np.abs(code - ref[j])
                self.assertLessEqual(diff.max(), 3.0, (name, j))
                self.assertLess(diff.mean(), 0.6, (name, j))

    def test_restriction_and_fill(self):
        tr = self.tr
        geo = self.scene["geo"]
        n4 = geo.h4 * geo.w4
        x_anchor = np.concatenate([self.scene["levels"]["P4"]["x"].reshape(n4, -1),
                                   self.scene["levels"]["P3"]["x"].reshape(-1, geo.channels)[tr["reserve"]]])
        ids = tr["ids_anchor"]
        omega = np.where(ids != UNASSIGNED, tr["e_anchor"], 0).astype(np.float64)
        H, mass = ops.restrict(B, x_anchor.astype(np.float64), ids_float(ids), omega, tr["F"])
        self.assertLessEqual(np.abs(H - tr["H"]).max(), 0.5 + 1e-9)        # integer means round half even
        filled = ops.fill(B, H, ids_float(ids))
        self.assertTrue(np.all(filled[ids == UNASSIGNED] == 0))

    def test_forecast_matches_the_integer_forecast(self):
        tr = self.tr
        F = tr["F"]
        adm = tr["adm"]
        # integer logits: (kk * KK_MULT + uu * UU_MULT) >> SHIFT in 1/16 units; float equivalent:
        kk_scale = gen.FC_KK_MULT / 2.0 ** gen.FC_SHIFT / 16.0
        uu_scale = gen.FC_UU_MULT / 2.0 ** gen.FC_SHIFT / 16.0
        K = tr["K"] * np.sqrt(kk_scale * np.sqrt(64.0))
        Uv = tr["Uv"] * np.sqrt(uu_scale)
        l0 = tr["null_raw"] * gen.FC_NULL_MULT / 2.0 ** gen.FC_NULL_SHIFT / 16.0
        al = tr["alpha_raw"] * gen.FC_ALPHA_MULT / 2.0 ** gen.FC_ALPHA_SHIFT / 256.0
        dh, a, a0, alpha = ops.forecast(B, K, Uv, tr["VO"].astype(float), tr["H"][:, :48].astype(float),
                                        l0, np.clip(al, -8, 8), adm, clip_b=64.0)
        ref, _ = fc_ref.forecast(tr["K"], tr["Uv"], tr["VO"], tr["H"][:, :48], tr["null_raw"], tr["alpha_raw"], adm)
        rows = [f for f in range(F) if adm[f].any()]
        self.assertTrue(rows)
        err = np.abs(dh[rows] - ref[rows])
        self.assertLessEqual(err.max(), 3.0)
        empty = [f for f in range(F) if not adm[f].any()]
        self.assertTrue(np.all(dh[empty] == 0))

    def test_gate_matches_the_integer_gate(self):
        tr = self.tr
        F = tr["F"]
        ids = ids_float(tr["ids_anchor"])
        prec = tr["hyp_prec"].astype(np.float64) / 2.0 ** gen.GATE_PREC_FRAC
        mu_f, var_f = ops.fuse_hypotheses(B, tr["hyp_mu"].astype(np.float64), prec, ids, F)
        tau = gen.GATE_TAU_BF[gate_ref.stratum(F)]
        tau = tau[0] * 2.0 ** tau[1] / 2.0 ** gen.GATE_PREC_FRAC
        adm = ops.admissible(B, ops.mahalanobis2(B, mu_f, var_f), tau)
        agree = (adm == tr["adm"]).mean()
        self.assertGreater(agree, 0.97)

    def test_float_exchange_matches_the_integer_exchange(self):
        tr = self.tr
        F = tr["F"]
        ev = np.zeros((F, F))
        for f in range(F):
            for g, s in zip(tr["partners"][f], tr["evidence"][f]):
                ev[f, g] = s / 256.0
        prior = [gen.PRIOR_BF[k][0] * 2.0 ** gen.PRIOR_BF[k][1] for k in range(gen.KMAX)]
        out = exchange_float.exchange(tr["adm"], ev, tr["need"] / 2.0 ** 23, tr["rho"] / 2.0 ** 23, tr["phi"] / 2.0 ** 23,
                                      tr["VO"].astype(float), tr["H"][:, :48].astype(float), prior,
                                      a_r=gen.A_R_Q39 / 2.0 ** 39, b_r=gen.B_R_Q39 / 2.0 ** 39, c_r=gen.C_R_Q39 / 2.0 ** 39)
        for f in range(F):
            p_ref = [m[0] * 2.0 ** m[1] for m in tr["marginals"][f]]
            self.assertTrue(np.allclose(out["marginals"][f], p_ref, rtol=1e-4, atol=1e-6), f)
        err = np.abs(out["dV"] - tr["dV"])
        self.assertLessEqual(err.max(), 2.0)
        sel_ref = {(f, int(g)) for f in range(F) for g in tr["sources"][f] if g != 0xFFFF}
        sel = {(f, g) for f in range(F) for g in np.flatnonzero(out["lam"][f] > 1e-9)}
        self.assertGreaterEqual(len(sel & sel_ref) / max(1, len(sel | sel_ref)), 0.9)

    def test_targets(self):
        boxes = np.array([[16, 16, 80, 48], [40, 8, 56, 120], [200, 200, 210, 300]], dtype=float)
        owner, count = ownership(boxes, 8, 40, 40)
        self.assertEqual(owner[2, 2], 0)                     # centre (20, 20) only in box 0
        self.assertEqual(owner[3, 6], 1)                     # (52, 28): in both, box 1 is smaller
        self.assertEqual(count[3, 6], 2)
        t = level_targets(boxes, 8, 40, 40)
        self.assertEqual(t["q"][3, 6], 0.5)
        self.assertEqual(t["t"][30, 25], 1.0)                # 10-px wide box is thin at stride 8
        self.assertTrue(np.all((t["pi"] == 0) | (t["e"][None] == 1)))
        wt = window_targets(np.array([[0, 0, -1], [1, 1, -1], [-1, -1, -1]]))
        self.assertEqual(wt[0, 0], 1.0)                      # (0,0)->(0,1): LINK_FWD[0] = (0, 1), same owner
        self.assertEqual(wt[0, 2], -1.0)                     # (1,-2) leaves the lattice
        self.assertEqual(wt[0, 4], 0.0)                      # (1,0): owner 1 vs owner 0


if __name__ == "__main__":
    unittest.main()
