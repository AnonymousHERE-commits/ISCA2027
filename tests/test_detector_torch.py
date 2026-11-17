"""The PyTorch SPLICE neck (detector/splice_det/neck.py) end to end: forward pass with the
CPU reserve / linkage / exchange steps, auxiliary losses, QAT sites and the YOLO26 glue.

Runs on real torch when it is installed, otherwise on tests/torch_shim.py (forward only), so
the neck's shapes, indexing and invariants are checked on every machine.  Training itself
(autograd) is exercised by `python -m splice_det.train --smoke` on the HPC.
"""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "detector"))

import torch_shim  # noqa: E402

torch = torch_shim.load()

from splice_det import exchange_float, ops  # noqa: E402
from splice_det.backend import NumpyBackend  # noqa: E402
from splice_det.config import ARMS, DESCRIPTORS, SpliceConfig  # noqa: E402
from splice_det.losses import aux_losses, total_loss  # noqa: E402
from splice_det.neck import SpliceNeck  # noqa: E402
from splice_det.qat import NeckQuant  # noqa: E402
from splice_det.yolo26 import SpliceYOLO, _add_offset, _combine, _postprocess  # noqa: E402

NB = NumpyBackend()
IN_CH, OUT_CH, STRIDES = (24, 40, 56), (32, 48, 64), (8, 16, 32)


def make_neck(seed=0, arm="e", descriptor="bands"):
    torch.manual_seed(seed)
    neck = SpliceNeck(IN_CH, OUT_CH, SpliceConfig(arm=arm, descriptor=descriptor), workers=0, strides=STRIDES)
    with torch.no_grad():
        for h in neck.heads:                      # eligible cells (e ~ 0.95) so that fragments form
            h.out.bias[9] = 3.0
        neck.score_bias.fill_(2.0)                # most window pairs pass the linkage threshold
    return neck


def inputs(n=2, size=128, seed=1):
    rng = np.random.default_rng(seed)
    feats = [torch.as_tensor(rng.standard_normal((n, c, size // s, size // s)).astype(np.float32))
             for c, s in zip(IN_CH, STRIDES)]
    boxes = []
    for _ in range(n):
        k = int(rng.integers(2, 6))
        xy = rng.uniform(0, size * 0.7, (k, 2))
        wh = rng.uniform(12, 48, (k, 2))
        boxes.append(np.concatenate([xy, np.minimum(xy + wh, size - 1)], 1))
    return feats, boxes


def to_np(t):
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)


class NeckForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.neck = make_neck()
        cls.feats, cls.boxes = inputs()
        cls.out = cls.neck(cls.feats, cls.boxes, probe=True)

    def test_shapes(self):
        out = self.out
        for l, (c, s) in enumerate(zip(OUT_CH, STRIDES)):
            self.assertEqual(tuple(out["cls_feats"][l].shape), (2, c, 128 // s, 128 // s))
            self.assertEqual(tuple(out["box_feats"][l].shape), (2, c, 128 // s, 128 // s))
            self.assertEqual(tuple(out["gates"][l].shape), (2, 1, 128 // s, 128 // s))
        self.assertEqual(tuple(out["window_logits"].shape), (2, 12, 8, 8))

    def test_fragments_and_exchange(self):
        cfg = self.neck.cfg
        frag = self.out["frag"]
        self.assertTrue(all(r["F"] > 0 for r in frag))
        self.assertGreater(self.out["fc_rows"], 0)
        for r in frag:
            F = r["F"]
            self.assertEqual(tuple(r["dh_hat"].shape), (F, cfg.transport))
            self.assertEqual(tuple(r["dH"].shape), (F, cfg.transport))
            adm = r["adm"].cpu().numpy()
            self.assertTrue(np.array_equal(adm, adm.T) and not adm.diagonal().any())
            lam = np.asarray(r["lam"])
            self.assertTrue(np.all(lam[~adm] == 0))                   # sources are admissible partners
            self.assertTrue(np.all((lam >= 0) & (lam <= 2)))
            self.assertTrue(np.all((lam > 0).sum(1) <= cfg.ms))
            # the forecast abstains (zero) for receivers without admissible donors
            empty = ~adm.any(1)
            self.assertTrue(np.all(to_np(r["dh_hat"])[empty] == 0))
            # dh_hat respects the shrink and the clip: |dh| <= (1 - eps) * clip_B
            self.assertLessEqual(float(np.abs(to_np(r["dh_hat"])).max()), (1 - cfg.eps_shrink) * cfg.clip_b + 1e-6)

    def test_fragment_ids_cover_levels(self):
        for r in self.out["frag"]:
            for l, s in enumerate(STRIDES):
                ids = r["ids"][f"P{l + 3}"].cpu().numpy()
                self.assertEqual(ids.size, (128 // s) ** 2)
                self.assertTrue(np.all((ids >= -1) & (ids < r["F"])))

    def test_readout_bands(self):
        """Class band: constant on every fragment's cells, zero on unassigned cells; box band:
        zero fragment mean (Q N = N - Pi N)."""
        neck, out = self.neck, self.out
        for l in range(3):
            N = to_np(out["N"][l])
            for n, r in enumerate(out["frag"]):
                ids = r["ids"][f"P{l + 3}"].cpu().numpy()
                rows = N[n].reshape(N.shape[1], -1).T
                e = np.round(to_np(out["heads"][l]["e"][n, 0]).ravel() * 255.0)
                w = ops.level_weights(NB, e, ids, r["F"])
                rz = to_np(neck.Z[l](r["r"]))
                cb = ops.class_band(NB, rows, ids, w, r["F"], rz)
                bb = ops.box_band(NB, rows, ids, w, r["F"])
                for f in np.unique(ids[ids >= 0])[:20]:
                    self.assertLess(np.abs(cb[ids == f] - cb[ids == f][0]).max(), 1e-5)
                self.assertTrue(np.all(cb[ids < 0] == 0))
                H, _ = ops.restrict(NB, bb, ids, w, r["F"])
                self.assertLess(np.abs(H).max(), 1e-4)

    def test_losses(self):
        aux = aux_losses(self.out, self.boxes, STRIDES, self.neck.cfg.hyp_weight)
        for k in ("e0", "q1", "t2", "g0", "pi1", "pair_p4", "pair_reserve", "hyp"):
            self.assertIn(k, aux)
        for k, v in aux.items():
            self.assertTrue(np.isfinite(float(v)), k)
        det = torch.as_tensor(np.float32(3.0))
        loss, extra = total_loss(det, self.out, aux, self.neck.cfg, 2)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(np.isfinite(extra["fc"]) and np.isfinite(extra["sens"]))
        self.assertGreater(float(loss), 3.0)


class Arms(unittest.TestCase):
    """The seven readout arms of Table 7 and the descriptor variants of arm (e)."""

    def _run(self, arm, descriptor="bands", train=True, seed=7):
        neck = make_neck(seed=seed, arm=arm, descriptor=descriptor)
        with torch.no_grad():                         # make the zero-initialised routes visible
            neck.E_F.weight.copy_(torch.as_tensor(np.eye(neck.cfg.transport, dtype=np.float32)))
            for z in neck.Z:
                z.weight.copy_(torch.as_tensor(np.full(tuple(z.weight.shape), 0.01, dtype=np.float32)))
            if arm == "c":
                for z in neck.score_off:
                    z.weight.copy_(torch.as_tensor(np.full(tuple(z.weight.shape), 0.02, dtype=np.float32)))
        neck.train(train)
        feats, boxes = inputs(n=2, seed=11)
        return neck, neck(feats, boxes, probe=train, stats=True)

    def test_every_arm_runs_in_both_modes(self):
        for arm, spec in ARMS.items():
            for train in (True, False):
                neck, out = self._run(arm, train=train)
                self.assertEqual(len(out["cls_feats"]), 3, arm)
                self.assertTrue(all(np.isfinite(to_np(c)).all() for c in out["cls_feats"] + out["box_feats"]), arm)
                uses = neck.cfg.needs_exchange(train)
                self.assertEqual(uses, spec["exchange"] == "always" or (spec["exchange"] == "train" and train), arm)
                for r in out["frag"]:
                    if r["F"] == 0:
                        continue
                    self.assertEqual(r["dH"] is not None, uses, (arm, train))
                    self.assertEqual(r["dh_hat"] is not None, spec["forecast"] is not None, arm)
                    self.assertEqual(r["r"] is not None, bool(spec["residual"]) and uses, arm)
                self.assertEqual(out["score_offset"] is not None, bool(spec["offset"]), arm)
                has_fc = spec["forecast"] is not None and uses
                self.assertEqual(out["fc_rows"] > 0, has_fc, (arm, train))

    def test_training_exchange_equals_reference(self):
        """The differentiable exchange (training mode) reproduces the exact exchange."""
        neck, out = self._run("e", train=True)
        cfg = neck.cfg
        for r in out["frag"]:
            if r["F"] < 2:
                continue
            ex = exchange_float.exchange(r["adm"].cpu().numpy(), to_np(r["ev"]), to_np(r["need"]), to_np(r["rho"]),
                                         to_np(r["phi"]), to_np(r["vo_t"]), to_np(r["H"])[:, :cfg.transport],
                                         list(to_np(neck.prior)), cfg.a_r, cfg.b_r, cfg.c_r, cfg.ms, cfg.n_cap, cfg.kmax)
            lim = cfg.transport_lsb * 127
            self.assertLess(np.abs(to_np(r["dH"]) - np.clip(ex["dV"], -lim, lim)).max(), 1e-5)
            self.assertTrue(np.all((to_np(r["need"]) >= 0) & (to_np(r["need"]) <= 1)))        # zeta in [0, 1]

    def test_residual_definitions(self):
        for arm in ("b", "d", "e"):
            _, out = self._run(arm, train=True)
            for r in out["frag"]:
                if r["F"] == 0:
                    continue
                want = to_np(r["dH"]) - (to_np(r["dh_hat"]) if r["dh_hat"] is not None else 0.0)
                self.assertLess(np.abs(to_np(r["r"]) - want).max(), 1e-6, arm)

    def test_exposure_discount(self):
        neck, out = self._run("e", train=True)
        eps = neck.cfg.eps_shrink
        seen = 0
        for r in out["frag"]:
            if r["F"] == 0:
                continue
            xi, kappa = to_np(r["xi"]), to_np(r["kappa"])
            self.assertTrue(np.all((xi >= 0) & (xi <= 1 - eps + 1e-9)))
            self.assertTrue(np.all((kappa >= 0) & (kappa <= 0.5 + 1e-9)))            # the lazy walk leaves with p <= 1/2
            seen += xi.size
        self.assertGreater(seen, 0)
        _, plain = self._run("e", descriptor="no_discount", train=True)
        self.assertTrue(all("xi" not in r for r in plain["frag"]))

    def test_descriptor_variants(self):
        for d in DESCRIPTORS:
            neck, out = self._run("e", descriptor=d, train=True)
            for r in out["frag"]:
                if r["F"] == 0:
                    continue
                nu = to_np(r["nu"])
                self.assertEqual(nu.shape, (r["F"], 12), d)
                self.assertTrue(np.isfinite(nu).all() and (nu >= 0).all(), d)
                bound = (1 - neck.cfg.eps_shrink) * neck.cfg.clip_b                # |dHhat| <= alpha (1 - y_0) Theta
                self.assertLessEqual(float(np.abs(to_np(r["dh_hat"])).max()), bound + 1e-6, d)

    def test_free_forecast_is_unconfined(self):
        _, out = self._run("d", train=True)
        for r in out["frag"]:
            if r["F"] < 2:
                continue
            y = to_np(r["y"])
            self.assertTrue(np.allclose(y.sum(1), 1.0, atol=1e-5))                 # no null option
            self.assertTrue(np.all(np.diag(y) == 0))

    def test_score_offset_reaches_the_class_logits(self):
        offs = [torch.ones(1, 1, 2, 2) * (l + 1) for l in range(2)]
        cur = {"one2many": {"boxes": torch.zeros(1, 4, 8), "scores": torch.zeros(1, 3, 8), "feats": []},
               "one2one": {"boxes": torch.zeros(1, 4, 8), "scores": torch.zeros(1, 3, 8), "feats": []}}
        o = _add_offset(cur, offs, 4)
        for k in ("one2many", "one2one"):
            self.assertTrue(np.allclose(to_np(o[k]["scores"])[0, :, :4], 1.0) and np.allclose(to_np(o[k]["scores"])[0, :, 4:], 2.0))
            self.assertEqual(float(o[k]["boxes"].sum()), 0.0)
        old = [torch.zeros(1, 7, 2, 2), torch.zeros(1, 7, 2, 2)]
        o = _add_offset(old, offs, 4)
        self.assertTrue(np.allclose(to_np(o[0])[:, 4:], 1.0) and np.allclose(to_np(o[1])[:, 4:], 2.0))
        self.assertTrue(np.allclose(to_np(o[0])[:, :4], 0.0))


class NeckQAT(unittest.TestCase):
    def test_quantized_sites_on_grid(self):
        neck = make_neck(seed=3)
        neck.quant = NeckQuant(neck, neck.cfg.transport_lsb)
        feats, boxes = inputs(n=1, seed=5)
        out = neck(feats, boxes)
        lsb = neck.cfg.transport_lsb
        for r in out["frag"]:
            if r["F"] == 0:
                continue
            for key, lo, hi in (("H", -128, 127), ("K", -128, 127), ("VO", -128, 127), ("dh_hat", -127, 127), ("dH", -127, 127)):
                code = to_np(r[key]) / lsb
                self.assertTrue(np.allclose(code, np.round(code), atol=1e-4), key)
                self.assertTrue(code.min() >= lo and code.max() <= hi, key)


class NeckEval(unittest.TestCase):
    def test_eval_without_boxes(self):
        neck = make_neck(seed=4)
        neck.eval()
        feats, _ = inputs(n=1, seed=6)
        out = neck(feats, None)
        self.assertNotIn("sens_loss", out)
        self.assertEqual(len(out["cls_feats"]), 3)


class YoloGlue(unittest.TestCase):
    def test_combine_current_format(self):
        a = {"one2many": {"boxes": torch.ones(1, 4, 5), "scores": torch.zeros(1, 2, 5), "feats": [1]},
             "one2one": {"boxes": torch.ones(1, 4, 5) * 2, "scores": torch.zeros(1, 2, 5), "feats": [1]}}
        b = {"one2many": {"boxes": torch.zeros(1, 4, 5), "scores": torch.ones(1, 2, 5) * 3, "feats": [2]},
             "one2one": {"boxes": torch.zeros(1, 4, 5), "scores": torch.ones(1, 2, 5) * 4, "feats": [2]}}
        c = _combine(a, b, 4)
        self.assertEqual(float(c["one2many"]["boxes"].sum()), 20.0)
        self.assertEqual(float(c["one2many"]["scores"].sum()), 30.0)
        self.assertEqual(float(c["one2one"]["boxes"].sum()), 40.0)
        self.assertEqual(float(c["one2one"]["scores"].sum()), 40.0)
        self.assertEqual(c["one2one"]["feats"], [1])

    def test_combine_older_format(self):
        a = [torch.ones(1, 6, 2, 2)]
        b = [torch.ones(1, 6, 2, 2) * 2]
        c = _combine(a, b, 4)[0]
        self.assertEqual(float(c[:, :4].sum()), 16.0)
        self.assertEqual(float(c[:, 4:].sum()), 16.0)

    def test_decode_gate_and_postprocess(self):
        class Head:
            nc, max_det = 2, 3

            def __init__(self, e2e, old):
                self.end2end, self.old = e2e, old

            def _inference(self, x):
                self.seen = x
                return torch.ones(1, 4 + self.nc, 5)

        class OldHead(Head):
            def postprocess(self, preds, max_det, nc=80):
                return ("old", tuple(preds.shape), max_det, nc)

        class NewHead(Head):
            def postprocess(self, preds):
                return ("new", tuple(preds.shape))

        raw = {"one2many": {"tag": "o2m"}, "one2one": {"tag": "o2o"}}
        gates = [torch.full((1, 1, 1, 5), 0.5)]
        model = SpliceYOLO.__new__(SpliceYOLO)
        for head, want in ((NewHead(True, False), ("new", (1, 5, 6))), (OldHead(True, True), ("old", (1, 5, 6), 3, 2))):
            model.__dict__["yolo"] = types.SimpleNamespace(model=[head])
            self.assertEqual(SpliceYOLO.decode(model, raw, gates), want)
            self.assertEqual(head.seen["tag"], "o2o")
        head = NewHead(False, False)
        model.__dict__["yolo"] = types.SimpleNamespace(model=[head])
        y = SpliceYOLO.decode(model, raw, gates)
        self.assertEqual(head.seen["tag"], "o2m")
        self.assertTrue(np.allclose(to_np(y[:, 4:]), 0.5) and np.allclose(to_np(y[:, :4]), 1.0))
        y = SpliceYOLO.decode(model, raw, gates, gate=False)                        # rows of Table 7 without the gate
        self.assertTrue(np.allclose(to_np(y[:, 4:]), 1.0))


if __name__ == "__main__":
    unittest.main()
