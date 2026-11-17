"""Metrics and statistics of the declared tests (detector/splice_det/metrics.py):
CrowdHuman matching and MR^-2 against a literal port of the official evaluation loop, the COCO
matcher and accumulator on cases with known answers, image weights, and the two-level
bootstrap."""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "detector"))

from splice_det import metrics as M  # noqa: E402


def official_crowd(dets, gts, thr=0.5):
    """Port of the official CrowdHuman 'compare_caltech' + eval_MR + eval_AP (loops)."""
    scorelist = []
    total_gt, n_img = 0, len(gts)
    for i, (d, g) in enumerate(zip(dets, gts)):
        order = np.argsort(g["ign"], kind="mergesort")
        boxes, ign = g["boxes"][order], g["ign"][order]
        total_gt += int((~ign).sum())
        d = d[np.argsort(-d[:, 4], kind="mergesort")]
        iou = M.iou_matrix(d[:, :4], boxes)
        ioa = M.ioa_matrix(d[:, :4], boxes)
        gt_matched = np.zeros(len(boxes), dtype=bool)
        for k in range(len(d)):
            maxpos, maxiou = -1, thr
            for j in range(len(boxes)):
                if gt_matched[j]:
                    continue
                if not ign[j]:
                    if iou[k, j] > maxiou:
                        maxiou, maxpos = iou[k, j], j
                else:
                    if maxpos >= 0:
                        break
                    if ioa[k, j] > thr:
                        maxiou, maxpos = ioa[k, j], j
            if maxpos >= 0:
                if not ign[maxpos]:
                    gt_matched[maxpos] = True
                    scorelist.append((d[k, 4], 1, i))
            else:
                scorelist.append((d[k, 4], 0, i))
    scorelist.sort(key=lambda x: -x[0])
    tp = fp = 0
    fppiX, fppiY, rec, prec = [], [], [], []
    for s, m, _ in scorelist:
        tp += m
        fp += 1 - m
        r = tp / total_gt
        rec.append(r); prec.append(tp / (tp + fp))
        fppiX.append(fp / n_img); fppiY.append(1.0 - r)
    score = []
    for ref in M.MR_REFS:
        idx = next((k for k, v in enumerate(fppiX) if v >= ref), len(fppiX) - 1)
        score.append(fppiY[idx])
    mr = float(np.exp(np.log(np.maximum(score, 1e-10)).mean()))
    ap = sum((rec[i] - rec[i - 1]) * (prec[i - 1] + prec[i]) / 2 for i in range(1, len(rec)))
    return {"AP": ap, "MR": mr, "n": len(scorelist), "tp": tp, "gt": total_gt}


def crowd_scene(rng, n_img=40):
    dets, gts = [], []
    for _ in range(n_img):
        m = int(rng.integers(1, 14))
        xy = rng.uniform(0, 500, (m, 2))
        wh = np.stack([rng.uniform(20, 90, m), rng.uniform(50, 220, m)], 1)
        boxes = np.concatenate([xy, xy + wh], 1)
        ign = rng.uniform(0, 1, m) < 0.2
        if ign.all():
            ign[0] = False
        found = rng.uniform(0, 1, m) < 0.8
        d = boxes[found] + rng.normal(0, 6, (int(found.sum()), 4))
        fp_xy = rng.uniform(0, 500, (int(rng.integers(0, 8)), 2))
        fpb = np.concatenate([fp_xy, fp_xy + rng.uniform(20, 150, fp_xy.shape)], 1)
        allb = np.concatenate([d, fpb], 0)
        sc = np.concatenate([rng.uniform(0.3, 1.0, len(d)), rng.uniform(0.0, 0.7, len(fpb))])
        dets.append(np.concatenate([allb, np.round(sc[:, None], 3)], 1))
        gts.append({"boxes": boxes, "ign": ign})
    return dets, gts


class CrowdHuman(unittest.TestCase):
    def test_against_the_official_loop(self):
        rng = np.random.default_rng(0)
        for trial in range(6):
            dets, gts = crowd_scene(rng)
            ref = official_crowd(dets, gts)
            arr = M.crowd_match(dets, gts)
            got = M.crowd_metrics(arr)
            self.assertEqual(len(arr["score"]), ref["n"], trial)
            self.assertEqual(int(arr["tp"].sum()), ref["tp"], trial)
            self.assertEqual(int(arr["n_gt"].sum()), ref["gt"], trial)
            self.assertAlmostEqual(got["MR"], ref["MR"], places=12)
            self.assertAlmostEqual(got["AP"], ref["AP"], places=12)

    def test_ignore_regions(self):
        g = [{"boxes": np.array([[0, 0, 10, 10], [20, 0, 40, 40.0]]), "ign": np.array([False, True])}]
        d = [np.array([[0, 0, 10, 10, 0.9], [22, 2, 30, 30, 0.8], [100, 100, 110, 110, 0.7]])]
        arr = M.crowd_match(d, g)
        self.assertEqual(arr["tp"].tolist(), [True, False])            # the detection on the ignore region is dropped
        self.assertEqual(arr["score"].tolist(), [0.9, 0.7])

    def test_strata(self):
        gt = {"a": {"width": 100, "height": 100, "boxes": [
            {"fbox": [0, 0, 10, 10], "fbox_clip": [0, 0, 10, 10], "ignore": 0, "vis": 0.9},
            {"fbox": [20, 0, 30, 10], "fbox_clip": [20, 0, 30, 10], "ignore": 0, "vis": 0.4},
            {"fbox": [-5, 50, 30, 80], "fbox_clip": [0, 50, 30, 80], "ignore": 1, "vis": 1.0}]}}
        allg = M.crowd_ground_truth(gt, ["a"], "all")[0]
        heavy = M.crowd_ground_truth(gt, ["a"], "heavy")[0]
        self.assertEqual(allg["ign"].tolist(), [False, False, True])
        self.assertEqual(heavy["ign"].tolist(), [True, False, True])    # only the occluded person is a positive
        raw = M.crowd_ground_truth(gt, ["a"], "all", clip=False)[0]
        self.assertEqual(raw["boxes"][2].tolist(), [-5, 50, 30, 80])

    def test_weights(self):
        rng = np.random.default_rng(3)
        dets, gts = crowd_scene(rng, 30)
        arr = M.crowd_match(dets, gts)
        base = M.crowd_metrics(arr)
        twice = M.crowd_metrics(arr, np.full(30, 2.0))
        for k in base:
            self.assertAlmostEqual(base[k], twice[k], places=12)
        w = np.ones(30)
        w[10:] = 0
        sub = M.crowd_metrics(M.crowd_match(dets[:10], gts[:10]))
        got = M.crowd_metrics(arr, w)
        for k in sub:
            self.assertAlmostEqual(sub[k], got[k], places=12)


def coco_truth(boxes_by_img, crowd=None):
    gt = {}
    for i, rows in boxes_by_img.items():
        for c, b in rows:
            gt.setdefault(i, {}).setdefault(c, []).append(tuple(b) + (int(bool(crowd and (i, c, tuple(b)) in crowd)),))
    return {"cats": [1, 2, 3], "gt": gt}


class Coco(unittest.TestCase):
    def test_perfect_and_empty(self):
        truth = coco_truth({7: [(1, (0, 0, 10, 10)), (2, (20, 20, 40, 60))], 9: [(1, (5, 5, 25, 25))]})
        dets = {7: np.array([[0, 0, 10, 10, 0.9, 1], [20, 20, 40, 60, 0.8, 2]]), 9: np.array([[5, 5, 25, 25, 0.7, 1.0]])}
        arr = M.coco_match_numpy(dets, truth, [7, 9])
        r = M.CocoAccumulator(arr)()
        self.assertAlmostEqual(r["AP"], 1.0, places=9)
        self.assertEqual(arr["npig"].sum(), 3)
        none = M.CocoAccumulator(M.coco_match_numpy({}, truth, [7, 9]))()
        self.assertEqual(none["AP"], 0.0)

    def test_iou_thresholds(self):
        # one object, one detection with IoU 0.6: counted at thresholds 0.50, 0.55, 0.60 only
        truth = coco_truth({1: [(1, (0, 0, 10, 10))]})
        dets = {1: np.array([[0, 0, 10, 6, 0.9, 1.0]])}
        arr = M.coco_match_numpy(dets, truth, [1])
        self.assertEqual(arr["det_tp"][:, 0].tolist(), [True] * 3 + [False] * 7)
        r = M.CocoAccumulator(arr)()
        self.assertAlmostEqual(r["AP"], 0.3, places=9)
        self.assertAlmostEqual(r["AP50"], 1.0, places=9)
        self.assertAlmostEqual(r["AP75"], 0.0, places=9)

    def test_ranking_and_duplicates(self):
        # two objects; detections: hit (0.9), duplicate of the first (0.8, a false positive), hit (0.7)
        truth = coco_truth({1: [(1, (0, 0, 10, 10)), (1, (50, 50, 70, 70))]})
        dets = {1: np.array([[0, 0, 10, 10, 0.9, 1], [0, 0, 10, 10, 0.8, 1], [50, 50, 70, 70, 0.7, 1.0]])}
        r = M.CocoAccumulator(M.coco_match_numpy(dets, truth, [1]))()
        # precision 1 up to recall 0.5 (51 of the 101 recall points), 2/3 beyond
        self.assertAlmostEqual(r["AP50"], (51 + 50 * 2 / 3) / 101, places=9)

    def test_crowd_is_ignored(self):
        box = (0, 0, 100, 100)
        truth = coco_truth({1: [(1, box), (1, (200, 200, 220, 220))]}, crowd={(1, 1, box)})
        dets = {1: np.array([[10, 10, 30, 30, 0.9, 1], [200, 200, 220, 220, 0.8, 1.0]])}
        arr = M.coco_match_numpy(dets, truth, [1])
        self.assertTrue(arr["det_ign"][:, 0].all())                     # inside the crowd region: ignored
        self.assertAlmostEqual(M.CocoAccumulator(arr)()["AP"], 1.0, places=9)

    def test_weights(self):
        rng = np.random.default_rng(5)
        gt, dets = {}, {}
        for i in range(30):
            rows = []
            for _ in range(int(rng.integers(1, 5))):
                xy = rng.uniform(0, 300, 2)
                rows.append((int(rng.integers(1, 4)), tuple(np.concatenate([xy, xy + rng.uniform(20, 80, 2)]))))
            gt[i] = rows
            d = [list(np.asarray(b) + rng.normal(0, 4, 4)) + [float(np.round(rng.uniform(0.2, 1), 3)), c] for c, b in rows
                 if rng.uniform() < 0.8]
            d += [list(rng.uniform(0, 300, 2)) + list(rng.uniform(300, 380, 2)) + [float(np.round(rng.uniform(0, 0.6), 3)),
                                                                                   int(rng.integers(1, 4))]
                  for _ in range(int(rng.integers(0, 3)))]
            dets[i] = np.array(d, dtype=np.float64).reshape(-1, 6)
        truth = coco_truth(gt)
        ids = list(range(30))
        acc = M.CocoAccumulator(M.coco_match_numpy(dets, truth, ids))
        base = acc()
        self.assertTrue(0.0 < base["AP"] < 1.0)
        twice = acc(np.full(30, 2.0))
        w = np.ones(30)
        w[12:] = 0
        sub = M.CocoAccumulator(M.coco_match_numpy(dets, truth, ids[:12]))()
        got = acc(w)
        for k in base:
            self.assertAlmostEqual(base[k], twice[k], places=9)
            self.assertAlmostEqual(sub[k], got[k], places=9)


class Bootstrap(unittest.TestCase):
    def test_effect_and_bounds(self):
        rng = np.random.default_rng(0)
        n = 400
        per_img_a = [rng.normal(0.50, 0.2, n) for _ in range(3)]
        per_img_b = [x - 0.02 + rng.normal(0, 0.01, n) for x in per_img_a]          # arm b is 0.02 lower
        mk = lambda v: (lambda w: float((v * w).sum() / w.sum()))
        st = M.two_level_bootstrap([mk(v) for v in per_img_a], [mk(v) for v in per_img_b], n, n_boot=400, seed=1)
        self.assertAlmostEqual(st["effect"], 0.02, delta=0.003)
        self.assertLess(st["lower"], st["effect"])
        self.assertGreater(st["upper"], st["effect"])
        self.assertGreater(st["se"], 0)
        self.assertEqual(M.non_inferiority(st, 0.01, True)["decision"], "PASS")       # a - b > -0.01
        rev = M.two_level_bootstrap([mk(v) for v in per_img_b], [mk(v) for v in per_img_a], n, n_boot=400, seed=1)
        self.assertEqual(M.non_inferiority(rev, 0.01, True)["decision"], "FAIL")      # b - a = -0.02 < -0.01
        self.assertEqual(M.non_inferiority(rev, 0.05, True)["decision"], "PASS")
        self.assertEqual(M.non_inferiority(st, 0.01, False)["decision"], "FAIL")      # lower-is-better: +0.02 > 0.01

    def test_bootstrap_is_the_same_for_any_number_of_workers(self):
        rng = np.random.default_rng(5)
        n = 300
        va = [rng.normal(0.5, 0.1, n) for _ in range(3)]
        vb = [rng.normal(0.48, 0.1, n) for _ in range(2)]                             # unpaired seeds
        mk = lambda v: (lambda w: float((v * w).sum() / w.sum()))
        one = M.two_level_bootstrap([mk(v) for v in va], [mk(v) for v in vb], n, n_boot=240, seed=7, workers=1)
        many = M.two_level_bootstrap([mk(v) for v in va], [mk(v) for v in vb], n, n_boot=240, seed=7, workers=3)
        self.assertEqual(one, many)
        other = M.two_level_bootstrap([mk(v) for v in va], [mk(v) for v in vb], n, n_boot=240, seed=8, workers=3)
        self.assertNotEqual(one["lower"], other["lower"])
        self.assertFalse(M._BOOT)                                                     # nothing is kept after a test

    def test_missing_seeds_are_not_a_pass(self):
        f = lambda w: 1.0
        st = M.two_level_bootstrap([f], [f], 10, n_boot=20)
        d = M.non_inferiority(st, 0.1, True)
        self.assertEqual(d["decision"], "NOT_QUALIFIED")
        self.assertTrue(d["would_pass"])


if __name__ == "__main__":
    unittest.main()
