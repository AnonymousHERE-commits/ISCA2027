"""Detection metrics and the statistics of the declared tests (SPLICE-V section 4, Table 6).

COCO          AP, AP50, AP75 by the COCO protocol.  Matching is done by pycocotools when it
              is installed (and by the same rules in numpy otherwise); the matches are kept
              per image as arrays, so that the AP of a resampled set of images is a weighted
              accumulation without re-matching.
CrowdHuman    AP and the log-average miss rate MR^-2 (nine FPPI points in [1e-2, 1]) with the
              rules of the official evaluation: score-ordered greedy matching at IoU 0.5,
              ignore regions matched by intersection over the detection's area, detections
              on ignore regions neither true nor false.  Strata: 'all' and 'heavy' (visible
              ratio below 0.65; the other persons are then ignore regions).
Tests         two-level bootstrap over seeds and whole images; effect, standard error and
              the one-sided bound at level alpha.

Everything here is numpy and is unit-tested without a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
MR_REFS = np.array([0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000])
HEAVY_VISIBLE = 0.65


# ----------------------------------------------------------------------------- geometry

def box_area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def intersections(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = intersections(a, b)
    return inter / np.maximum(box_area(a)[:, None] + box_area(b)[None, :] - inter, 1e-12)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of the first box (the detection)."""
    return intersections(a, b) / np.maximum(box_area(a)[:, None], 1e-12)


# ----------------------------------------------------------------------------- CrowdHuman

def crowd_ground_truth(gt: Dict, ids: Sequence[str], stratum: str = "all", clip: bool = True,
                       heavy: float = HEAVY_VISIBLE) -> List[Dict]:
    """Per image {'boxes' (m, 4), 'ign' (m,) bool} for a stratum of gt_val.json."""
    out = []
    key = "fbox_clip" if clip else "fbox"
    for i in ids:
        bs = gt[i]["boxes"]
        boxes = np.array([b[key] for b in bs], dtype=np.float64).reshape(-1, 4)
        ign = np.array([bool(b["ignore"]) for b in bs], dtype=bool)
        if stratum == "heavy":
            vis = np.array([b["vis"] for b in bs], dtype=np.float64)
            ign = ign | (vis >= heavy)
        elif stratum != "all":
            raise ValueError(stratum)
        out.append({"boxes": boxes, "ign": ign})
    return out


def crowd_match(dets: Sequence[np.ndarray], gts: Sequence[Dict], thr: float = 0.5) -> Dict[str, np.ndarray]:
    """Official CrowdHuman matching.  dets: per image (n, >= 5) [x0, y0, x1, y1, score].
    Returns the arrays of the counted detections (those on ignore regions are dropped):
    score, tp, img (image index) and n_gt (per image)."""
    scores, tps, imgs = [], [], []
    n_gt = np.zeros(len(gts), dtype=np.int64)
    for i, (d, g) in enumerate(zip(dets, gts)):
        real = g["boxes"][~g["ign"]]
        igb = g["boxes"][g["ign"]]
        n_gt[i] = len(real)
        if len(d) == 0:
            continue
        d = d[np.argsort(-d[:, 4], kind="mergesort")]
        iou = iou_matrix(d[:, :4], real)
        ioa = ioa_matrix(d[:, :4], igb)
        free = np.ones(len(real), dtype=bool)
        for k in range(len(d)):
            j = -1
            if len(real):
                cand = np.where(free, iou[k], -1.0)
                j = int(np.argmax(cand))
                if cand[j] <= thr:
                    j = -1
            if j >= 0:
                free[j] = False
                scores.append(d[k, 4]); tps.append(True); imgs.append(i)
            elif len(igb) and (ioa[k] > thr).any():
                continue                                    # on an ignore region: not counted
            else:
                scores.append(d[k, 4]); tps.append(False); imgs.append(i)
    return {"score": np.asarray(scores, dtype=np.float64), "tp": np.asarray(tps, dtype=bool),
            "img": np.asarray(imgs, dtype=np.int64), "n_gt": n_gt}


def crowd_metrics(arr: Dict[str, np.ndarray], weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """AP (trapezoid over the recall points), MR^-2 and the recall of all counted detections;
    `weights` (per image) gives the metric of a resampled set of images."""
    n_img = len(arr["n_gt"])
    w = np.ones(n_img) if weights is None else np.asarray(weights, dtype=np.float64)
    total_gt = float((arr["n_gt"] * w).sum())
    images = float(w.sum())
    if total_gt <= 0 or images <= 0:
        return {"AP": float("nan"), "MR": float("nan"), "recall": float("nan")}
    o = np.argsort(-arr["score"], kind="mergesort")
    wd = w[arr["img"][o]]
    tp = np.cumsum(arr["tp"][o] * wd)
    fp = np.cumsum((~arr["tp"][o]) * wd)
    keep = wd > 0
    tp, fp = tp[keep], fp[keep]
    if tp.size == 0:
        return {"AP": 0.0, "MR": 1.0, "recall": 0.0}
    rec = tp / total_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    ap = float(np.sum((rec[1:] - rec[:-1]) * (prec[1:] + prec[:-1]) / 2.0))
    fppi = fp / images
    miss = 1.0 - rec
    idx = np.searchsorted(fppi, MR_REFS, side="left")               # first point with fppi >= ref
    idx = np.where(idx < fppi.size, idx, fppi.size - 1)
    mr = float(np.exp(np.mean(np.log(np.maximum(miss[idx], 1e-10)))))
    return {"AP": ap, "MR": mr, "recall": float(rec[-1])}


# ----------------------------------------------------------------------------- COCO

def coco_ground_truth(coco_json: str, img_ids: Optional[Sequence[int]] = None) -> Dict:
    js = json.loads(Path(coco_json).read_text())
    cats = sorted(c["id"] for c in js["categories"])
    keep = set(int(i) for i in img_ids) if img_ids is not None else None
    gt: Dict[int, Dict[int, List]] = {}
    for a in js["annotations"]:
        i = int(a["image_id"])
        if keep is not None and i not in keep:
            continue
        x, y, w, h = a["bbox"]
        gt.setdefault(i, {}).setdefault(int(a["category_id"]), []).append(
            (x, y, x + w, y + h, int(a.get("iscrowd", 0) or a.get("ignore", 0))))
    return {"cats": cats, "gt": gt}


def coco_match_numpy(dets: Dict[int, np.ndarray], truth: Dict, img_ids: Sequence[int], max_det: int = 100) -> Dict:
    """COCO matching (COCOeval.evaluateImg, area range 'all') in numpy.
    dets[image_id]: (n, 6) [x0, y0, x1, y1, score, category_id]."""
    cats = truth["cats"]
    cidx = {c: k for k, c in enumerate(cats)}
    T = len(IOU_THRS)
    sc, im, ct, tp, ig = [], [], [], [], []
    npig = np.zeros((len(cats), len(img_ids)), dtype=np.int64)
    for i, iid in enumerate(img_ids):
        g_img = truth["gt"].get(int(iid), {})
        d_img = dets.get(int(iid), np.zeros((0, 6)))
        for c in set(g_img) | set(int(x) for x in np.unique(d_img[:, 5])) if len(d_img) else set(g_img):
            if c not in cidx:
                continue
            g = np.array(g_img.get(c, []), dtype=np.float64).reshape(-1, 5)
            crowd = g[:, 4] > 0
            order = np.argsort(crowd, kind="mergesort")                       # ignored ground truth last
            g, crowd = g[order], crowd[order]
            npig[cidx[c], i] = int((~crowd).sum())
            d = d_img[d_img[:, 5] == c] if len(d_img) else np.zeros((0, 6))
            if len(d) == 0:
                continue
            d = d[np.argsort(-d[:, 4], kind="mergesort")][:max_det]
            inter = intersections(d[:, :4], g[:, :4])
            ad, ag = box_area(d[:, :4]), box_area(g[:, :4])
            union = np.where(crowd[None, :], ad[:, None], ad[:, None] + ag[None, :] - inter)
            ious = inter / np.maximum(union, 1e-12)
            dtm = np.zeros((T, len(d)), dtype=bool)
            dti = np.zeros((T, len(d)), dtype=bool)
            for t, thr in enumerate(IOU_THRS):
                taken = np.zeros(len(g), dtype=bool)
                for k in range(len(d)):
                    best, m = min(thr, 1 - 1e-10), -1
                    for j in range(len(g)):
                        if taken[j] and not crowd[j]:
                            continue
                        if m > -1 and not crowd[m] and crowd[j]:
                            break
                        if ious[k, j] < best:
                            continue
                        best, m = ious[k, j], j
                    if m == -1:
                        continue
                    dti[t, k] = crowd[m]
                    dtm[t, k] = True
                    taken[m] = True
            sc.append(d[:, 4]); im.append(np.full(len(d), i)); ct.append(np.full(len(d), cidx[c]))
            tp.append(dtm); ig.append(dti)
    cat = lambda xs, ax=0: np.concatenate(xs, axis=ax) if xs else np.zeros((0,))
    return {"cats": np.asarray(cats), "img_ids": np.asarray(img_ids), "det_score": cat(sc), "det_img": cat(im).astype(np.int64),
            "det_cat": cat(ct).astype(np.int64), "det_tp": cat(tp, 1) if tp else np.zeros((T, 0), dtype=bool),
            "det_ign": cat(ig, 1) if ig else np.zeros((T, 0), dtype=bool), "npig": npig}


def coco_match_pycocotools(dets: Dict[int, np.ndarray], coco_json: str, img_ids: Sequence[int]):
    """The same arrays from pycocotools' matching, with its summary statistics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(coco_json)
    res = []
    for iid in img_ids:
        for x0, y0, x1, y1, s, c in dets.get(int(iid), np.zeros((0, 6)))[:, :6]:
            res.append({"image_id": int(iid), "category_id": int(c), "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "score": float(s)})
    cats = sorted(gt.getCatIds())
    T = len(IOU_THRS)
    if not res:
        return {"cats": np.asarray(cats), "img_ids": np.asarray(img_ids), "det_score": np.zeros(0), "det_img": np.zeros(0, dtype=np.int64),
                "det_cat": np.zeros(0, dtype=np.int64), "det_tp": np.zeros((T, 0), dtype=bool), "det_ign": np.zeros((T, 0), dtype=bool),
                "npig": np.zeros((len(cats), len(img_ids)), dtype=np.int64)}, [0.0] * 12
    ev = COCOeval(gt, gt.loadRes(res), "bbox")
    ev.params.imgIds = [int(i) for i in img_ids]
    ev.evaluate()
    p = ev.params
    I, A, max_det = len(p.imgIds), len(p.areaRng), p.maxDets[-1]
    pos = {int(i): k for k, i in enumerate(img_ids)}
    sc, im, ct, tp, ig = [], [], [], [], []
    npig = np.zeros((len(p.catIds), len(img_ids)), dtype=np.int64)
    for k in range(len(p.catIds)):
        for i in range(I):
            e = ev.evalImgs[k * A * I + i]                      # area range 0 = all
            if e is None:
                continue
            col = pos[int(e["image_id"])]
            npig[k, col] = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
            n = min(len(e["dtScores"]), max_det)
            if n == 0:
                continue
            sc.append(np.asarray(e["dtScores"][:n], dtype=np.float64))
            im.append(np.full(n, col)); ct.append(np.full(n, k))
            tp.append(np.asarray(e["dtMatches"])[:, :n] > 0)
            ig.append(np.asarray(e["dtIgnore"])[:, :n].astype(bool))
    ev.accumulate()
    ev.summarize()
    cat = lambda xs, ax=0: np.concatenate(xs, axis=ax) if xs else np.zeros((0,))
    arr = {"cats": np.asarray(p.catIds), "img_ids": np.asarray(img_ids), "det_score": cat(sc), "det_img": cat(im).astype(np.int64),
           "det_cat": cat(ct).astype(np.int64), "det_tp": cat(tp, 1) if tp else np.zeros((T, 0), dtype=bool),
           "det_ign": cat(ig, 1) if ig else np.zeros((T, 0), dtype=bool), "npig": npig}
    return arr, [float(x) for x in ev.stats]


class CocoAccumulator:
    """COCOeval.accumulate on the kept matches, with per-image weights."""

    def __init__(self, arr: Dict):
        self.n_img = len(arr["img_ids"])
        self.npig = np.asarray(arr["npig"], dtype=np.float64)
        self.cats = []
        for k in range(self.npig.shape[0]):
            sel = np.flatnonzero(arr["det_cat"] == k)
            o = sel[np.argsort(-arr["det_score"][sel], kind="mergesort")]
            real = ~arr["det_ign"][:, o]
            self.cats.append((arr["det_img"][o], (arr["det_tp"][:, o] & real).astype(np.float64),
                              (~arr["det_tp"][:, o] & real).astype(np.float64)))

    def __call__(self, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        w = np.ones(self.n_img) if weights is None else np.asarray(weights, dtype=np.float64)
        T, R = len(IOU_THRS), len(REC_THRS)
        prec = []
        for k, (img, tpm, fpm) in enumerate(self.cats):
            n_pos = float((self.npig[k] * w).sum())
            if n_pos <= 0:
                continue
            q = np.zeros((T, R))
            if img.size:
                wd = w[img]
                tp = np.cumsum(tpm * wd[None, :], axis=1)
                fp = np.cumsum(fpm * wd[None, :], axis=1)
                rc = tp / n_pos
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                for t in range(T):
                    inds = np.searchsorted(rc[t], REC_THRS, side="left")
                    ok = inds < img.size
                    q[t, ok] = pr[t, inds[ok]]
            prec.append(q)
        if not prec:
            return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}
        P = np.stack(prec, 2)                                    # (T, R, K')
        return {"AP": float(P.mean()), "AP50": float(P[0].mean()), "AP75": float(P[5].mean())}


# ----------------------------------------------------------------------------- declared tests

_BOOT: Dict = {}                 # the bootstrap's arms, shared with forked workers (large arrays: never pickled)


def _boot_chunk(indices: Sequence[int]) -> np.ndarray:
    a, b, n_img, paired, seed = _BOOT["a"], _BOOT["b"], _BOOT["n_img"], _BOOT["paired"], _BOOT["seed"]
    out = np.empty(len(indices))
    for j, i in enumerate(indices):
        rng = np.random.default_rng([seed, int(i)])               # one stream per replicate: the result does not
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float64)   # depend on the worker count
        sa = rng.integers(0, len(a), len(a))
        sb = sa if paired else rng.integers(0, len(b), len(b))
        va = {int(s): a[int(s)](w) for s in set(sa.tolist())}
        vb = {int(s): b[int(s)](w) for s in set(sb.tolist())}
        out[j] = np.mean([va[int(s)] for s in sa]) - np.mean([vb[int(s)] for s in sb])
    return out


def two_level_bootstrap(a: Sequence[Callable], b: Sequence[Callable], n_img: int, n_boot: int = 2000, seed: int = 0,
                        alpha: float = 0.05 / 4, workers: int = 1) -> Dict[str, float]:
    """Difference of two arms, mean over seeds of metric(a) - metric(b).

    a, b: one callable per seed, weights (n_img,) -> metric on the reweighted images (the
    same images for both arms).  Level 1 resamples the seeds (jointly when both arms have
    the same number of seeds), level 2 the whole images.  Returns the effect on the full
    data, the bootstrap standard error and the one-sided bounds at level alpha.

    workers > 1 spreads the replicates over forked processes; every replicate has its own
    random stream, so the result is the same for any number of workers."""
    full = np.ones(n_img)
    effect = float(np.mean([f(full) for f in a]) - np.mean([f(full) for f in b]))
    _BOOT.update(a=list(a), b=list(b), n_img=int(n_img), paired=len(a) == len(b), seed=int(seed))
    try:
        idx = np.arange(n_boot)
        diffs = None
        if workers > 1 and n_boot >= 4 * workers:
            try:
                import multiprocessing as mp
                with mp.get_context("fork").Pool(workers) as pool:
                    diffs = np.concatenate(pool.map(_boot_chunk, np.array_split(idx, 4 * workers)))
            except (ValueError, OSError):                        # no fork on this platform
                diffs = None
        if diffs is None:
            diffs = _boot_chunk(idx)
    finally:
        _BOOT.clear()
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"effect": effect, "se": float("nan"), "lower": float("nan"), "upper": float("nan"), "n_boot": 0,
                "seeds_a": len(a), "seeds_b": len(b), "alpha": alpha}
    return {"effect": effect, "se": float(diffs.std(ddof=1)) if diffs.size > 1 else float("nan"),
            "lower": float(np.quantile(diffs, alpha)), "upper": float(np.quantile(diffs, 1.0 - alpha)),
            "n_boot": int(diffs.size), "seeds_a": len(a), "seeds_b": len(b), "alpha": alpha}


def non_inferiority(stat: Dict[str, float], margin: float, higher_is_better: bool, min_seeds: int = 3) -> Dict:
    """Decision of a one-sided non-inferiority test: for a metric where higher is better the
    lower bound of (candidate - reference) must exceed -margin; where lower is better the
    upper bound must stay below +margin.  Fewer seeds than declared is not a pass."""
    bound = stat["lower"] if higher_is_better else stat["upper"]
    if not np.isfinite(bound):
        return {"decision": "NOT_EVALUATED", "reason": "no finite bootstrap bound", "bound": None, "margin": margin}
    ok = bound > -margin if higher_is_better else bound < margin
    if min(stat["seeds_a"], stat["seeds_b"]) < min_seeds:
        return {"decision": "NOT_QUALIFIED", "reason": f"fewer than {min_seeds} seeds per arm", "bound": bound,
                "margin": margin, "would_pass": bool(ok)}
    return {"decision": "PASS" if ok else "FAIL", "bound": bound, "margin": margin}
