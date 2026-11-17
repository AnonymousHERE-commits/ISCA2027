"""Evaluate a SPLICE checkpoint or the native YOLO26-S on a prepared dataset view.

    python -m splice_det.evaluate --ckpt runs/coco_e_s0/trained.pt --dataset <views>/coco/dataset.json --out eval/
    python -m splice_det.evaluate --native yolo26s.pt --dataset <views>/coco/dataset.json --out eval/

COCO: AP, AP50, AP75 (pycocotools matching and summary; the kept matches reproduce its AP).
CrowdHuman: AP and MR^-2 by the rules of the official evaluation, overall and on heavily
occluded persons (visible ratio below 0.65).  SPLICE models are evaluated with and without
the objectness gate.  Besides metrics.json the folder receives the per-image matches
(matches_*.npz, the input of the bootstrap tests) and frames.json (the regime variables of
every frame: fragments, candidates per row, cap and budget exits, skip shares).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import metrics


def regime_summary(frames) -> Optional[Dict]:
    if not frames:
        return None
    F = np.array([f["F"] for f in frames], dtype=np.float64)
    pos = F[F > 0]
    part = np.array([f["partners"] for f in frames], dtype=np.float64)
    ratio = part[F > 0] / pos if pos.size else np.zeros(0)
    q = lambda a, p: float(np.quantile(a, p)) if a.size else None
    return {"frames": len(frames), "F_mean": float(F.mean()), "F_p50": q(F, 0.5), "F_p99": q(F, 0.99), "F_max": float(F.max()),
            "F_log_std": float(np.log(pos).std()) if pos.size > 1 else None,
            "F_cv": float(pos.std() / pos.mean()) if pos.size > 1 else None,
            "cap_frames_share": float(np.mean([f["overflow"] or f["F"] >= 256 for f in frames])),
            "budget_exit_share": float(np.mean([f["budget_hit"] for f in frames])),
            "rounds_mean": float(np.mean([f["rounds"] for f in frames])),
            "N_f_mean": float(part.mean()), "N_over_F_mean": float(ratio.mean()) if ratio.size else None,
            "N_f_max": int(max(f["partners_max"] for f in frames)),
            "skipped_share": {f"block{b}": float(np.mean([f[f"skip{b}"] for f in frames])) for b in (1, 2, 4)}}


def evaluate_model(model, data: Dict, device, out_dir: Path, batch: int = 16, workers: int = 8, imgsz: int = 640,
                   limit: Optional[int] = None, theta: float = 0.25) -> Dict:
    from .data import make_loader
    from .engine import collect_detections
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(data, "val", "val", batch, workers, imgsz, shuffle=False)
    res = collect_detections(model, loader, device, limit, theta)
    files = res["files"]
    rec: Dict = {"dataset": data["name"], "images": len(files), "variants": {}, "limit": limit}
    if data["name"] == "coco":
        ids = [int(Path(f).stem) for f in files]
        cat_ids = data["category_ids"]
        for key, dets in res["dets"].items():
            by_img = {}
            for iid, d in zip(ids, dets):
                dd = d.copy()
                if len(dd):
                    dd[:, 5] = [cat_ids[int(c)] for c in d[:, 5]]
                by_img[iid] = dd
            try:
                arr, stats = metrics.coco_match_pycocotools(by_img, data["coco_json"], ids)
                matcher = "pycocotools"
            except ImportError:
                arr, stats = metrics.coco_match_numpy(by_img, metrics.coco_ground_truth(data["coco_json"], ids), ids), None
                matcher = "numpy (pycocotools not installed)"
            m = metrics.CocoAccumulator(arr)()
            v = {**m, "matcher": matcher, "detections": int(arr["det_score"].size)}
            if stats is not None:
                v["pycocotools"] = dict(zip(("AP", "AP50", "AP75", "APs", "APm", "APl"), stats[:6]))
                v["accumulate_abs_diff"] = abs(stats[0] - m["AP"])
            np.savez_compressed(out_dir / f"matches_{key}.npz", **arr)
            rec["variants"][key] = v
    elif data["name"] == "crowdhuman":
        gt = json.loads(Path(data["gt_val"]).read_text())
        ids = [Path(f).stem for f in files]
        for key, dets in res["dets"].items():
            dd = [d[d[:, 5] == 0][:, :5] if len(d) else np.zeros((0, 5)) for d in dets]
            v = {}
            for stratum in ("all", "heavy"):
                arr = metrics.crowd_match(dd, metrics.crowd_ground_truth(gt, ids, stratum))
                m = metrics.crowd_metrics(arr)
                v[stratum] = {"AP": m["AP"], "MR": m["MR"], "recall": m["recall"], "gt": int(arr["n_gt"].sum()),
                              "counted_detections": int(arr["score"].size)}
                np.savez_compressed(out_dir / f"matches_{key}_{stratum}.npz", **arr)
            rec["variants"][key] = v
        rec["ground_truth"] = "full-body boxes clipped to the image; ignore regions by the official rule"
    else:
        raise ValueError(f"unknown dataset {data['name']}")
    rec["regime"] = regime_summary(res["frames"])
    (out_dir / "frames.json").write_text(json.dumps({"files": [Path(f).name for f in files], "frames": res["frames"]}) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--native", help="YOLO26-S weights evaluated as they are")
    ap.add_argument("--dataset", required=True, help="dataset.json of a prepared view")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--link-workers", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    import torch
    from .data import load_dataset
    from .engine import build_model, load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(a.dataset)
    if a.ckpt:
        model, _ = load_model(a.ckpt, device, a.link_workers, a.imgsz)
        src = a.ckpt
    else:
        model = build_model(a.native, device, "native")
        src = a.native
    rec = evaluate_model(model, data, device, Path(a.out), a.batch, a.workers, a.imgsz, a.limit)
    rec["source"] = src
    rec["sha256"] = hashlib.sha256(Path(src).read_bytes()).hexdigest() if Path(src).exists() else None
    (Path(a.out) / "metrics.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "regime"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
