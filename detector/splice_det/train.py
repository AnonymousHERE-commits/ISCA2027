"""Train one SPLICE-B run on YOLO26-S outside the Slurm suite:
train -> calibrate -> (arm e) quantization-aware training -> export.

    python -m splice_det.train --dataset <views>/coco/dataset.json --weights yolo26s.pt \
        --arm e --seed 0 --out runs/coco_e_s0

The Arrhenius suite (detector/hpc/run_splicev.sh) runs the same stages as separate,
gated Slurm jobs for every arm, descriptor, dataset and seed of the protocol; this tool is for
a single run.  The Ultralytics data pipeline (mosaic, augmentation, letterbox to 640) and
YOLO26's own detection criterion are reused; the loop, the SPLICE neck, its auxiliary losses,
the calibration and QAT are this package's.  The backbone and head start from the published
YOLO26-S weights; the neck is new.  --arm native trains YOLO26-S itself with the same loop.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import ARMS, DESCRIPTORS


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset.json of a prepared view (splice_det.data)")
    ap.add_argument("--weights", default="yolo26s.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="e", choices=sorted(ARMS) + ["native"])
    ap.add_argument("--descriptor", default="bands", choices=sorted(DESCRIPTORS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--qat-epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8, help="data loader workers")
    ap.add_argument("--link-workers", type=int, default=8, help="processes for reserve / linkage / exchange decisions (CPU)")
    ap.add_argument("--lr", type=float, default=0.01, help="neck learning rate (SGD)")
    ap.add_argument("--lr-pretrained", type=float, default=0.001, help="backbone / head learning rate")
    ap.add_argument("--warmup-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-batches", type=int, default=200)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the training images")
    ap.add_argument("--max-iters", type=int, help="iterations per epoch (short test runs)")
    return ap.parse_args()


def main() -> int:
    a = parse()
    import torch
    from . import engine, export, qat
    from .data import load_dataset
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.jsonl", "a")

    def log(rec):
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(a.dataset)
    engine.set_seed(a.seed)
    model = engine.build_model(a.weights, device, a.arm, a.descriptor, a.link_workers, a.imgsz, data["nc"])
    splice = model.neck is not None
    meta = {"weights": a.weights, "nc": data["nc"], "seed": a.seed, "dataset": data["name"], "arm": a.arm,
            "descriptor": a.descriptor}
    log({"stage": "setup", "device": str(device), "torch": torch.__version__, **meta,
         "layout": getattr(model, "layout", None)})
    st = engine.train_epochs(model, data, out / "last.pt", epochs=a.epochs, batch=a.batch, workers=a.workers, imgsz=a.imgsz,
                             lr=a.lr, lr_pre=a.lr_pretrained, warmup=a.warmup_iters, seed=a.seed, fraction=a.fraction,
                             log=log, max_iters=a.max_iters, meta=meta)
    engine.save_checkpoint(out / "trained.pt", model, {**meta, "final": True, "epoch": st["epoch"]})
    final = out / "trained.pt"
    if splice:
        record = engine.calibrate_model(model, data, batches=a.calib_batches, batch=a.batch, workers=a.workers, imgsz=a.imgsz, log=log)
        engine.save_checkpoint(out / "calibrated.pt", model, {**meta, "final": True, "epoch": st["epoch"], "calibration": record})
        final = out / "calibrated.pt"
        if a.arm == "e" and a.qat_epochs > 0:
            summary = qat.attach(model, record.get("act_lsb"))
            log({"stage": "qat_attach", **summary})
            engine.train_epochs(model, data, out / "last_qat.pt", epochs=a.qat_epochs, batch=a.batch, workers=a.workers,
                                imgsz=a.imgsz, lr=a.lr * 0.1, lr_pre=a.lr_pretrained * 0.1, warmup=100, seed=a.seed,
                                fraction=a.fraction, log=log, max_iters=a.max_iters, tag="qat", use_ema=False,
                                close_mosaic=a.qat_epochs, meta=meta)
            qat.detach(model)
            engine.save_checkpoint(out / "qat.pt", model, {**meta, "final": True, "epoch": st["epoch"], "calibration": record,
                                                           "qat": summary})
            final = out / "qat.pt"
        rep = export.export_torch(str(final), str(out / "export"), run=out.name)
        log({"stage": "export", "checkpoint": str(final), **rep})
        if model.neck._pool is not None:
            model.neck._pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
