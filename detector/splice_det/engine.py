"""Training and inference engine (PyTorch + Ultralytics) used by the Arrhenius suite and by
the command-line tools.

  build_model        SPLICE-B in a readout arm, or the native YOLO26-S (baseline rows)
  train_epochs       resumable training (or QAT fine-tuning): SGD, warm-up + cosine, EMA,
                     bf16 autocast outside the neck; checkpoints every epoch; stops cleanly
                     before a wall-clock deadline so that a requeued job continues
  calibrate_model    calibration split: band exponents, conformal thresholds, partner prior,
                     activation scales, measured sensitivity
  collect_detections FP32 inference over a loader, detections in original image coordinates,
                     with and without the objectness gate, plus per-frame regime statistics
"""
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import ARMS, SpliceConfig


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(weights: str, device, arm: Optional[str] = "e", descriptor: str = "bands", link_workers: int = 0,
                imgsz: int = 640, nc: Optional[int] = None, cfg: Optional[SpliceConfig] = None):
    """YOLO26-S from `weights` with the SPLICE neck in readout arm `arm`; arm None or 'native'
    keeps the native neck.  `nc` rebuilds the class outputs when the dataset's class count
    differs from the checkpoint's (CrowdHuman: one class)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    from .yolo26 import NativeYOLO, SpliceYOLO, load_yolo
    yolo = load_yolo(weights)
    if nc is not None and int(getattr(yolo.model[-1], "nc", nc)) != int(nc):
        yolo = _rebuild_classes(yolo, int(nc))
    yolo.args = get_cfg(DEFAULT_CFG)                     # loss gains (box, cls, dfl) of the default recipe
    for p in yolo.parameters():
        p.requires_grad_(True)
    yolo.to(device)
    if arm in (None, "native"):
        return NativeYOLO(yolo).to(device)
    cfg = cfg or SpliceConfig(arm=arm, descriptor=descriptor)
    return SpliceYOLO(yolo, cfg, workers=link_workers, imgsz=imgsz).to(device)


def _rebuild_classes(yolo, nc: int):
    """A detection model with `nc` classes that keeps every weight whose shape still fits
    (backbone, neck, box branch); the class outputs start fresh."""
    from copy import deepcopy
    from ultralytics.nn.tasks import DetectionModel
    cfg = deepcopy(yolo.yaml)
    cfg["nc"] = nc
    new = DetectionModel(cfg, ch=cfg.get("channels", cfg.get("ch", 3)), nc=nc, verbose=False)
    old_sd, new_sd = yolo.state_dict(), new.state_dict()
    keep = {k: v for k, v in old_sd.items() if k in new_sd and tuple(v.shape) == tuple(new_sd[k].shape)}
    new.load_state_dict(keep, strict=False)
    new.names = {i: str(i) for i in range(nc)}
    new.transferred = {"kept": len(keep), "fresh": len(new_sd) - len(keep)}
    return new


def model_config(model) -> Optional[Dict]:
    return model.neck.cfg.to_dict() if getattr(model, "neck", None) is not None else None


def save_checkpoint(path: Path, model, meta: Dict, ema=None, opt=None) -> None:
    import torch
    st = dict(meta)
    st["model"] = model.state_dict()
    st["cfg"] = model_config(model)
    if ema is not None:
        st["ema"] = ema.ema.state_dict()
        st["ema_updates"] = int(ema.updates)
    if opt is not None:
        st["opt"] = opt.state_dict()
    tmp = Path(str(path) + ".tmp")
    torch.save(st, tmp)
    os.replace(tmp, path)


def load_model(ckpt: str, device, link_workers: int = 0, imgsz: int = 640, weights: Optional[str] = None,
               use_ema: bool = True):
    """Rebuild the model of a checkpoint written by this package.  Returns (model, checkpoint)."""
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = SpliceConfig.from_dict(ck["cfg"]) if ck.get("cfg") else None
    model = build_model(weights or ck["weights"], device, cfg.arm if cfg else "native", cfg.descriptor if cfg else "bands",
                        link_workers, imgsz, ck.get("nc"), cfg)
    state = ck["ema"] if (use_ema and "ema" in ck and not ck.get("final", False)) else ck["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "parametrizations" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint {ckpt} does not fit the model: missing {real_missing[:8]}, unexpected {unexpected[:8]}")
    cal = ck.get("calibration") or {}
    if cfg is not None and cal.get("gate_tau"):
        with torch.no_grad():
            model.neck.gate_tau.copy_(torch.tensor(cal["gate_tau"], dtype=torch.float32))
            model.neck.prior.copy_(torch.tensor(cal["prior"], dtype=torch.float32))
            model.neck.band_exp.copy_(torch.tensor(cal["band_exponents"], dtype=torch.float32))
    return model, ck


def optimizer(model, lr: float, lr_pre: float):
    """SGD with four groups: the new neck at `lr`, the pretrained backbone and head at
    `lr_pre`; no weight decay on biases and normalisation weights."""
    import torch
    neck, pre_decay, pre_nodecay, neck_nodecay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nd = p.ndim <= 1 or n.endswith(".bias")
        if n.startswith("neck."):
            (neck_nodecay if nd else neck).append(p)
        else:
            (pre_nodecay if nd else pre_decay).append(p)
    groups = [{"params": neck, "lr": lr, "weight_decay": 5e-4, "base_lr": lr},
              {"params": neck_nodecay, "lr": lr, "weight_decay": 0.0, "base_lr": lr},
              {"params": pre_decay, "lr": lr_pre, "weight_decay": 5e-4, "base_lr": lr_pre},
              {"params": pre_nodecay, "lr": lr_pre, "weight_decay": 0.0, "base_lr": lr_pre}]
    return torch.optim.SGD([g for g in groups if g["params"]], momentum=0.937, nesterov=True)


def set_lr(opt, it: int, total: int, warmup: int) -> None:
    f = min(1.0, (it + 1) / max(1, warmup)) * (0.5 * (1 + math.cos(math.pi * min(1.0, it / max(1, total)))) * 0.99 + 0.01)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * f


def to_device(batch: Dict, device) -> Dict:
    import torch
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _close_mosaic(loader) -> bool:
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    ds = getattr(loader, "dataset", None)
    if ds is None or not hasattr(ds, "close_mosaic"):
        return False
    ds.close_mosaic(hyp=get_cfg(DEFAULT_CFG))
    if hasattr(loader, "reset"):
        loader.reset()
    return True


def train_epochs(model, data: Dict, ck_path: Path, *, epochs: int, batch: int, workers: int, imgsz: int = 640,
                 lr: float = 0.01, lr_pre: float = 0.001, warmup: int = 1000, seed: int = 0, fraction: float = 1.0,
                 log: Callable[[Dict], None] = print, deadline: Optional[float] = None, max_iters: Optional[int] = None,
                 tag: str = "train", use_ema: bool = True, close_mosaic: int = 10, log_every: int = 50,
                 meta: Optional[Dict] = None) -> Dict:
    """Train `model` for `epochs` epochs, resuming from `ck_path` when it exists.

    Returns {'status': 'DONE' | 'CONTINUE', ...}.  'CONTINUE' means the wall-clock `deadline`
    (unix time) would be passed by another epoch: the state up to the last finished epoch is
    in `ck_path` and the same call continues from there."""
    import torch
    from ultralytics.utils.torch_utils import ModelEMA
    from . import losses
    from .data import make_loader
    device = next(model.parameters()).device
    splice = getattr(model, "neck", None) is not None
    set_seed(seed)
    loader = make_loader(data, "train", "train", batch, workers, imgsz, fraction)
    per_epoch = min(len(loader), max_iters) if max_iters else len(loader)
    total = epochs * per_epoch
    opt = optimizer(model, lr, lr_pre)
    if splice:
        model.neck._pool = None                     # the EMA copy must not share the process pool
    ema = ModelEMA(model) if use_ema else None
    start_ep, it, ep_seconds, mosaic_closed = 0, 0, None, False
    meta = dict(meta or {})
    if Path(ck_path).exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if ema is not None and "ema" in ck:
            ema.ema.load_state_dict(ck["ema"])
            ema.updates = ck.get("ema_updates", 0)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_ep, it, ep_seconds = ck["epoch"] + 1, ck["it"], ck.get("epoch_seconds")
        log({"stage": tag, "event": "resume", "from_epoch": start_ep, "it": it, "checkpoint": str(ck_path)})
    strides = model.neck.strides if splice else None
    cfg = model.neck.cfg if splice else None
    for ep in range(start_ep, epochs):
        if deadline is not None and ep_seconds is not None and time.time() + 1.15 * ep_seconds + 300 > deadline:
            return {"status": "CONTINUE", "epoch": ep - 1, "epochs": epochs, "it": it, "epoch_seconds": ep_seconds}
        if close_mosaic and not mosaic_closed and ep >= epochs - close_mosaic:
            mosaic_closed = _close_mosaic(loader) or True
            log({"stage": tag, "event": "close_mosaic", "epoch": ep})
        model.train()
        t0 = time.time()
        seen = 0
        for i, batch_ in enumerate(loader):
            if max_iters is not None and i >= max_iters:
                break
            if deadline is not None and time.time() > deadline - 180:
                return {"status": "CONTINUE", "epoch": ep - 1, "epochs": epochs, "it": it - seen, "epoch_seconds": ep_seconds,
                        "note": "deadline inside an epoch; that epoch restarts"}
            set_lr(opt, it, total, warmup)
            img = batch_["img"].to(device, non_blocking=True).float() / 255.0
            boxes = losses.batch_boxes(batch_) if splice else None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw, nk = model(img, boxes, probe=True)
                det, items = model.detection_loss(raw, to_device(batch_, device))
            det = det.sum().float()
            if splice:
                aux = losses.aux_losses(nk, boxes, strides, cfg.hyp_weight)
                loss, extra = losses.total_loss(det, nk, aux, cfg, img.shape[0])
            else:
                loss, extra = det, {}
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep} iteration {i}: det={float(det)} extra={extra}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            if ema is not None:
                ema.update(model)
            if i % log_every == 0:
                rec = {"stage": tag, "epoch": ep, "iter": i, "it": it, "loss": float(loss), "det": float(det),
                       "det_items": [float(x) for x in torch.as_tensor(items).flatten()[:4]], **extra,
                       "lr": opt.param_groups[0]["lr"], "s_per_it": (time.time() - t0) / (i + 1)}
                if splice:
                    rec.update(F_mean=float(np.mean([r["F"] for r in nk["frag"]])), rows=nk["fc_rows"])
                log(rec)
            it += 1
            seen += 1
        ep_seconds = time.time() - t0
        save_checkpoint(Path(ck_path), model, {**meta, "epoch": ep, "it": it, "epoch_seconds": ep_seconds, "epochs": epochs,
                                               "tag": tag}, ema, opt)
        log({"stage": tag, "event": "epoch_end", "epoch": ep, "seconds": ep_seconds, "iters": seen})
    if ema is not None:
        model.load_state_dict(ema.ema.state_dict())
    return {"status": "DONE", "epoch": epochs - 1, "epochs": epochs, "it": it, "epoch_seconds": ep_seconds}


def calibrate_model(model, data: Dict, *, split: str = "calib", batches: int = 200, batch: int = 16, workers: int = 8,
                    imgsz: int = 640, log: Callable[[Dict], None] = print) -> Dict:
    """Calibration on the calibration split (never the validation set); writes the
    calibrated values into the model buffers and returns the record."""
    from . import calibrate
    from .data import make_loader
    loader = make_loader(data, split, "val", batch, workers, imgsz, shuffle=False)
    stats = calibrate.collect(model, loader, batches, model.neck.strides)
    bands = calibrate.band_statistics(model, loader, min(batches, 50))
    record = calibrate.apply(model, stats, bands, model.neck.cfg.gate_delta, len(model.neck.cfg.gate_tau))
    record["sensitivity"] = calibrate.sensitivity(model, loader, min(batches, 50))
    record["split"] = split
    record["images"] = int(min(batches * batch, len(loader.dataset)))
    log({"stage": "calibrate", **{k: v for k, v in record.items() if k != "act_lsb"}})
    return record


def _nms():
    try:                                                   # current releases
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:                                    # older releases
        from ultralytics.utils.ops import non_max_suppression
    return non_max_suppression


def frame_statistics(nk: Dict, theta: float) -> List[Dict]:
    """Regime variables of every frame of a batch (Table 5, channel 'train'): fragment count,
    cap and budget exits, linkage rounds, candidates per row, and the share of class-head
    blocks whose cells all have gamma < theta (the skip rule) for block sides 1, 2 and 4."""
    import torch
    import torch.nn.functional as F
    out = []
    gates = nk["gates"]
    skip = {}
    for b in (1, 2, 4):
        num = torch.zeros(gates[0].shape[0], device=gates[0].device)
        den = 0
        for g in gates:
            m = F.max_pool2d(g.float(), b, b, ceil_mode=True) if b > 1 else g.float()
            num = num + (m < theta).float().flatten(1).sum(1)
            den += m[0].numel()
        skip[b] = (num / den).cpu().numpy()
    for n, rec in enumerate(nk["frag"]):
        lk = rec["link"]
        row = {"F": int(rec["F"]), "budget_hit": bool(lk["budget_hit"]), "overflow": bool(lk["overflow"]),
               "rounds": int(lk["rounds"]), "work": int(lk["work"]), "skip1": float(skip[1][n]), "skip2": float(skip[2][n]),
               "skip4": float(skip[4][n]), "partners": 0.0, "partners_max": 0}
        if rec["F"] and "adm" in rec:
            cnt = rec["adm"].sum(1).float()
            row["partners"] = float(cnt.mean())
            row["partners_max"] = int(cnt.max())
        out.append(row)
    return out


def collect_detections(model, loader, device, limit: Optional[int] = None, theta: float = 0.25,
                       conf: float = 0.001, max_det: int = 300) -> Dict:
    """FP32 inference.  Returns {'files', 'dets': {variant: [per image (n, 6)]}, 'frames'} with
    boxes [x0, y0, x1, y1, score, class] in original image coordinates.  Variants: 'gate' and
    'nogate' for SPLICE models (class score with and without sigma(l) gamma), 'native'."""
    import torch
    from ultralytics.utils import ops as uops
    nms = _nms()
    splice = getattr(model, "neck", None) is not None
    out = {"files": [], "dets": {}, "frames": []}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device).float() / 255.0
            if splice:
                ys, nk = model(img, gate="both")
                out["frames"].extend(frame_statistics(nk, theta))
            else:
                y, _ = model(img)
                ys = {"native": y}
            h, w = img.shape[-2:]
            for key, y in ys.items():
                if y.ndim == 3 and y.shape[-1] == 6:                          # end-to-end: (N, max_det, 6)
                    preds = [p[p[:, 4] > conf] for p in y]
                else:                                                          # (N, 4 + nc, A): NMS
                    preds = nms(y, conf, 0.7, max_det=max_det)
                store = out["dets"].setdefault(key, [])
                for n, p in enumerate(preds):
                    p = p.detach().float().cpu()
                    if len(p):
                        p[:, :4] = uops.scale_boxes((h, w), p[:, :4].clone(), batch["ori_shape"][n], ratio_pad=batch["ratio_pad"][n])
                    store.append(p.numpy()[:, :6].astype(np.float64))
            out["files"].extend(batch["im_file"])
            if limit is not None and len(out["files"]) >= limit:
                break
    if limit is not None:
        out["files"] = out["files"][:limit]
        out["dets"] = {k: v[:limit] for k, v in out["dets"].items()}
        out["frames"] = out["frames"][:limit]
    return out
