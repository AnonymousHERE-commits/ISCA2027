"""Datasets of the SPLICE experiments (SPLICE-V section 4): COCO 2017 and CrowdHuman.

The raw downloads are left untouched.  `prepare_coco` / `prepare_crowdhuman` build a *view*
per dataset in the Ultralytics layout,

    <views>/<name>/images/{train_fit,calib,val}/<image>     links to the raw images (hard links where the
                                                            file system allows, else symbolic)
    <views>/<name>/labels/{train_fit,calib,val}/<image>.txt YOLO labels (class cx cy w h, normalised)
    <views>/<name>/gt_val.json                              CrowdHuman: evaluation ground truth
    <views>/<name>/dataset.json                             paths, names, counts, source hashes

with three disjoint roles:
    train_fit  training images (the training split minus the calibration split)
    calib      calibration split, held out of training by a hash of the file name: band
               exponents, conformal thresholds, partner prior, activation scales, the
               sensitivity and leakage measurements (never the validation set)
    val        the validation set (COCO val2017: 5,000 images; CrowdHuman val: 4,370)

CrowdHuman: full-body boxes ('fbox') of 'person' entries are class 0.  Entries with
tag != person or extra.ignore = 1 are ignore regions: they are left out of the training
labels and kept, flagged, in gt_val.json.  YOLO labels cannot leave the image, so boxes are
clipped to it; gt_val.json keeps the raw and the clipped full-body box and the visible box
(visible ratio = area(vbox) / area(fbox); "heavily occluded" is a ratio below 0.65).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IMG_EXT = (".jpg", ".jpeg", ".png")
CALIB_MOD = {"coco": 24, "crowdhuman": 12}        # 1 / 24 of train2017 (about 4,900), 1 / 12 of CrowdHuman train (1,250)


class DataError(RuntimeError):
    """A dataset is missing or incomplete (the message says what was searched)."""


def is_calibration(name: str, mod: int) -> bool:
    """Deterministic calibration membership of an image (hash of its file stem)."""
    return int(hashlib.sha1(Path(name).stem.encode()).hexdigest()[:8], 16) % mod == 0


def _find_dirs(root: Path, name: str, depth: int = 3) -> List[Path]:
    out = []
    if not root.is_dir():
        return out
    stack = [(root, 0)]
    while stack:
        d, k = stack.pop()
        try:
            subs = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except OSError:
            continue
        for p in subs:
            if p.name == name:
                out.append(p)
            elif k + 1 < depth:
                stack.append((p, k + 1))
    return sorted(out)


def _find_file(root: Path, name: str, depth: int = 4) -> Optional[Path]:
    if not root.is_dir():
        return None
    stack = [(root, 0)]
    while stack:
        d, k = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            if p.is_file() and p.name == name:
                return p
            if p.is_dir() and not p.name.startswith(".") and k + 1 < depth:
                stack.append((p, k + 1))
    return None


def _images_in(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def _link(src: Path, dst: Path) -> str:
    """Make `dst` the image `src` without copying it: a hard link where the file system allows
    one (it adds no file to the storage project's file-count quota), a symbolic link otherwise
    (other file system, or SPLICE_VIEW_LINKS=symbolic).  Returns 'hard' or 'symbolic'."""
    try:
        if os.path.samefile(src, dst):                    # already this image
            return "symbolic" if dst.is_symlink() else "hard"
    except OSError:
        pass
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if os.environ.get("SPLICE_VIEW_LINKS", "hard") != "symbolic":
        try:
            os.link(src, dst)
            return "hard"
        except OSError:
            pass
    os.symlink(str(src), str(dst))
    return "symbolic"


def _sha256(path: Path, limit: int = 1 << 62) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        while n < limit:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest()


def _list_hash(names: Iterable[str]) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode() + b"\n")
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


# ----------------------------------------------------------------------------- COCO

def discover_coco(root: Path) -> Dict[str, Path]:
    """Locate train2017 / val2017 image folders and the instances annotations under `root`."""
    root = Path(root)
    found: Dict[str, Path] = {}
    for split in ("train2017", "val2017"):
        cands = [d for d in _find_dirs(root, split) if d.parent.name != "labels" and any(
            p.suffix.lower() in IMG_EXT for p in list(d.iterdir())[:50])]
        if cands:
            found[split] = cands[0]
        ann = _find_file(root, f"instances_{split}.json")
        if ann:
            found[f"ann_{split}"] = ann
        labs = [d for d in _find_dirs(root, split) if d.parent.name == "labels"]
        if labs:
            found[f"labels_{split}"] = labs[0]
    missing = [k for k in ("train2017", "val2017", "ann_val2017") if k not in found]
    if "labels_train2017" not in found and "ann_train2017" not in found:
        missing.append("ann_train2017 (or YOLO labels/train2017)")
    if missing:
        raise DataError(
            f"COCO 2017 incomplete under {root}: missing {missing}. Expected the folders train2017/ and val2017/ with "
            f"the images (directly or below images/) and annotations/instances_train2017.json, instances_val2017.json. "
            f"Run detector/hpc/download_datasets.sh coco, or set SPLICE_COCO_ROOT.")
    return found


def _coco_label_rows(ann_file: Path) -> Tuple[Dict[str, List[str]], List[int], Dict[int, str]]:
    """YOLO label rows per image file name from an instances json (iscrowd excluded)."""
    js = json.loads(Path(ann_file).read_text())
    cats = sorted(c["id"] for c in js["categories"])
    names = {i: next(c["name"] for c in js["categories"] if c["id"] == cid) for i, cid in enumerate(cats)}
    cidx = {cid: i for i, cid in enumerate(cats)}
    imgs = {im["id"]: im for im in js["images"]}
    rows: Dict[str, List[str]] = {im["file_name"]: [] for im in js["images"]}
    for a in js["annotations"]:
        if a.get("iscrowd", 0):
            continue
        im = imgs[a["image_id"]]
        W, H = float(im["width"]), float(im["height"])
        x, y, w, h = a["bbox"]
        x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(W, x + w), min(H, y + h)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        rows[im["file_name"]].append(f"{cidx[a['category_id']]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return rows, cats, names


def prepare_coco(root: Path, views: Path, log=print) -> Dict:
    """Build the COCO view.  Idempotent: a complete view with matching counts is reused."""
    src = discover_coco(Path(root))
    out = Path(views) / "coco"
    marker = out / "dataset.json"
    train_imgs = _images_in(src["train2017"])
    val_imgs = _images_in(src["val2017"])
    key = {"train_dir": str(src["train2017"]), "val_dir": str(src["val2017"]), "n_train": len(train_imgs),
           "n_val": len(val_imgs), "calib_mod": CALIB_MOD["coco"]}
    if marker.exists():
        old = json.loads(marker.read_text())
        if all(old.get("source", {}).get(k) == v for k, v in key.items()):
            log(f"[DATA] COCO view is up to date: {out}")
            return old
    log(f"[DATA] building the COCO view under {out} ({len(train_imgs)} train, {len(val_imgs)} val images)")
    roles = {"train_fit": [], "calib": [], "val": val_imgs}
    for p in train_imgs:
        roles["calib" if is_calibration(p.name, CALIB_MOD["coco"]) else "train_fit"].append(p)
    label_src = {}
    names: Dict[int, str] = {}
    cats: List[int] = []
    for split in ("train2017", "val2017"):
        if f"ann_{split}" in src:
            label_src[split], cats, names = _coco_label_rows(src[f"ann_{split}"])
        else:
            label_src[split] = None
    if not names:
        raise DataError("no COCO instances json found to read the category names")
    counts, links = {}, {"hard": 0, "symbolic": 0}
    for role, imgs in roles.items():
        idir, ldir = out / "images" / role, out / "labels" / role
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        split = "val2017" if role == "val" else "train2017"
        rows = label_src[split]
        n_lab = 0
        for p in imgs:
            links[_link(p, idir / p.name)] += 1
            dst = ldir / (p.stem + ".txt")
            if rows is not None:
                r = rows.get(p.name, [])
                dst.write_text("\n".join(r) + ("\n" if r else ""))
                n_lab += len(r)
            else:                                           # reuse existing YOLO labels
                lab = src[f"labels_{split}"] / (p.stem + ".txt")
                text = lab.read_text() if lab.exists() else ""
                dst.write_text(text)
                n_lab += text.count("\n")
        counts[role] = {"images": len(imgs), "boxes": n_lab, "list_sha256": _list_hash(sorted(p.name for p in imgs))}
        log(f"[DATA] coco/{role}: {len(imgs)} images, {n_lab} boxes")
    rec = {"name": "coco", "path": str(out), "train": "images/train_fit", "calib": "images/calib", "val": "images/val",
           "names": {str(k): v for k, v in names.items()}, "nc": len(names), "category_ids": cats,
           "coco_json": str(src["ann_val2017"]), "coco_json_sha256": _sha256(src["ann_val2017"]),
           "counts": counts, "links": links, "source": key}
    _write_json(marker, rec)
    return rec


# ----------------------------------------------------------------------------- CrowdHuman

def discover_crowdhuman(root: Path) -> Dict:
    root = Path(root)
    found = {}
    for split in ("train", "val"):
        f = _find_file(root, f"annotation_{split}.odgt")
        if f:
            found[split] = f
    if "train" not in found or "val" not in found:
        raise DataError(
            f"CrowdHuman incomplete under {root}: annotation_train.odgt / annotation_val.odgt not found. Expected the "
            f"odgt files and the extracted images (any folder below it, e.g. Images/). Run "
            f"detector/hpc/download_datasets.sh crowdhuman, or set SPLICE_CROWDHUMAN_ROOT.")
    index: Dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("splice_views",)]
        for fn in filenames:
            if fn.lower().endswith(IMG_EXT):
                index.setdefault(Path(fn).stem, Path(dirpath) / fn)
    if not index:
        raise DataError(f"CrowdHuman: no images found under {root} (extract CrowdHuman_train01-03.zip and CrowdHuman_val.zip)")
    found["index"] = index
    return found


def _image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def crowdhuman_records(odgt: Path, index: Dict[str, Path]) -> List[Dict]:
    """One record per annotated image that exists on disk."""
    out = []
    for line in Path(odgt).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        src = index.get(rec["ID"])
        if src is None:
            continue
        W, H = _image_size(src)
        boxes = []
        for g in rec.get("gtboxes", []):
            fx, fy, fw, fh = g["fbox"]
            ign = int(g.get("tag") != "person" or g.get("extra", {}).get("ignore", 0) == 1)
            raw = [float(fx), float(fy), float(fx + fw), float(fy + fh)]
            clip = [max(0.0, raw[0]), max(0.0, raw[1]), min(float(W), raw[2]), min(float(H), raw[3])]
            vis = 1.0
            if "vbox" in g and fw > 0 and fh > 0:
                vis = max(0.0, min(1.0, (g["vbox"][2] * g["vbox"][3]) / float(fw * fh)))
            boxes.append({"fbox": raw, "fbox_clip": clip, "ignore": ign, "vis": round(vis, 6)})
        out.append({"id": rec["ID"], "file": str(src), "width": W, "height": H, "boxes": boxes})
    return out


def prepare_crowdhuman(root: Path, views: Path, log=print) -> Dict:
    src = discover_crowdhuman(Path(root))
    out = Path(views) / "crowdhuman"
    marker = out / "dataset.json"
    key = {"odgt_train_sha256": _sha256(src["train"]), "odgt_val_sha256": _sha256(src["val"]),
           "n_indexed": len(src["index"]), "calib_mod": CALIB_MOD["crowdhuman"]}
    if marker.exists():
        old = json.loads(marker.read_text())
        if all(old.get("source", {}).get(k) == v for k, v in key.items()):
            log(f"[DATA] CrowdHuman view is up to date: {out}")
            return old
    log(f"[DATA] building the CrowdHuman view under {out} ({len(src['index'])} images indexed)")
    train = crowdhuman_records(src["train"], src["index"])
    val = crowdhuman_records(src["val"], src["index"])
    if not train or not val:
        raise DataError(f"CrowdHuman: annotations found but no matching images (train {len(train)}, val {len(val)})")
    roles = {"train_fit": [r for r in train if not is_calibration(r["id"], CALIB_MOD["crowdhuman"])],
             "calib": [r for r in train if is_calibration(r["id"], CALIB_MOD["crowdhuman"])], "val": val}
    counts, links = {}, {"hard": 0, "symbolic": 0}
    for role, recs in roles.items():
        idir, ldir = out / "images" / role, out / "labels" / role
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        n_lab = n_ign = 0
        for r in recs:
            p = Path(r["file"])
            links[_link(p, idir / (r["id"] + p.suffix.lower()))] += 1
            rows = []
            W, H = r["width"], r["height"]
            for b in r["boxes"]:
                if b["ignore"]:
                    n_ign += 1
                    continue
                x0, y0, x1, y1 = b["fbox_clip"]
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                rows.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            (ldir / (r["id"] + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
            n_lab += len(rows)
        counts[role] = {"images": len(recs), "boxes": n_lab, "ignore_regions": n_ign,
                        "list_sha256": _list_hash(sorted(r["id"] for r in recs))}
        log(f"[DATA] crowdhuman/{role}: {len(recs)} images, {n_lab} person boxes, {n_ign} ignore regions")
    for role in ("val", "calib"):
        _write_json(out / f"gt_{role}.json", {r["id"]: {"width": r["width"], "height": r["height"], "boxes": r["boxes"]}
                                              for r in roles[role]})
    rec = {"name": "crowdhuman", "path": str(out), "train": "images/train_fit", "calib": "images/calib",
           "val": "images/val", "names": {"0": "person"}, "nc": 1, "gt_val": str(out / "gt_val.json"),
           "gt_calib": str(out / "gt_calib.json"), "counts": counts, "links": links, "source": key}
    _write_json(marker, rec)
    return rec


# ----------------------------------------------------------------------------- loaders

def load_dataset(record_or_path) -> Dict:
    """The dict the Ultralytics dataset builders expect (no download or font checks)."""
    rec = record_or_path if isinstance(record_or_path, dict) else json.loads(Path(record_or_path).read_text())
    root = Path(rec["path"])
    data = dict(rec)
    data["names"] = {int(k): v for k, v in rec["names"].items()}
    data["nc"] = int(rec["nc"])
    data["channels"] = 3
    for k in ("train", "calib", "val"):
        data[k] = str(root / rec[k])
    return data


def make_loader(data: Dict, split: str, mode: str, batch: int, workers: int, imgsz: int = 640, fraction: float = 1.0,
                shuffle: Optional[bool] = None):
    """Ultralytics dataset + loader for one role.  mode 'train': mosaic and augmentation of
    the default recipe; mode 'val': letterbox to imgsz, no augmentation."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG
    cfg = get_cfg(DEFAULT_CFG, {"imgsz": imgsz, "batch": batch, "fraction": fraction if mode == "train" else 1.0,
                                "rect": False, "cache": False, "task": "detect"})
    ds = build_yolo_dataset(cfg, data[split], batch, data, mode=mode, rect=False, stride=32)
    return build_dataloader(ds, batch, workers, shuffle=(mode == "train") if shuffle is None else shuffle, rank=-1)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build the dataset views (COCO 2017, CrowdHuman) for the SPLICE experiments.")
    ap.add_argument("--coco-root", help="folder that holds train2017/, val2017/ and annotations/")
    ap.add_argument("--crowdhuman-root", help="folder that holds annotation_train.odgt, annotation_val.odgt and the images")
    ap.add_argument("--views", required=True, help="output folder of the views")
    a = ap.parse_args()
    if a.coco_root:
        print(json.dumps(prepare_coco(Path(a.coco_root), Path(a.views))["counts"], indent=1))
    if a.crowdhuman_root:
        print(json.dumps(prepare_crowdhuman(Path(a.crowdhuman_root), Path(a.views))["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
