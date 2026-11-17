"""Dataset views (detector/splice_det/data.py) on tiny synthetic COCO and CrowdHuman trees:
discovery of the raw layouts, disjoint roles, labels, ignore regions, visible ratios and the
idempotent rebuild."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "detector"))

from splice_det import data as D  # noqa: E402

try:
    from PIL import Image
except ImportError:                                   # pragma: no cover
    Image = None


def make_coco(root: Path, n_train=60, n_val=6, nested=True):
    cats = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 90, "name": "toothbrush"}]
    for split, n in (("train2017", n_train), ("val2017", n_val)):
        d = (root / "images" / split) if nested else (root / split)
        d.mkdir(parents=True)
        images, anns = [], []
        for i in range(n):
            iid = 1000 * (split == "val2017") + i + 1
            name = f"{iid:012d}.jpg"
            Image.new("RGB", (64, 48), (i % 255, 10, 20)).save(d / name)
            images.append({"id": iid, "file_name": name, "width": 64, "height": 48})
            anns.append({"id": len(anns) + 1 + 10000 * (split == "val2017"), "image_id": iid, "category_id": [1, 3, 90][i % 3],
                         "bbox": [4, 6, 20, 30], "iscrowd": 0, "area": 600})
            if i % 4 == 0:                             # a crowd region and a box leaving the image
                anns.append({"id": len(anns) + 1 + 10000 * (split == "val2017"), "image_id": iid, "category_id": 1,
                             "bbox": [30, 0, 30, 40], "iscrowd": 1, "area": 1200})
                anns.append({"id": len(anns) + 1 + 10000 * (split == "val2017"), "image_id": iid, "category_id": 3,
                             "bbox": [50, 30, 40, 40], "iscrowd": 0, "area": 1600})
        (root / "annotations").mkdir(exist_ok=True)
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(
            {"images": images, "annotations": anns, "categories": cats}))


def make_crowdhuman(root: Path, n_train=48, n_val=5):
    (root / "CrowdHuman_train01" / "Images").mkdir(parents=True)
    (root / "val" / "Images").mkdir(parents=True)
    for split, n, d in (("train", n_train, root / "CrowdHuman_train01" / "Images"), ("val", n_val, root / "val" / "Images")):
        lines = []
        for i in range(n):
            ID = f"{split}_{i:04d},abc"
            Image.new("RGB", (80, 60), (5, i % 255, 9)).save(d / f"{ID}.jpg")
            gt = [{"tag": "person", "fbox": [10, 5, 20, 40], "vbox": [10, 5, 20, 40], "hbox": [12, 5, 8, 8], "extra": {}},
                  {"tag": "person", "fbox": [-6, 10, 30, 60], "vbox": [0, 10, 12, 30], "hbox": [0, 10, 6, 6], "extra": {"ignore": 0}},
                  {"tag": "mask", "fbox": [50, 20, 25, 30], "vbox": [50, 20, 25, 30], "hbox": [50, 20, 5, 5], "extra": {"ignore": 1}}]
            lines.append(json.dumps({"ID": ID, "gtboxes": gt}))
        (root / f"annotation_{split}.odgt").write_text("\n".join(lines) + "\n")


@unittest.skipIf(Image is None, "Pillow not installed")
class Views(unittest.TestCase):
    def test_coco_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, views = Path(tmp) / "coco", Path(tmp) / "views"
            make_coco(root)
            rec = D.prepare_coco(root, views, log=lambda *_: None)
            c = rec["counts"]
            self.assertEqual(c["train_fit"]["images"] + c["calib"]["images"], 60)
            self.assertGreater(c["calib"]["images"], 0)
            self.assertEqual(c["val"]["images"], 6)
            self.assertEqual(rec["category_ids"], [1, 3, 90])
            self.assertEqual(rec["names"], {"0": "person", "1": "car", "2": "toothbrush"})
            names = {r: {p.name for p in (views / "coco" / "images" / r).iterdir()} for r in ("train_fit", "calib", "val")}
            self.assertFalse(names["train_fit"] & names["calib"])
            self.assertTrue(all(D.is_calibration(n, D.CALIB_MOD["coco"]) for n in names["calib"]))
            img = sorted((views / "coco" / "images" / "val").iterdir())[0]
            raw = root / "images" / "val2017" / img.name
            self.assertTrue(os.path.samefile(img, raw))             # a link to the raw image, never a copy
            self.assertEqual(sum(rec["links"].values()), 66)
            rows = (views / "coco" / "labels" / "val" / (img.stem + ".txt")).read_text().split("\n")
            first = rows[0].split()
            self.assertEqual(first[0], "0")                                    # category 1 -> class 0
            self.assertAlmostEqual(float(first[1]), (4 + 10) / 64, places=5)
            self.assertAlmostEqual(float(first[4]), 30 / 48, places=5)
            self.assertEqual(len([r for r in rows if r]), 2)                   # the crowd region is not a label
            clipped = rows[1].split()                                          # box [50, 30, 90, 70] clipped to the image
            self.assertEqual(clipped[0], "1")
            self.assertAlmostEqual(float(clipped[3]), 14 / 64, places=5)
            data = D.load_dataset(views / "coco" / "dataset.json")
            self.assertEqual(data["nc"], 3)
            self.assertTrue(Path(data["train"]).is_dir() and Path(data["calib"]).is_dir() and Path(data["val"]).is_dir())
            self.assertTrue(Path(data["coco_json"]).exists())
            again = D.prepare_coco(root, views, log=lambda *_: None)          # idempotent
            self.assertEqual(again["counts"], rec["counts"])

    def test_coco_flat_layout_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "coco"
            make_coco(root, n_train=30, n_val=3, nested=False)
            rec = D.prepare_coco(root, Path(tmp) / "views", log=lambda *_: None)
            self.assertEqual(rec["counts"]["val"]["images"], 3)
            with self.assertRaises(D.DataError) as cm:
                D.discover_coco(Path(tmp) / "nothing")
            self.assertIn("train2017", str(cm.exception))

    def test_crowdhuman_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, views = Path(tmp) / "crowdhuman", Path(tmp) / "views"
            make_crowdhuman(root)
            rec = D.prepare_crowdhuman(root, views, log=lambda *_: None)
            c = rec["counts"]
            self.assertEqual(c["train_fit"]["images"] + c["calib"]["images"], 48)
            self.assertEqual(c["val"]["images"], 5)
            self.assertEqual(c["val"]["boxes"], 10)                            # two persons per image
            self.assertEqual(c["val"]["ignore_regions"], 5)
            gt = json.loads((views / "crowdhuman" / "gt_val.json").read_text())
            one = gt[sorted(gt)[0]]
            self.assertEqual([b["ignore"] for b in one["boxes"]], [0, 0, 1])
            self.assertEqual(one["boxes"][1]["fbox"], [-6.0, 10.0, 24.0, 70.0])          # raw box leaves the image
            self.assertEqual(one["boxes"][1]["fbox_clip"], [0.0, 10.0, 24.0, 60.0])
            self.assertAlmostEqual(one["boxes"][1]["vis"], 12 * 30 / (30 * 60), places=6)   # heavily occluded
            self.assertAlmostEqual(one["boxes"][0]["vis"], 1.0, places=6)
            lab = (views / "crowdhuman" / "labels" / "val" / (sorted(gt)[0] + ".txt")).read_text().strip().split("\n")
            self.assertEqual(len(lab), 2)
            self.assertTrue(all(r.startswith("0 ") for r in lab))
            from splice_det import metrics as M
            heavy = M.crowd_ground_truth(gt, sorted(gt), "heavy")[0]
            self.assertEqual(heavy["ign"].tolist(), [True, False, True])
            with self.assertRaises(D.DataError):
                D.discover_crowdhuman(Path(tmp) / "nothing")

    def test_view_links(self):
        """Hard links by default (no additional file on the storage quota), symbolic links on
        request or when a hard link is impossible; a rebuild keeps or converts the links."""
        with tempfile.TemporaryDirectory() as tmp:
            root, views = Path(tmp) / "coco", Path(tmp) / "views"
            make_coco(root, n_train=24, n_val=2)
            rec = D.prepare_coco(root, views, log=lambda *_: None)
            img = sorted((views / "coco" / "images" / "val").iterdir())[0]
            raw = root / "images" / "val2017" / img.name
            if rec["links"]["hard"]:                                           # this file system supports hard links
                self.assertEqual(rec["links"], {"hard": 26, "symbolic": 0})
                self.assertFalse(img.is_symlink())
                self.assertEqual(os.stat(raw).st_ino, os.stat(img).st_ino)
            old = os.environ.get("SPLICE_VIEW_LINKS")
            os.environ["SPLICE_VIEW_LINKS"] = "symbolic"
            try:
                (views / "coco" / "dataset.json").unlink()                     # force a rebuild
                again = D.prepare_coco(root, views, log=lambda *_: None)
                self.assertEqual(sum(again["links"].values()), 26)
                self.assertEqual(D._link(raw, Path(tmp) / "fresh.jpg"), "symbolic")
                self.assertTrue((Path(tmp) / "fresh.jpg").is_symlink())
                stale = Path(tmp) / "stale.jpg"
                os.symlink(str(Path(tmp) / "missing.jpg"), str(stale))          # a broken link is replaced
                self.assertEqual(D._link(raw, stale), "symbolic")
                self.assertTrue(os.path.samefile(stale, raw))
            finally:
                if old is None:
                    os.environ.pop("SPLICE_VIEW_LINKS", None)
                else:
                    os.environ["SPLICE_VIEW_LINKS"] = old
            self.assertEqual(again["counts"], rec["counts"])

    def test_calibration_split_is_stable(self):
        names = [f"{i:012d}.jpg" for i in range(5000)]
        share = np.mean([D.is_calibration(n, 24) for n in names])
        self.assertAlmostEqual(share, 1 / 24, delta=0.01)
        self.assertEqual(D.is_calibration("000000000139.jpg", 24), D.is_calibration("/x/y/000000000139.png", 24))


if __name__ == "__main__":
    unittest.main()
