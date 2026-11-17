"""detector/hpc/download_datasets.sh against a local web server with tiny archives in the
layout of the real sources: download, extraction, completeness checks, a second run that
skips everything, an incomplete mirror, and the folders the launcher's dataset discovery
then finds."""
import functools
import http.server
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "detector" / "hpc" / "download_datasets.sh"
sys.path.insert(0, str(REPO / "detector"))

try:
    from PIL import Image
except ImportError:                                   # pragma: no cover
    Image = None


def _jpg(i: int) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (32, 24), (i % 255, 3, 4)).save(b, "JPEG")
    return b.getvalue()


def make_mirror(srv: Path) -> None:
    (srv / "zips").mkdir(parents=True)
    (srv / "annotations").mkdir()
    (srv / "ch").mkdir()
    for name, n in (("train2017", 7), ("val2017", 3)):
        with zipfile.ZipFile(srv / "zips" / f"{name}.zip", "w") as z:
            for i in range(n):
                z.writestr(f"{name}/{i:012d}.jpg", _jpg(i))
    with zipfile.ZipFile(srv / "annotations" / "annotations_trainval2017.zip", "w") as z:
        for s in ("train2017", "val2017"):
            z.writestr(f"annotations/instances_{s}.json", json.dumps({"images": [], "annotations": [], "categories": []}))
    ids = {"train": [f"t{i},x" for i in range(6)], "val": [f"v{i},y" for i in range(2)]}
    parts = {"CrowdHuman_train01": ids["train"][:2], "CrowdHuman_train02": ids["train"][2:4],
             "CrowdHuman_train03": ids["train"][4:], "CrowdHuman_val": ids["val"]}
    for part, names in parts.items():
        with zipfile.ZipFile(srv / "ch" / f"{part}.zip", "w") as z:
            for i, n in enumerate(names):
                z.writestr(f"Images/{n}.jpg", _jpg(i))
    for s in ("train", "val"):
        (srv / "ch" / f"annotation_{s}.odgt").write_text("\n".join(json.dumps({"ID": i, "gtboxes": []}) for i in ids[s]) + "\n")
    (srv / "yolo26s.pt").write_bytes(b"weights placeholder")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):                     # keep the test output clean
        pass


@unittest.skipIf(Image is None, "Pillow not installed")
@unittest.skipUnless(shutil.which("bash") and (shutil.which("wget") or shutil.which("curl")), "needs bash and wget or curl")
class Download(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        make_mirror(root / "srv")
        self.base = root / "SPLICE-V"
        self.base.mkdir()
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(root / "srv")))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("SPLICE_")}
        self.env.update(SPLICE_BASE=str(self.base), SPLICE_COCO_URL=url, SPLICE_CROWDHUMAN_URL=f"{url}/ch",
                        SPLICE_WEIGHTS_URL=f"{url}/yolo26s.pt", SPLICE_N_COCO_TRAIN="7", SPLICE_N_COCO_VAL="3",
                        SPLICE_N_CROWD_TRAIN="6", SPLICE_N_CROWD_VAL="2")

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def run_script(self, *items, **extra):
        return subprocess.run(["bash", str(SCRIPT), *items], env=dict(self.env, **extra), capture_output=True, text=True)

    def test_download_extract_verify(self):
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        coco, crowd = self.base / "datasets" / "coco", self.base / "datasets" / "crowdhuman"
        self.assertEqual(len(list((coco / "train2017").iterdir())), 7)
        self.assertEqual(len(list((coco / "val2017").iterdir())), 3)
        self.assertEqual(len(list((crowd / "Images").iterdir())), 8)
        self.assertEqual((self.base / "checkpoints" / "pretrained" / "yolo26s.pt").read_bytes(), b"weights placeholder")
        self.assertFalse(list(self.base.rglob("*.zip")) + list(self.base.rglob("*.part")))      # archives removed after the check
        from splice_det import data as D                # the launcher's discovery finds exactly these folders
        found = D.discover_coco(coco)
        self.assertEqual(found["train2017"], coco / "train2017")
        self.assertEqual(found["ann_val2017"], coco / "annotations" / "instances_val2017.json")
        self.assertEqual(len(D.discover_crowdhuman(crowd)["index"]), 8)
        again = self.run_script()                       # nothing left to do
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertNotIn("downloading", again.stdout)
        self.assertIn("COCO train2017: complete (7 images)", again.stdout)

    def test_incomplete_mirror_and_missing_file_fail(self):
        r = self.run_script("coco", SPLICE_N_COCO_VAL="5")
        self.assertEqual(r.returncode, 1)
        self.assertIn("COCO val2017: 3 images after extraction, expected 5", r.stderr)
        self.assertTrue((self.base / "datasets" / "coco" / "val2017.zip").exists())             # kept for inspection
        r = self.run_script("weights", SPLICE_WEIGHTS_URL=self.env["SPLICE_COCO_URL"] + "/nothing.pt")
        self.assertEqual(r.returncode, 1)
        self.assertIn("download failed", r.stderr)
        self.assertFalse((self.base / "checkpoints" / "pretrained" / "yolo26s.pt").exists())
        r = self.run_script("imagenet")
        self.assertEqual(r.returncode, 1)
        self.assertIn("unknown item", r.stderr)

    def test_keep_archives(self):
        r = self.run_script("crowdhuman", SPLICE_KEEP_ZIPS="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(list((self.base / "datasets" / "crowdhuman").glob("*.zip"))), 4)


if __name__ == "__main__":
    unittest.main()
