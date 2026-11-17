"""Build configuration tooling: rendered tool configs, generated AIE graphs, release checks."""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
from _path import REPO

TOOLS = REPO / "tools"


def run(*args, check=True):
    p = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise AssertionError(p.stdout + p.stderr)
    return p


class BuildToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.table = json.loads((REPO / "pl" / "kernels.json").read_text())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_hls_configs_render_for_every_kernel(self):
        for k, kd in self.table["kernels"].items():
            for caps in self.table["caps"]:
                out = self.tmp / caps / k / "hls_config.cfg"
                run(TOOLS / "build_cfg.py", "hls", "--kernel", k, "--caps", caps, "--out", out)
                text = out.read_text()
                body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
                self.assertNotIn("@", body)
                self.assertIn(f"syn.top={k}", text)
                for key in ("syn.file", "tb.file"):
                    m = re.search(rf"^{re.escape(key)}=(\S+)", text, re.M)
                    if m:
                        self.assertTrue(Path(m.group(1)).is_file(), m.group(1))
                self.assertEqual(kd.get("tb") is not None, "tb.file=" in text)
                for name, val in self.table["caps"][caps].items():
                    self.assertIn(f"-D{name}={val}", text)

    def test_connectivity_matches_generated_workers(self):
        for system, sd in self.table["systems"].items():
            args = [TOOLS / "build_cfg.py", "connectivity", "--system", system, "--out", self.tmp / f"{system}.cfg"]
            if sd["aie"] != "smoke":
                src = self.tmp / sd["aie"]
                run(TOOLS / "gen_aie_graph.py", "--set", sd["aie"], "--out", src)
                args += ["--workers-json", src / "workers.json"]
                workers = json.loads((src / "workers.json").read_text())["workers"]
            run(*args)
            text = (self.tmp / f"{system}.cfg").read_text()
            if sd["aie"] == "smoke":
                self.assertIn("ai_engine_0.smoke_in", text)
                continue
            self.assertIn(f"nk=sv_mm2s:{len(workers)}:", text)
            for w in workers:
                self.assertIn(f"stream_connect={w['mm2s']}.s:ai_engine_0.{w['plio_in']}", text)
                self.assertIn(f"stream_connect=ai_engine_0.{w['plio_out']}:{w['s2mm']}.s", text)
            for k in sd["kernels"]:
                if k not in ("sv_mm2s", "sv_s2mm"):
                    self.assertIn(f"nk={k}:1:{k}_0", text)

    def test_package_config(self):
        run(TOOLS / "build_cfg.py", "package", "--out", self.tmp / "p.cfg", "--out-dir", self.tmp / "pk")
        self.assertIn("defer_aie_run=1", (self.tmp / "p.cfg").read_text())
        run(TOOLS / "build_cfg.py", "package", "--out", self.tmp / "q.cfg", "--out-dir", self.tmp / "pk", "--autostart")
        self.assertNotIn("defer_aie_run", (self.tmp / "q.cfg").read_text())

    def test_graph_generator_refuses_uncovered_opcodes(self):
        p = run(TOOLS / "gen_aie_graph.py", "--masks", "0x03,0x44", "--out", self.tmp / "g", check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("LUT", p.stdout + p.stderr)

    def test_every_worker_class_set_covers_all_opcodes(self):
        sets = json.loads((REPO / "aie" / "worker_classes.json").read_text())["sets"]
        for name in sets:
            run(TOOLS / "gen_aie_graph.py", "--set", name, "--out", self.tmp / name)

    def test_release_refuses_an_incomplete_build(self):
        p = run(TOOLS / "release.py", "--system", "slice1", "--platform", self.tmp / "none.xpfm", "--build", self.tmp / "b",
                "--out-root", self.tmp / "rel", check=False)
        self.assertEqual(p.returncode, 1)
        rel = json.loads((self.tmp / "rel" / "slice1" / "release.json").read_text())
        self.assertTrue(rel["problems"])

    @unittest.skipUnless(shutil.which("make"), "make not installed")
    def test_makefile_hardware_targets_expand(self):
        fake = self.tmp / "fake.xpfm"
        fake.write_text("")
        for target, extra in (("system", ["SYSTEM=slice3"]), ("system", ["SYSTEM=smoke"]), ("aie-sim", ["AIE_SET=uniform1", "AIE_TARGET=x86sim"])):
            p = subprocess.run(["make", "-n", target, f"PLATFORM={fake}", *extra], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
