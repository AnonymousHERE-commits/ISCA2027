"""The PS board code (WorkerPool, Engines, BoardBackend, apps) on an emulated device.

The emulated kernels have the argument lists of the HLS sources (checked below), so these
tests catch wrong argument indices, buffer sizes, routing and response framing in the host
program before it reaches the board.  They say nothing about the hardware itself.
"""
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import _path  # noqa: F401
from _path import REPO

sys.path.insert(0, str(REPO / "ps"))

from splicev_ps.fakexrt import SIGNATURES, FakeDevice  # noqa: E402
from splicev_ps.slice_runner import BoardBackend, compare, run_slice  # noqa: E402
from splicev_ps.workers import Worker, WorkerPool  # noqa: E402
from splicev_ps.xrtcompat import XrtError  # noqa: E402
from splicev_ref import routing, toy, units  # noqa: E402
from splicev_ref import unit_builders as ub  # noqa: E402

SPLIT3 = [0x03, 0x44, 0xB8]


def hls_signatures():
    sigs = {}
    for f in sorted((REPO / "pl" / "engines").glob("*.cpp")):
        text = f.read_text()
        for m in re.finditer(r'extern "C" void (\w+)\(([^)]*)\)', text):
            params = []
            for p in m.group(2).split(","):
                p = " ".join(p.split())
                name = re.findall(r"(\w+)$", p)[0]
                if "hls::stream" in p:
                    kind = "stream"
                elif "*" in p:
                    kind = "in" if p.startswith("const ") else "out"
                else:
                    kind = "scalar"
                params.append((kind, name))
            sigs[m.group(1)] = params
    return sigs


def workers(masks):
    return [{"index": i, "mask": m, "mm2s": f"mm2s_{i}", "s2mm": f"s2mm_{i}"} for i, m in enumerate(masks)]


class BoardPathTests(unittest.TestCase):
    def test_emulated_kernels_have_the_hls_signatures(self):
        self.assertEqual(hls_signatures(), SIGNATURES)

    def test_slice_through_board_backend(self):
        for masks in ([0xFF], SPLIT3):
            be = BoardBackend(FakeDevice(masks), workers(masks))
            for seed in (0, 4):
                got = run_slice(toy.make_scene(seed), be)
                self.assertTrue(compare(got, toy.run(toy.make_scene(seed)))["all_zero_mismatch"], (masks, seed))

    def test_worker_pool_matches_reference_responses(self):
        rng = np.random.default_rng(3)
        cmds = [ub.gemm(rng.integers(-50, 50, size=(4, 9)), rng.integers(-50, 50, size=(9, 5)), shift=3, slot=i) for i in range(3)]
        cmds += [ub.lut_sigmoid([5, -7, 9000], slot=7), ub.nop(slot=8), ub.gemm(np.zeros((1, 200)), np.zeros((200, 256)), slot=9)]
        want = units.split_responses(units.run_stream([w for c in cmds for w in c]))
        pool = WorkerPool(FakeDevice(SPLIT3), workers(SPLIT3))
        got, meta = pool.submit(cmds)
        self.assertEqual([units.split_responses(g)[0] for g in got], want)
        self.assertEqual(len(meta["workers"]), len([g for g in routing.group(cmds, SPLIT3) if g]))

    def test_short_response_is_a_bounded_failure(self):
        dev = FakeDevice([0xFF])
        w = Worker(dev, 0, 0xFF, timeout_ms=10)
        cmd = ub.lut_expf([1, 2, 3])
        with self.assertRaises(XrtError):
            w.submit([cmd], n_out=units.response_length(cmd) + 2)   # sink waits for words that never come

    def test_apps_run_on_the_emulated_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(REPO / "tools" / "make_vectors.py"), "--out", tmp], check=True, capture_output=True)
            runs = [["g02_smoke.py"], ["g05_units.py", "--vectors", tmp, "--masks", "0x03,0x44,0xB8"],
                    ["g06_engines.py", "--vectors", tmp], ["g07_slice.py", "--seeds", "1"]]
            for r in runs:
                p = subprocess.run([sys.executable, str(REPO / "ps" / "apps" / r[0]), "--backend", "emulated"] + r[1:],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertIn("PASS [EMULATED", p.stdout)


if __name__ == "__main__":
    unittest.main()
