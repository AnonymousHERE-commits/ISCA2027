"""Synthetic-scene training of arm (e) (detector/splice_det/toytrain.py): gradients of the
forecast against finite differences, a small train -> calibrate -> export run, and the
committed artifacts (artifacts/trained/synthetic_arm_e) against the contract checker."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "detector"))

from splice_det import toytrain as tt  # noqa: E402
from splicev_ref import gen, svb  # noqa: E402

ART = REPO / "artifacts" / "trained" / "synthetic_arm_e"


class ForecastGradient(unittest.TestCase):
    def test_finite_differences(self):
        base = tt.shared_heads()
        data = tt.traces([1000, 1001, 1002, 1003], base)
        self.assertTrue(data)
        d = data[0]
        rng = np.random.default_rng(0)
        p = {k: np.asarray(base[k], dtype=np.float64) + rng.normal(0, 0.3, np.shape(base[k])) for k in tt.TRAINED}
        G = rng.normal(0, 1, (d["F"], gen.C_TRANSPORT))
        _, cache = tt.forecast_forward(p, d["H"], d["nu"], d["adm"])
        grads = tt.forecast_backward(p, G, cache)
        for k in tt.TRAINED:
            flat = p[k].reshape(-1)
            for idx in rng.choice(flat.size, size=min(6, flat.size), replace=False):
                h = 1e-4
                old = flat[idx]
                flat[idx] = old + h
                up = float((tt.forecast_forward(p, d["H"], d["nu"], d["adm"])[0] * G).sum())
                flat[idx] = old - h
                dn = float((tt.forecast_forward(p, d["H"], d["nu"], d["adm"])[0] * G).sum())
                flat[idx] = old
                num = (up - dn) / (2 * h)
                ana = float(grads[k].reshape(-1)[idx])
                self.assertLess(abs(num - ana), 1e-4 * max(1.0, abs(num)), (k, idx, num, ana))


class SmallRun(unittest.TestCase):
    def test_train_calibrate_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            cmd = [sys.executable, "-m", "splice_det.toytrain", "--out", str(out), "--train", "10", "--val", "4",
                   "--calib", "12", "--test", "4", "--epochs", "4", "--class-epochs", "2"]
            res = subprocess.run(cmd, cwd=str(REPO / "detector"), capture_output=True, text=True, timeout=900)
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            rep = json.loads((out / "report.json").read_text())
            self.assertTrue(rep["slice_check"]["all"])
            for f in ("tensors.svb", "float_tensors.npz", "inventory.json", "calibration.json", "manifest.model.json",
                      "export_report.json", "heads.json", "train_log.jsonl"):
                self.assertTrue((out / f).exists(), f)
            chk = subprocess.run([sys.executable, str(REPO / "tools" / "check_contract.py"), "--manifest",
                                  str(out / "manifest.model.json")], capture_output=True, text=True)
            self.assertEqual(chk.returncode, 0, chk.stdout + chk.stderr)


@unittest.skipUnless((ART / "manifest.model.json").exists(), "synthetic artifacts not present")
class CommittedArtifacts(unittest.TestCase):
    def test_hashes_and_contract(self):
        rep = json.loads((ART / "export_report.json").read_text())
        from splice_det.export import sha256_file
        for f, h in rep["sha256"].items():
            self.assertEqual(sha256_file(ART / f), h, f)
        chk = subprocess.run([sys.executable, str(REPO / "tools" / "check_contract.py"), "--manifest",
                              str(ART / "manifest.model.json")], capture_output=True, text=True)
        self.assertEqual(chk.returncode, 0, chk.stdout + chk.stderr)
        self.assertIn("deployable: NO", chk.stdout)

    def test_trained_heads_reduce_the_residual(self):
        rep = json.loads((ART / "report.json").read_text())
        before, after = rep["residual_test_before"], rep["residual_test_after"]
        self.assertLess(after["mse_r"], 0.6 * before["mse_r"])
        self.assertLess(after["mse_r"], after["mse_dH_zero_forecast"])
        self.assertTrue(rep["slice_check"]["all"])
        heads = svb.read(ART / "tensors.svb")
        for k in tt.TRAINED:
            w = heads[f"heads.{k}"]
            self.assertTrue(np.all(np.abs(w) <= 127), k)

    def test_slice_bit_exact_under_calibrated_profile(self):
        """The trained heads run through the units and engines bit-exactly with the reference,
        under the generated sources of the calibrated manifest (one scene)."""
        env = dict(os.environ, SPLICEV_GEN_DIR=str(ART / "generated" / "py"))
        code = ("import json, numpy as np, sys; sys.path.insert(0, 'detector');"
                "from splice_det import toytrain as tt;"
                f"h = {{k: np.asarray(v) for k, v in json.load(open(r'{ART / 'heads.json'}')).items()}};"
                "print(json.dumps(tt.slice_check([9000], h)))")
        res = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        self.assertTrue(json.loads(res.stdout.strip().splitlines()[-1])["all"])


if __name__ == "__main__":
    unittest.main()
