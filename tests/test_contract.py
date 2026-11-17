import json
import subprocess
import sys
import unittest

import _path  # noqa: F401
from _path import REPO
from svgen.manifest import load_manifest, load_raw, null_fields


class ContractTests(unittest.TestCase):
    def test_paper_seed_is_not_deployable(self):
        m = load_manifest(REPO / "contracts" / "manifest.paper.json")
        self.assertFalse(m["deployable"])
        nulls = null_fields(m)
        for field in ("bands.exponents", "partition.catalog", "gate.thresholds", "trust.c_r", "forecast.exp_table"):
            self.assertIn(field, nulls)

    def test_dev_profile_documents_every_override(self):
        raw = load_raw(REPO / "contracts" / "manifest.dev.json")
        missing = [p for p in raw["overrides"] if p not in raw["decisions"]]
        self.assertEqual(missing, [])
        for p, d in raw["decisions"].items():
            self.assertIn(d["kind"], ("design", "synthetic"), p)
            self.assertTrue(d["id"].startswith(("DD-", "SYN-")), p)

    def test_dev_profile_never_deployable(self):
        m = load_manifest(REPO / "contracts" / "manifest.dev.json")
        self.assertFalse(m["deployable"])
        r = subprocess.run([sys.executable, str(REPO / "tools" / "check_contract.py"), "--require-deployable"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("deployable: NO", r.stdout)

    def test_e_dup_is_not_silently_capped(self):
        m = load_manifest(REPO / "contracts" / "manifest.dev.json")
        self.assertIsNone(m["posterior"]["duplicate_edge_bound"])

    def test_detector_export_spec_covers_every_open_field(self):
        """Every SYN/DD value and every null of the dev profile is an item the trained export must supply."""
        r = subprocess.run([sys.executable, str(REPO / "tools" / "make_export_spec.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)
        spec = json.loads((REPO / "detector" / "export_spec.json").read_text())
        listed = {it["manifest_field"] for it in spec["manifest_fields"]}
        m = load_manifest(REPO / "contracts" / "manifest.dev.json")
        self.assertTrue(set(m["decisions"]) <= listed)
        self.assertTrue(set(null_fields(m)) <= listed)
        for name in ("W_K", "U", "u0", "u_alpha", "O_F", "Z_l", "hypothesis_head"):
            self.assertIn(name, {t["name"] for t in spec["tensors"]})


if __name__ == "__main__":
    unittest.main()
