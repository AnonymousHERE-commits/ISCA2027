import subprocess
import sys
import unittest
from decimal import Decimal

import _path  # noqa: F401
from _path import REPO
import splicev_gen as gen
from svgen import tables as T
from svgen.manifest import load_manifest, manifest_sha256


class GeneratedSourcesTests(unittest.TestCase):
    def test_generated_tree_is_current(self):
        r = subprocess.run([sys.executable, str(REPO / "tools" / "generate_sources.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_hash_matches_manifest(self):
        m = load_manifest(REPO / "contracts" / "manifest.dev.json")
        self.assertEqual(gen.MANIFEST_SHA256, manifest_sha256(m))
        self.assertEqual(gen.DEPLOYABLE, 0)

    def test_paper_constants(self):
        self.assertEqual((gen.F_CAP, gen.N_CAP, gen.KMAX, gen.MS, gen.KQP), (256, 128, 3, 8, 16))
        self.assertEqual((gen.N_ANCHOR_P4, gen.N_RESERVE, gen.N_ANCHOR), (1600, 256, 1856))
        self.assertEqual((gen.P3_STEPS, gen.P4_STEPS, gen.P5_STEPS), (4, 8, 12))
        self.assertEqual((gen.TAP_Q, gen.STATE_FRAC, gen.COEF_R), (14, 15, 15))
        self.assertEqual(gen.LINK_BUDGET, 48000)
        self.assertEqual(gen.FC_B, 64)
        self.assertEqual((gen.FC_SHRINK_NUM, gen.FC_SHRINK_SHIFT), (15, 4))
        self.assertEqual(gen.RES_BOUND, 127 + 64)

    def test_tables_follow_their_formulas(self):
        self.assertEqual(gen.FC_EXP[128], round(256 * Decimal(8).exp()))
        self.assertTrue(all(a < b for a, b in zip(gen.FC_EXP, gen.FC_EXP[1:])))
        self.assertTrue(all(0 < v < (1 << 32) for v in gen.FC_EXP))
        self.assertTrue(all(a <= b for a, b in zip(gen.SIG_TABLE, gen.SIG_TABLE[1:])))
        self.assertLess(max(gen.SIG_TABLE), 1 << 15)
        self.assertEqual(list(gen.RECIP_SEED), T.recip_seed_table())
        self.assertEqual(list(gen.EXP2_HI), T.exp2_hi_table())
        self.assertEqual(gen.EXP2_HI[0], 1 << 23)

    def test_cpp_header_carries_the_same_hash(self):
        text = (REPO / "generated" / "cpp" / "splicev_gen.h").read_text()
        self.assertIn(gen.MANIFEST_SHA256, text)
        self.assertIn("static const int F_CAP = 256;", text)


if __name__ == "__main__":
    unittest.main()
