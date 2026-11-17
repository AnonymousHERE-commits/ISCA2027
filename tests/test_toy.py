import unittest

import numpy as np

import _path  # noqa: F401
import splicev_gen as gen
from splicev_ref import toy


class ToySliceTests(unittest.TestCase):
    def test_slice_is_deterministic_and_bounded(self):
        for seed in range(4):
            a = toy.run(toy.make_scene(seed))
            b = toy.run(toy.make_scene(seed))
            for k in a["boundaries"]:
                self.assertTrue(np.array_equal(a["boundaries"][k], b["boundaries"][k]), (seed, k))
            self.assertLessEqual(int(np.abs(a["r"]).max(initial=0)), gen.RES_BOUND)
            self.assertLessEqual(int(np.abs(a["dh_hat"]).max(initial=0)), gen.FC_B)
            self.assertTrue(np.array_equal(a["adm"], a["adm"].T))

    def test_exchange_is_exercised(self):
        tr = toy.run(toy.make_scene(0))
        self.assertGreater(int((tr["sources"] != 0xFFFF).sum()), 0)
        self.assertGreater(int(np.abs(tr["dV"]).max()), 0)
        self.assertGreater(int(tr["adm"].sum()), 0)

    def test_residual_never_reaches_the_box_band(self):   # Prop. 5(i)
        sc = toy.make_scene(1)
        tr = toy.run(sc)
        sc["heads"]["Z"] = -sc["heads"]["Z"]          # changes r Z, hence only the class band
        tr2 = toy.run(sc)
        self.assertTrue(np.array_equal(tr["box_in"], tr2["box_in"]))


if __name__ == "__main__":
    unittest.main()
