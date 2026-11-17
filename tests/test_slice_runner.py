import sys
import unittest

import _path  # noqa: F401
from _path import REPO

sys.path.insert(0, str(REPO / "ps"))

from splicev_ps.slice_runner import RefBackend, compare, run_slice  # noqa: E402
from splicev_ref import toy  # noqa: E402


class SliceRunnerTests(unittest.TestCase):
    def test_orchestrated_slice_equals_reference_slice(self):
        """The unit/engine orchestration (as run on the board) reproduces toy.run bit for bit."""
        for seed in range(3):
            sc = toy.make_scene(seed)
            want = toy.run(sc)
            got = run_slice(toy.make_scene(seed), RefBackend())
            cmp = compare(got, want)
            self.assertTrue(cmp["all_zero_mismatch"], (seed, cmp))

    def test_worker_classes_do_not_change_the_slice(self):
        """Routing units onto split worker classes (16 KB program fit) is invisible at the boundaries."""
        want = toy.run(toy.make_scene(7))
        for masks in ([0x03, 0x44, 0xB8], [0x01, 0x02, 0x04, 0x40, 0x38, 0x88]):
            got = run_slice(toy.make_scene(7), RefBackend(masks))
            self.assertTrue(compare(got, want)["all_zero_mismatch"], masks)


if __name__ == "__main__":
    unittest.main()
