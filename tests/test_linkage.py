import itertools
import os
import random
import unittest

import numpy as np

import _path  # noqa: F401
import splicev_gen as gen
from splicev_ref import linkage as lk

LEVELS = (0, 96, 127, 128, 179, 205, 255)


def same(a, b):
    return lk.partition_of(a) == lk.partition_of(b)


class LinkageTests(unittest.TestCase):
    def test_four_anchor_windows_exhaustive(self):   # Lemma 2: all 7^6 (complete) and 7^4 (four pairs)
        n = 4
        active = [True] * n
        for pairs in (list(itertools.combinations(range(4), 2)), [(0, 3), (1, 2), (1, 3), (2, 3)]):
            for qs in itertools.product(LEVELS, repeat=len(pairs)):
                edges = [(a, b, q) for (a, b), q in zip(pairs, qs)]
                self.assertTrue(same(lk.certified(n, edges, active, None), lk.sequential(n, edges, active)), edges)

    def test_random_tie_rich_windows(self):
        rnd = random.Random(1)
        trials = int(os.environ.get("SPLICEV_LINK_TRIALS", "600"))
        plain_fail = small_fail = 0
        for _ in range(trials):
            n = rnd.randint(5, 9)
            edges = [(a, b, rnd.choice(LEVELS)) for a, b in itertools.combinations(range(n), 2) if rnd.random() < 0.7]
            active = [True] * n
            seq = lk.sequential(n, edges, active)
            self.assertTrue(same(lk.certified(n, edges, active, None), seq), edges)
            plain_fail += not same(lk.certified(n, edges, active, None, plain_mnn=True), seq)
            small_fail += not same(lk.certified(n, edges, active, None, key_rule="min"), lk.sequential(n, edges, active, key_rule="min"))
        # the uncertified variants are different algorithms; record that they do diverge
        self.assertGreater(plain_fail + small_fail, 0)

    def test_budget_branch_is_deterministic(self):
        rnd = np.random.default_rng(3)
        n = 400
        edges = []
        for i in range(n):
            for j in range(i + 1, min(n, i + 6)):
                edges.append((i, j, int(rnd.integers(100, 256))))
        active = [True] * n
        full = lk.certified(n, edges, active, None)
        cut = lk.certified(n, edges, active, budget=60)
        cut2 = lk.certified(n, edges, active, budget=60)
        self.assertTrue(cut.budget_hit)
        self.assertEqual(cut.labels, cut2.labels)
        self.assertEqual(cut.work, cut2.work)
        self.assertGreaterEqual(len(set(cut.labels.values())), len(set(full.labels.values())))
        self.assertTrue(same(full, lk.sequential(n, edges, active)))

    def test_fragment_selection_caps_by_mass(self):
        res = lk.certified(6, [(0, 1, 255), (2, 3, 255), (4, 5, 255)], [True] * 6, None)
        ids, F = lk.select_fragments(res, 6, [10, 10, 200, 200, 50, 50], f_cap=2)
        self.assertEqual(F, 2)
        self.assertEqual(ids[0], gen.UNASSIGNED)
        self.assertEqual(len({int(ids[2]), int(ids[4])}), 2)

    def test_window_edges_geometry(self):
        sc = np.full((16, 12), 200)
        edges = lk.window_edges(sc, np.array([0]), np.full((1, 25), 150), np.full((1, 8), 0xFFFF), np.zeros((1, 8)), 4, 4, 8)
        p4 = [e for e in edges if e[1] < 16]
        self.assertEqual(len(p4), len({(a, b) for a, b, _ in p4}))
        for a, b, _ in p4:
            ya, xa = divmod(a, 4)
            yb, xb = divmod(b, 4)
            self.assertLessEqual(max(abs(ya - yb), abs(xa - xb)), 2)
        self.assertEqual(len([e for e in edges if e[1] == 16]), 9)   # reserve at (0,0): 3x3 inside


if __name__ == "__main__":
    unittest.main()
