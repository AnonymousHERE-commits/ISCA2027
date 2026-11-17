import random
import unittest

import numpy as np

import _path  # noqa: F401
from splicev_ref.arith import (Counters, check_word, div_rne, div_rne_np, rshift_rne, rshift_rne_np, sat8, sat8_sym)


class RoundingTests(unittest.TestCase):
    def test_half_even_ties_positive_and_negative(self):
        # x / 2: ties go to the even neighbour
        cases = {1: 0, 3: 2, 5: 2, 7: 4, -1: 0, -3: -2, -5: -2, -7: -4, 2: 1, -2: -1}
        for x, want in cases.items():
            self.assertEqual(rshift_rne(x, 1), want, x)
        self.assertEqual(rshift_rne(6, 2), 2)    # 1.5 -> 2
        self.assertEqual(rshift_rne(10, 2), 2)   # 2.5 -> 2
        self.assertEqual(rshift_rne(-6, 2), -2)  # -1.5 -> -2
        self.assertEqual(rshift_rne(-10, 2), -2) # -2.5 -> -2

    def test_zero_and_negative_shift_is_exact(self):
        for x in (-129, -1, 0, 1, 127, 1 << 40):
            self.assertEqual(rshift_rne(x, 0), x)
            self.assertEqual(rshift_rne(x, -3), x * 8)

    def test_div_rne_matches_fraction_rounding(self):
        from fractions import Fraction
        rnd = random.Random(7)
        for _ in range(3000):
            n = rnd.randint(-10 ** 6, 10 ** 6)
            d = rnd.randint(1, 5000)
            q = div_rne(n, d)
            exact = Fraction(n, d)
            self.assertLessEqual(abs(exact - q), Fraction(1, 2))
            if abs(exact - q) == Fraction(1, 2):
                self.assertEqual(q % 2, 0)

    def test_numpy_forms_equal_scalar(self):
        rnd = np.random.default_rng(3)
        x = rnd.integers(-(1 << 40), 1 << 40, size=4000)
        for s in (1, 5, 14, 23, 39):
            got = rshift_rne_np(x, s)
            want = np.array([rshift_rne(int(v), s) for v in x])
            self.assertTrue(np.array_equal(got, want), s)
        d = rnd.integers(1, 1 << 20, size=4000)
        self.assertTrue(np.array_equal(div_rne_np(x, d), np.array([div_rne(int(a), int(b)) for a, b in zip(x, d)])))

    def test_saturation_is_counted(self):
        c = Counters()
        self.assertEqual(sat8(200, c), 127)
        self.assertEqual(sat8(-200, c), -128)
        self.assertEqual(sat8_sym(-200, c, "sym"), -127)
        self.assertEqual(c.get("sat8"), 2)
        self.assertEqual(c.get("sym"), 1)

    def test_check_word_is_order_sensitive(self):
        a = check_word([1, 2, 3])
        b = check_word([3, 2, 1])
        self.assertNotEqual(a, b)
        self.assertEqual(a, check_word([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
