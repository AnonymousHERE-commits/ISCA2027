import math
import random
import unittest
from fractions import Fraction

import _path  # noqa: F401
from splicev_ref import bf24 as bf


def val(a):
    return Fraction(a[0]) * (Fraction(2) ** a[1])


class BF24Tests(unittest.TestCase):
    def setUp(self):
        self.rnd = random.Random(11)

    def rand_bf(self):
        return bf.norm(self.rnd.randint(1, (1 << 30)), self.rnd.randint(-40, 40))

    def test_norm_is_round_half_even(self):
        a = bf.norm((1 << 24) + 1, 0)      # 2^24 + 1 -> tie at 1/2 ulp -> even (2^23 * 2)
        self.assertEqual(a, (1 << 23, 1))
        b = bf.norm((1 << 24) + 3, 0)      # ...+3 -> 1.5 ulp tie -> rounds up to even mantissa
        self.assertEqual(b, ((1 << 23) + 2, 1))

    def test_mul_and_add_are_correctly_rounded(self):
        for _ in range(3000):
            a, b = self.rand_bf(), self.rand_bf()
            for op, exact in ((bf.mul, val(a) * val(b)), (bf.add, val(a) + val(b))):
                r = op(a, b)
                ulp = Fraction(2) ** r[1]
                self.assertLessEqual(abs(val(r) - exact), ulp / 2)
                self.assertTrue((1 << 23) <= r[0] < (1 << 24))

    def test_recip_relative_error(self):
        worst = 0.0
        for _ in range(4000):
            a = self.rand_bf()
            r = bf.recip(a)
            rel = abs(val(r) * val(a) - 1)
            worst = max(worst, float(rel))
        self.assertLess(worst, 2.0 ** -23)

    def test_exp_relative_error_below_2_pow_minus_20(self):
        worst = 0.0
        for s in list(range(-2000, 2000, 7)) + [-32768 // 4, 32767 // 4]:
            got = bf.to_float(bf.exp_q78(s))
            want = math.exp(s / 256.0)
            worst = max(worst, abs(got / want - 1))
        self.assertLess(worst, 2.0 ** -20)

    def test_compare_and_fixed(self):
        a, b = bf.from_int(5), bf.from_int(7)
        self.assertEqual(bf.cmp(a, b), -1)
        self.assertEqual(bf.cmp(b, a), 1)
        self.assertEqual(bf.cmp(a, a), 0)
        self.assertEqual(bf.cmp(bf.ZERO, a), -1)
        self.assertEqual(bf.to_fixed(bf.from_int(3), 39), 3 << 39)


if __name__ == "__main__":
    unittest.main()
