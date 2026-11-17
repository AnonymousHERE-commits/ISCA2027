import random
import unittest
from fractions import Fraction

import _path  # noqa: F401
import splicev_gen as gen
from splicev_ref import bf24 as bf
from splicev_ref import posterior, trust
from splicev_ref.exact import oracles


def val(a):
    return Fraction(a[0]) * (Fraction(2) ** a[1])


class PosteriorTests(unittest.TestCase):
    def test_marginals_match_enumeration(self):   # Prop. 2(i)
        rnd = random.Random(5)
        worst = 0.0
        for _ in range(150):
            n = rnd.randint(1, 9)
            s = [rnd.randint(-900, 900) for _ in range(n)]
            w = [bf.exp_q78(x) for x in s]
            prior = posterior.prior_row(0)
            got, _z = posterior.esp_marginals(w, prior)
            want = oracles.exact_marginals([val(x) for x in w], [val(p) for p in prior])
            for g, e in zip(got, want):
                worst = max(worst, float(abs(val(g) / e - 1)))
        self.assertLess(worst, 2.0 ** -14.7)

    def test_marginals_sum_to_expected_partner_count(self):
        w = [bf.exp_q78(0)] * 5
        p, _ = posterior.esp_marginals(w, posterior.prior_row(0))
        prior = [bf.to_float(x) for x in posterior.prior_row(0)]
        expected = sum((k + 1) * prior[k] for k in range(3)) / sum(prior)
        self.assertAlmostEqual(sum(bf.to_float(x) for x in p), expected, places=5)

    def test_mutual_posterior_is_the_minimum(self):
        partners = [[1], [0]]
        marg = [[bf.from_fixed(3, 2)], [bf.from_fixed(1, 2)]]
        mu = posterior.mutual_posteriors(partners, marg)
        self.assertEqual(mu[0][0], bf.from_fixed(1, 2))
        self.assertEqual(mu[1][0], bf.from_fixed(1, 2))


class TrustTests(unittest.TestCase):
    ONE = 1 << 39

    def random_row(self, rnd, n):
        O = self.ONE
        p = [rnd.randint(0, O) for _ in range(n)]
        rho = [rnd.randint(O // 8, 4 * O) for _ in range(n)]
        phi = [[rnd.randint(-O // 2, O // 2) for _ in range(4)] for _ in range(n)]
        need = rnd.randint(0, 2 * O)
        return need, p, rho, phi

    def exact_problem(self, need, p, rho, phi, keep):
        O = Fraction(self.ONE)
        g = Fraction(need) / O
        A, b = [], []
        for i in keep:
            row = []
            for j in keep:
                v = g + sum(Fraction(phi[i][t]) * Fraction(phi[j][t]) for t in range(4)) / (O * O)
                if i == j:
                    d = Fraction(gen.A_R_Q39) / Fraction(rho[i]) + Fraction(gen.B_R_Q39) / O * (1 - Fraction(p[i]) / O)
                    d = min(max(d, Fraction(1, 16)), Fraction(64))
                    v += d
                row.append(v)
            A.append(row)
            b.append(-g + Fraction(gen.C_R_Q39) / O * (1 - Fraction(p[i]) / O))
        return A, b

    def test_fixed_point_solution_is_near_the_exact_minimiser(self):   # Prop. 2(iii)
        rnd = random.Random(9)
        worst = 0.0
        for _ in range(120):
            n = rnd.randint(1, 8)
            need, p, rho, phi = self.random_row(rnd, n)
            row = trust.solve_row(need, list(range(n)), p, rho, phi)
            if row.status != "converged":
                continue
            keep = list(row.sources)
            if not keep:
                continue
            A, b = self.exact_problem(need, p, rho, phi, keep)
            x = oracles.exact_nnqp(A, b)
            for got, want in zip(row.lam, x):
                worst = max(worst, abs(got / self.ONE - float(want)))
        self.assertLess(worst, 2.0 ** -13.2)

    def test_screening_is_sound(self):   # Prop. 2(ii): screened units are zero at the full optimum
        rnd = random.Random(13)
        checked = 0
        for _ in range(200):
            n = rnd.randint(2, 6)
            need, p, rho, phi = self.random_row(rnd, n)
            row = trust.solve_row(need, list(range(n)), p, rho, phi, ms=n)
            if row.screened == 0:
                continue
            A, b = self.exact_problem(need, p, rho, phi, list(range(n)))
            x = oracles.exact_nnqp(A, b)
            for k in range(n):
                if k not in row.sources:
                    self.assertEqual(x[k], 0)
            checked += 1
        self.assertGreater(checked, 5)

    def test_iterates_stay_feasible_and_clamped(self):
        rnd = random.Random(21)
        for _ in range(200):
            n = rnd.randint(1, 8)
            need, p, rho, phi = self.random_row(rnd, n)
            row = trust.solve_row(need, list(range(n)), p, rho, phi, kqp=3)
            self.assertTrue(all(0 <= l <= gen.LAMBDA_MAX_Q39 for l in row.lam))
            self.assertIn(row.status, ("converged", "capped", "empty"))
            self.assertLessEqual(len(row.sources), gen.MS)

    def test_empty_row(self):
        row = trust.solve_row(1 << 39, [], [], [], [])
        self.assertEqual(row.source_list(), [0xFFFF] * gen.MS)


if __name__ == "__main__":
    unittest.main()
