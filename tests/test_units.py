import unittest

import numpy as np

import _path  # noqa: F401
import splicev_gen as gen
from splicev_ref import forecast as fc, gate, membership as mb, readout, units, walk
from splicev_ref import unit_builders as ub
from splicev_ref.arith import rshift_rne_np


def run(cmd):
    r = units.split_responses(units.run_stream(cmd))
    assert len(r) == 1 and r[0]["hdr"]["header_ok"] and r[0]["hdr"]["trailer_ok"]
    return r[0]["hdr"], r[0]["payload"]


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(10)

    def test_gemm_int8_and_int16(self):
        A = self.rng.integers(-128, 128, size=(5, 37))
        B = self.rng.integers(-128, 128, size=(37, 11))
        bias = self.rng.integers(-5000, 5000, size=11)
        hdr, out = run(ub.gemm(A, B, bias=bias, shift=9))
        self.assertEqual(hdr["status"], gen.ST_OK)
        want = np.clip(rshift_rne_np(A @ B + bias, 9), -128, 127)
        self.assertTrue(np.array_equal(units.unpack_rows(out, 5, 11, "8", True), want))
        A16 = self.rng.integers(-191, 192, size=(4, 48))
        hdr, out = run(ub.gemm(A16, B[:37, :8].repeat(2, axis=0)[:48], shift=7, out="16"))
        self.assertEqual(hdr["status"], gen.ST_OK)

    def test_stencil_tile_with_halo_equals_full_lattice(self):
        H = W = 22
        s = 4
        wc = np.where(walk.inside_mask(H, W), self.rng.integers(0, 1 << 15, size=(H, W, 8)), 0)
        taps = walk.export_taps(wc, walk.inside_mask(H, W))
        betas = np.sort(self.rng.integers(2000, 63000, size=(H, W, 4)), axis=-1) + np.arange(4)
        X = self.rng.integers(-128, 128, size=(H, W, 3))
        full = walk.walk_bands(X, taps, betas, s, (1, 2, 2, 3), 0)
        y0, x0, th, tw = 5, 3, 14, 15
        tile = (slice(y0, y0 + th), slice(x0, x0 + tw))
        hdr, out = run(ub.stencil(taps[tile], X[tile], s, s, betas[tile], (1, 2, 2, 3, 0)))
        self.assertEqual(hdr["status"], gen.ST_OK)
        ho, wo = th - 2 * s, tw - 2 * s
        per = units.row_words(ho * wo, "8")
        for ch in range(3):
            for j in range(5):
                blk = out[(ch * 5 + j) * per:(ch * 5 + j + 1) * per]
                got = units.unpack(blk, ho * wo, "8", True).reshape(ho, wo)
                want = full[j, y0 + s:y0 + s + ho, x0 + s:x0 + s + wo, ch]
                self.assertTrue(np.array_equal(got, want), (ch, j))

    def test_segops_match_membership(self):
        n, F, C = 50, 6, 12
        ids = self.rng.integers(0, F, size=n)
        ids[::7] = gen.UNASSIGNED
        w = self.rng.integers(1, 256, size=n)
        X = self.rng.integers(-128, 128, size=(n, C))
        hdr, out = run(ub.segsum(ids, w, X, F))
        sums, ws = mb.fragment_sums(X, ids, w, F)
        self.assertTrue(np.array_equal(units.unpack_rows(out[:F * C], F, C, "32", True), sums))
        hdr, out = run(ub.segmean(sums, ws))
        H = units.unpack_rows(out, F, C, "8", True)
        self.assertTrue(np.array_equal(H, mb.fragment_means(sums, ws)))
        hdr, out = run(ub.fill(H, ids))
        self.assertTrue(np.array_equal(units.unpack_rows(out, n, C, "8", True), mb.fill(H, ids)))
        hdr, out = run(ub.detail(H, ids, X))
        self.assertTrue(np.array_equal(units.unpack_rows(out, n, C, "16", True), mb.detail(X, H, ids)))
        dH = self.rng.integers(-60, 60, size=(F, C))
        hdr, out = run(ub.synth(dH, ids, X))
        self.assertTrue(np.array_equal(units.unpack_rows(out, n, C, "8", True), mb.synthesize(X, ids, dH, None)))

    def test_forecast_unit_matches_reference(self):
        F = 9
        K = self.rng.integers(-60, 60, size=(F, 64))
        Uv = self.rng.integers(-60, 60, size=(F, gen.FC_UDIM))
        VO = self.rng.integers(-128, 128, size=(F, 48))
        Hs = self.rng.integers(-128, 128, size=(F, 48))
        nr = self.rng.integers(-3000, 3000, size=F)
        ar = self.rng.integers(-3000, 3000, size=F)
        adm = np.triu(self.rng.random((F, F)) < 0.5, 1)
        adm = adm | adm.T
        words = gate.pack_mask(adm).reshape(gen.F_CAP, -1)
        want, _ = fc.forecast(K, Uv, VO, Hs, nr, ar, adm)
        hdr, out = run(ub.forecast(K, Uv, VO, words[2:7], Hs[2:7], nr[2:7], ar[2:7], 2))
        self.assertEqual(hdr["status"], gen.ST_OK)
        per = units.row_words(48, "8") + 1
        for r in range(5):
            got = units.unpack(out[r * per:r * per + 12], 48, "8", True)
            self.assertTrue(np.array_equal(got, want[2 + r]))

    def test_score_unit(self):
        lg = self.rng.integers(-32768, 32768, size=70)
        g = self.rng.integers(0, 1 << 15, size=70)
        hdr, out = run(ub.score(lg, g))
        p = units.unpack(out, 70, "16", False)
        self.assertTrue(np.array_equal(p, readout.class_scores(lg, g)))
        self.assertEqual(hdr["aux0"], len(readout.candidates(readout.class_scores(lg, g))))

    def test_error_statuses_keep_the_stream_aligned(self):
        good = ub.lut_sigmoid([0, 100, -100])
        bad_shape = list(good)
        bad_shape[gen.DESC_FIELDS["out_words"][0]] += 1
        from splicev_ref.arith import check_word
        bad_shape[15] = check_word(bad_shape[:15])
        bad_check = list(good)
        bad_check[15] ^= 1
        stream = bad_shape + good + bad_check[:16] + good
        statuses = [r["hdr"]["status"] for r in units.split_responses(units.run_stream(stream))]
        self.assertEqual(statuses, [gen.ST_BAD_SHAPE, gen.ST_OK, gen.ST_BAD_CHECK, gen.ST_OK])

    def test_scratch_limit(self):
        A = np.zeros((1, 200), dtype=np.int64)
        B = np.zeros((200, 256), dtype=np.int64)
        hdr, _ = run(ub.gemm(A, B, shift=0))
        self.assertEqual(hdr["status"], gen.ST_SCRATCH)

    def test_split_and_predicted_response_lengths(self):
        """The host sizes counted receives from predict_status/response_length, never by guessing."""
        rng = np.random.default_rng(5)
        good = ub.gemm(rng.integers(-9, 9, size=(3, 5)), rng.integers(-9, 9, size=(5, 4)), shift=2)
        big = ub.gemm(np.zeros((1, 200)), np.zeros((200, 256)))
        bad_check = list(ub.lut_sigmoid([1, 2]))
        bad_check[15] ^= 1
        cmds = [good, ub.nop(), big, bad_check[:16], ub.lut_expf([3, -4, 5])]
        stream = [w for c in cmds for w in c]
        self.assertEqual([list(c) for c in units.split_commands(stream)], [[int(w) & 0xFFFFFFFF for w in c] for c in cmds])
        for mask in (0xFF, 0x03, 0xB8):
            for c in cmds:
                rsp = units.run_stream(c, mask)
                self.assertEqual(len(rsp), units.response_length(c, mask))
                self.assertEqual(units.split_responses(rsp)[0]["hdr"]["status"], units.predict_status(c, mask))

    def test_worker_class_mask(self):
        """A worker class without a body answers BAD_OPCODE and keeps the stream aligned."""
        g = ub.gemm(np.ones((2, 3), dtype=np.int64), np.ones((3, 2), dtype=np.int64))
        s = ub.lut_sigmoid([0, 1])
        statuses = [r["hdr"]["status"] for r in units.split_responses(units.run_stream(g + s + g, body_mask=0x08))]
        self.assertEqual(statuses, [gen.ST_BAD_OPCODE, gen.ST_OK, gen.ST_BAD_OPCODE])


if __name__ == "__main__":
    unittest.main()
