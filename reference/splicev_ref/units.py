"""Reference semantics of the eight AIE dispatcher unit bodies (word-level ABI).

A command is a 16-word descriptor followed by 'in_words' payload words; the
worker answers with an 8-word response header followed by 'out_words' words.
All arrays are little-endian and packed into 32-bit words: int8/u8 four per
word, int16/u16 two per word, int32 one per word, int64 two words (low, high).
2-D arrays are row-padded: every row starts on a word boundary.  The worker
validates magic, version, check word, opcode, sub-op, shape, scratch need and
both lengths before computing; for a valid header with a bad body it consumes
the payload so the stream stays aligned.

Bodies (SPLICE-V Fig. 3(b)): CONV, GEMM, STENCIL, LUT, PERMUTE, SEGOPS, TAPMOD, HEAD.
The C++ twin is aie/kernels/sv_units.h.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import gen
from .arith import check_word, div_rne, rshift_rne, rshift_rne_np
from .walk import band_coefficients, walk_step

M32 = 0xFFFFFFFF
F = gen  # short alias for generated constants


# ----------------------------------------------------------------- packing

def _words_for(n: int, per_word: int) -> int:
    return -(-n // per_word)


def pack_u8(a) -> List[int]:
    b = (np.asarray(a, dtype=np.int64) & 0xFF).astype(np.uint8).ravel()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return [int(x) for x in b.view("<u4")] if b.size else []


def pack_u16(a) -> List[int]:
    b = (np.asarray(a, dtype=np.int64) & 0xFFFF).astype(np.uint16).ravel()
    if b.size % 2:
        b = np.concatenate([b, np.zeros(1, dtype=np.uint16)])
    return [int(x) for x in b.view("<u4")] if b.size else []


def pack_u32(a) -> List[int]:
    return [int(x) & M32 for x in np.asarray(a, dtype=np.int64).ravel()]


def pack_i64(a) -> List[int]:
    out = []
    for x in np.asarray(a, dtype=object).ravel():
        v = int(x) & 0xFFFFFFFFFFFFFFFF
        out += [v & M32, v >> 32]
    return out


def pack_rows(a, kind: str) -> List[int]:
    a = np.asarray(a, dtype=np.int64)
    if a.ndim == 1:
        a = a[None, :]
    fn = {"8": pack_u8, "16": pack_u16, "32": pack_u32}[kind]
    out = []
    for row in a:
        out += fn(row)
    return out


def unpack(words: Sequence[int], n: int, kind: str, signed: bool) -> np.ndarray:
    w = np.asarray([int(x) & M32 for x in words], dtype=np.uint32)
    if kind == "8":
        v = w.view(np.uint8)[:n].astype(np.int64)
        return np.where(v >= 128, v - 256, v) if signed else v
    if kind == "16":
        v = w.view(np.uint16)[:n].astype(np.int64)
        return np.where(v >= 32768, v - 65536, v) if signed else v
    if kind == "32":
        v = w[:n].astype(np.int64)
        return np.where(v >= (1 << 31), v - (1 << 32), v) if signed else v
    if kind == "64":
        lo = w[0:2 * n:2].astype(object)
        hi = w[1:2 * n:2].astype(object)
        vals = [int(a) | (int(b) << 32) for a, b in zip(lo, hi)]
        return np.asarray([v - (1 << 64) if (signed and v >= (1 << 63)) else v for v in vals], dtype=object)
    raise ValueError(kind)


def row_words(n: int, kind: str) -> int:
    return {"8": _words_for(n, 4), "16": _words_for(n, 2), "32": n}[kind]


def unpack_rows(words: Sequence[int], rows: int, cols: int, kind: str, signed: bool) -> np.ndarray:
    rw = row_words(cols, kind)
    out = np.zeros((rows, cols), dtype=np.int64)
    for r in range(rows):
        out[r] = unpack(words[r * rw:(r + 1) * rw], cols, kind, signed)
    return out


# ----------------------------------------------------------------- descriptor

@dataclass
class Desc:
    opcode: int
    dim_m: int = 0
    dim_n: int = 0
    dim_k: int = 0
    param0: int = 0
    param1: int = 0
    param2: int = 0
    frame_id: int = 0
    slot: int = 0
    phase: int = 0
    flags: int = 0
    layer: int = 0
    band: int = 0
    partition: int = 0
    worker: int = 0
    n_workers: int = 1
    lanes: int = 0
    token: int = 0
    in_words: int = 0
    out_words: int = 0
    version: int = F.PROTO_VERSION
    manifest_hash16: int = F.MANIFEST_HASH16

    def words(self) -> List[int]:
        w = [0] * F.DESC_WORDS
        vals = dict(self.__dict__)
        vals["magic"] = F.CMD_MAGIC
        for name, (word, lsb, width) in F.DESC_FIELDS.items():
            if name == "check":
                continue
            v = int(vals.get(name, 0)) & ((1 << width) - 1)
            w[word] |= v << lsb
        w[F.DESC_FIELDS["check"][0]] = check_word(w[:F.DESC_WORDS - 1])
        return [x & M32 for x in w]


def decode_fields(words: Sequence[int], fields: Dict[str, Tuple[int, int, int]]) -> Dict[str, int]:
    out = {}
    for name, (word, lsb, width) in fields.items():
        out[name] = (int(words[word]) >> lsb) & ((1 << width) - 1)
    return out


def response_words(d: Dict[str, int], status: int, out_words: int) -> List[int]:
    """8-word response header (counters travel in the trailer)."""
    w = [0] * F.RSP_WORDS
    w[0] = F.RSP_MAGIC
    w[1] = d.get("frame_id", 0)
    w[2] = (d.get("opcode", 0) & 0xFF) | ((status & 0xFF) << 8) | ((d.get("slot", 0) & 0xFF) << 16) | ((d.get("worker", 0) & 0xFF) << 24)
    w[3] = out_words
    w[4] = d.get("token", 0)
    w[5] = d.get("in_words", 0)
    w[6] = 0
    w[7] = check_word(w[:7])
    return [x & M32 for x in w]


def trailer_words(header: Sequence[int], sat: int, aux0: int, aux1: int) -> List[int]:
    t = [sat & M32, aux0 & M32, aux1 & M32]
    return t + [check_word(t, seed=int(header[F.RSP_WORDS - 1]))]


def frame(d: Dict[str, int], status: int, out: Sequence[int], sat: int = 0, aux0: int = 0, aux1: int = 0) -> List[int]:
    hdr = response_words(d, status, len(out))
    return hdr + list(out) + trailer_words(hdr, sat, aux0, aux1)


def split_responses(words: Sequence[int]) -> List[Dict]:
    """Parse a response stream into header fields (with trailer counters) and payloads."""
    out = []
    i = 0
    words = [int(w) & M32 for w in words]
    while i < len(words):
        hdr = words[i:i + F.RSP_WORDS]
        h = decode_fields(hdr, F.RSP_FIELDS)
        n = h["out_words"]
        payload = words[i + F.RSP_WORDS:i + F.RSP_WORDS + n]
        trl = words[i + F.RSP_WORDS + n:i + F.RSP_WORDS + n + F.TRL_WORDS]
        t = decode_fields(trl, F.TRL_FIELDS)
        h["header_ok"] = h["magic"] == F.RSP_MAGIC and h["check"] == check_word(hdr[:F.RSP_WORDS - 1])
        h["trailer_ok"] = t["check"] == check_word(trl[:3], seed=hdr[F.RSP_WORDS - 1])
        for k, v in t.items():
            if k != "check":
                h[k] = v
        out.append({"hdr": h, "payload": payload})
        i += F.RSP_WORDS + n + F.TRL_WORDS
    return out


# ----------------------------------------------------------------- shapes

def _hw(v: int) -> Tuple[int, int]:
    return v & 0xFFFF, v >> 16


def a4(nbytes: int) -> int:
    """Scratch allocations are word aligned."""
    return (nbytes + 3) & ~3


def even(n: int) -> int:
    """Payload and output lengths are padded to whole 64-bit PLIO beats."""
    return n + (n & 1)


def shape_words(d: Dict[str, int]) -> Tuple[int, int, int]:
    """(in_words, out_words, scratch_bytes); in/out words are padded to an even count."""
    win, wout, scratch = _shape_words(d)
    return even(win), even(wout), scratch


def _shape_words(d: Dict[str, int]) -> Tuple[int, int, int]:
    """Unpadded lengths; scratch_bytes is exactly the sum of the word-aligned buffers the C++ body allocates."""
    op, m, n, k = d["opcode"], d["dim_m"], d["dim_n"], d["dim_k"]
    p0, p1, p2 = d["param0"], d["param1"], d["param2"]
    rw8, rw16 = (lambda x: row_words(x, "8")), (lambda x: row_words(x, "16"))
    if op == F.OP_GEMM:
        a16 = bool(p0 & F.GEMM_F_A16)
        kind_out = "32" if p0 & F.GEMM_F_OUT32 else ("16" if p0 & F.GEMM_F_OUT16 else "8")
        if m < 1 or n < 1 or k < 1:
            raise ValueError("GEMM dims")
        win = k * rw8(n) + (n if p0 & F.GEMM_F_BIAS else 0) + (rw8(n) if p0 & F.GEMM_F_PERCH else 0) \
            + m * (rw16(k) if a16 else rw8(k))
        wout = m * row_words(n, kind_out)
        scratch = 4 * (k * rw8(n)) + 4 * n + 4 * rw8(n) + 4 * rw16(k)
        return win, wout, scratch
    if op == F.OP_CONV:
        hin, win_ = _hw(m)
        cout, cin = n, k
        kk, stride = p2 & 0xFF, (p2 >> 8) & 0xFF
        if kk < 1 or stride < 1 or hin < kk or win_ < kk or cout < 1 or cin < 1:
            raise ValueError("CONV dims")
        mod = bool(p0 & F.CONV_F_MOD)
        if mod and kk != 3:
            raise ValueError("modulation needs k = 3")
        hout, wout_ = (hin - kk) // stride + 1, (win_ - kk) // stride + 1
        w_words = cout * rw8(kk * kk * cin)
        per_row = rw8(win_ * cin) + (rw16(win_ * 8) if mod else 0)
        win = w_words + (cout if p0 & F.CONV_F_BIAS else 0) + (rw8(cout) if p0 & F.CONV_F_PERCH else 0) \
            + (rw16(cin) if mod else 0) + hin * per_row
        wout = hout * rw8(wout_ * cout)
        scratch = 4 * w_words + 4 * cout + 4 * rw8(cout) + 4 * rw16(cin) + 4 * kk * per_row + 4 * rw8(wout_ * cout)
        return win, wout, scratch
    if op == F.OP_STENCIL:
        h, w = _hw(m)
        c, steps, halo = n, k, p2
        # halo >= steps for an interior tile; halo = 0 is exact for a whole lattice whose border taps are zero
        if h < 1 or w < 1 or c < 1 or steps > 15 or 2 * halo >= min(h, w):
            raise ValueError("STENCIL dims")
        ho, wo = h - 2 * halo, w - 2 * halo
        bands = bool(p0 & F.STENCIL_F_BANDS)
        in16 = bool(p0 & F.STENCIL_F_IN16)
        win = h * w * 5 + (h * w * 2 if bands else 0) + c * (rw16(h * w) if in16 else rw8(h * w))
        wout = c * (5 * rw8(ho * wo) if bands else rw16(ho * wo))
        scratch = a4(18 * h * w) + (a4(8 * ho * wo) if bands else 0) + a4(4 * h * w) + a4(2 * (steps + 1) * ho * wo)
        return win, wout, scratch
    if op == F.OP_LUT:
        if m < 1:
            raise ValueError("LUT count")
        if p0 == F.LUT_TABLE8:
            return 64 + rw8(m), rw8(m), 256
        if p0 == F.LUT_SIGMOID:
            return rw16(m), rw16(m), 0
        if p0 == F.LUT_EXPF:
            return rw16(m), m, 0
        if p0 == F.LUT_REQUANT32:
            if k < 1:
                raise ValueError("REQUANT32 parts")
            return k * m, (rw16(m) if p2 & 2 else rw8(m)), 8 * m
        raise KeyError("LUT subop")
    if op == F.OP_PERMUTE:
        nout, rwords, nin = m, n, k
        if nout < 1 or rwords < 1 or nin < 1:
            raise ValueError("PERMUTE dims")
        return rw16(nout) + nin * rwords, nout * rwords, 4 * rw16(nout) + 4 * nin * rwords
    if op == F.OP_SEGOPS:
        cells, c, nf = m, n, k
        if cells < 1 or c < 1 or nf < 1 or c > 256:
            raise ValueError("SEGOPS dims (c <= 256: one row buffer of 64 words)")
        if p0 == F.SEGOPS_SEGSUM:
            win = rw16(cells) + rw8(cells) + cells * rw8(c)
            return win, nf * c + nf, 4 * nf * c + 4 * nf + 4 * rw16(cells) + 4 * rw8(cells)
        if p0 == F.SEGOPS_SEGMEAN:
            return nf + nf * c, nf * rw8(c), 4 * nf
        if p0 == F.SEGOPS_FILL:
            return nf * rw8(c) + rw16(cells), cells * rw8(c), 4 * nf * rw8(c)
        if p0 == F.SEGOPS_DETAIL:
            return nf * rw8(c) + rw16(cells) + cells * rw8(c), cells * rw16(c), 4 * nf * rw8(c) + 4 * rw16(cells)
        if p0 == F.SEGOPS_SYNTH:
            reps = 2 if p1 & 1 else 1
            return nf * rw8(c) + rw16(cells) + reps * cells * rw8(c), cells * rw8(c), 4 * nf * rw8(c) + 4 * rw16(cells)
        if p0 == F.SEGOPS_CLSBAND:
            return 2 * nf * rw8(c) + rw16(cells), cells * rw8(c), 8 * nf * rw8(c)
        raise KeyError("SEGOPS subop")
    if op == F.OP_TAPMOD:
        cells, c, nf = m, n, k
        if p0 == F.TAPMOD_MODULATE:
            if cells < 1 or c < 1 or c > 256:
                raise ValueError("MODULATE dims")
            return rw16(c) + rw16(cells) + cells * rw8(c), cells * rw8(c), 4 * rw16(c) + 4 * rw16(cells)
        if p0 == F.TAPMOD_ENERGY:
            j = p1
            if not 1 <= j <= 5 or cells < 1 or c < 1 or nf < 1 or c > 256:
                raise ValueError("ENERGY dims")
            return rw16(cells) + j * cells * rw8(c), nf * j * 2 + nf, 8 * nf * j + 4 * nf + 4 * rw16(cells)
        if p0 == F.TAPMOD_NU:
            j = n
            if not 1 <= j <= 16 or nf < 1:
                raise ValueError("NU dims")
            return nf * 3 * j, nf * rw8(j), 0
        if p0 == F.TAPMOD_MIX:
            j = k
            if not 1 <= j <= 5 or cells < 1 or c < 1 or c > 256:
                raise ValueError("MIX dims")
            return rw16(j) + cells * j * rw8(c), cells * rw8(c), 4 * rw16(j) + 4 * c
        raise KeyError("TAPMOD subop")
    if op == F.OP_HEAD:
        if p0 == F.HEAD_FORECAST:
            rows, nf, du = m, k, n
            row0, width = p1, p2
            if not (1 <= nf <= F.F_CAP) or width != F.C_TRANSPORT or rows < 1 or du < 1 or row0 + rows > nf:
                raise ValueError("FORECAST dims")
            res = nf * (rw8(F.KEY_D) + rw8(du) + rw8(width))
            per = F.F_CAP // 32 + rw8(width) + 2
            return res + rows * per, rows * (rw8(width) + 1), 4 * res
        if p0 == F.HEAD_SCORE:
            cells = m
            if cells < 1:
                raise ValueError("SCORE dims")
            return 2 * rw16(cells), rw16(cells) + _words_for(cells, 32), 4 * rw16(cells) + 4 * _words_for(cells, 32)
        raise KeyError("HEAD subop")
    raise KeyError("opcode")


# ----------------------------------------------------------------- execution

@dataclass
class UnitResult:
    status: int
    out: List[int]
    sat: int = 0
    aux0: int = 0
    aux1: int = 0


class _Reader:
    def __init__(self, words: Sequence[int]):
        self.w = list(words)
        self.i = 0

    def take(self, n: int) -> List[int]:
        if self.i + n > len(self.w):
            raise ValueError("payload underrun")
        s = self.w[self.i:self.i + n]
        self.i += n
        return s


def _sat_count(x: np.ndarray, lo: int, hi: int) -> int:
    return int(np.count_nonzero((x < lo) | (x > hi)))


def execute(d: Dict[str, int], payload: Sequence[int]) -> UnitResult:
    op, m, n, k = d["opcode"], d["dim_m"], d["dim_n"], d["dim_k"]
    p0, p1, p2 = d["param0"], d["param1"], d["param2"]
    rd = _Reader(payload)
    sat = 0

    if op == F.OP_GEMM:
        a16 = bool(p0 & F.GEMM_F_A16)
        b = unpack_rows(rd.take(k * row_words(n, "8")), k, n, "8", True)
        bias = unpack(rd.take(n), n, "32", True) if p0 & F.GEMM_F_BIAS else np.zeros(n, dtype=np.int64)
        shifts = unpack(rd.take(row_words(n, "8")), n, "8", False) if p0 & F.GEMM_F_PERCH else np.full(n, p1, dtype=np.int64)
        a = unpack_rows(rd.take(m * row_words(k, "16" if a16 else "8")), m, k, "16" if a16 else "8", True)
        acc = a @ b + bias[None, :]
        if p0 & F.GEMM_F_OUT32:
            sat += _sat_count(acc, -(1 << 31), (1 << 31) - 1)
            return UnitResult(F.ST_OK, pack_rows(np.clip(acc, -(1 << 31), (1 << 31) - 1), "32"), sat)
        lo, hi, kind = ((-32768, 32767, "16") if p0 & F.GEMM_F_OUT16 else ((-127, 127, "8") if p0 & F.GEMM_F_SYM else (-128, 127, "8")))
        y = np.stack([rshift_rne_np(acc[:, j], int(shifts[j])) for j in range(n)], axis=1)
        sat += _sat_count(y, lo, hi)
        return UnitResult(F.ST_OK, pack_rows(np.clip(y, lo, hi), kind), sat)

    if op == F.OP_CONV:
        hin, win_ = _hw(m)
        cout, cin = n, k
        kk, stride = p2 & 0xFF, (p2 >> 8) & 0xFF
        hout, wout = (hin - kk) // stride + 1, (win_ - kk) // stride + 1
        wts = unpack_rows(rd.take(cout * row_words(kk * kk * cin, "8")), cout, kk * kk * cin, "8", True).reshape(cout, kk, kk, cin)
        bias = unpack(rd.take(cout), cout, "32", True) if p0 & F.CONV_F_BIAS else np.zeros(cout, dtype=np.int64)
        shifts = unpack(rd.take(row_words(cout, "8")), cout, "8", False) if p0 & F.CONV_F_PERCH else np.full(cout, p1, dtype=np.int64)
        mod = bool(p0 & F.CONV_F_MOD)
        gamma = unpack(rd.take(row_words(cin, "16")), cin, "16", False) if mod else None
        x = np.zeros((hin, win_, cin), dtype=np.int64)
        aff = np.zeros((hin, win_, 8), dtype=np.int64)
        for y in range(hin):
            x[y] = unpack(rd.take(row_words(win_ * cin, "8")), win_ * cin, "8", True).reshape(win_, cin)
            if mod:
                aff[y] = unpack(rd.take(row_words(win_ * 8, "16")), win_ * 8, "16", False).reshape(win_, 8)
        out_rows = []
        lo, hi = (-127, 127) if p0 & F.CONV_F_SYM else (-128, 127)
        dir_of = {(-1, -1): 0, (-1, 0): 1, (-1, 1): 2, (0, -1): 3, (0, 1): 4, (1, -1): 5, (1, 0): 6, (1, 1): 7}
        for oy in range(hout):
            row = np.zeros((wout, cout), dtype=np.int64)
            for ox in range(wout):
                y0, x0 = oy * stride, ox * stride
                win = x[y0:y0 + kk, x0:x0 + kk].copy()
                if mod:
                    cy, cx = y0 + 1, x0 + 1
                    for ky in range(3):
                        for kx in range(3):
                            if ky == 1 and kx == 1:
                                continue
                            wd = int(aff[cy, cx, dir_of[(ky - 1, kx - 1)]])
                            a_code = rshift_rne_np(((1 << 14) - gamma) * (1 << 15) + gamma * wd, 15)
                            xm = rshift_rne_np(a_code * win[ky, kx], 14)
                            sat += _sat_count(xm, -128, 127)
                            win[ky, kx] = np.clip(xm, -128, 127)
                acc = np.einsum("oyxc,yxc->o", wts, win) + bias
                yv = np.array([rshift_rne(int(acc[o]), int(shifts[o])) for o in range(cout)], dtype=np.int64)
                sat += _sat_count(yv, lo, hi)
                row[ox] = np.clip(yv, lo, hi)
            out_rows.append(row.ravel())
        return UnitResult(F.ST_OK, pack_rows(np.asarray(out_rows), "8"), sat)

    if op == F.OP_STENCIL:
        h, w = _hw(m)
        c, steps, halo = n, k, p2
        ho, wo = h - 2 * halo, w - 2 * halo
        bands = bool(p0 & F.STENCIL_F_BANDS)
        in16 = bool(p0 & F.STENCIL_F_IN16)
        taps = np.zeros((h * w, 9), dtype=np.int64)
        for i in range(h * w):
            taps[i] = unpack(rd.take(5), 9, "16", True)
        taps = taps.reshape(h, w, 9)
        betas = None
        if bands:
            betas = np.zeros((h * w, 4), dtype=np.int64)
            for i in range(h * w):
                betas[i] = unpack(rd.take(2), 4, "16", False)
            betas = betas.reshape(h, w, 4)
        exps = [(p1 >> (4 * j)) & 0xF for j in range(5)]
        out = []
        for ch in range(c):
            plane = unpack(rd.take(row_words(h * w, "16" if in16 else "8")), h * w, "16" if in16 else "8", True).reshape(h, w)
            xs = plane if in16 else (plane << 8)
            pw = [xs[halo:h - halo, halo:w - halo].copy()]
            cur = xs[..., None]
            for _ in range(steps):
                cur = walk_step(cur, taps)
                pw.append(cur[halo:h - halo, halo:w - halo, 0].copy())
            pw = np.stack(pw, axis=0)
            if not bands:
                out += pack_u16(pw[-1].ravel())
                continue
            dco, low = band_coefficients(betas[halo:h - halo, halo:w - halo], steps)
            for j in range(4):
                acc = np.einsum("hwt,thw->hw", dco[:, :, j, :], pw)
                y = rshift_rne_np(acc, 23 - exps[j])
                sat += _sat_count(y, -128, 127)
                out += pack_u8(np.clip(y, -128, 127).ravel())
            acc = np.einsum("hwt,thw->hw", low, pw)
            y = rshift_rne_np(acc, 23 - exps[4])
            sat += _sat_count(y, -128, 127)
            out += pack_u8(np.clip(y, -128, 127).ravel())
        return UnitResult(F.ST_OK, out, sat)

    if op == F.OP_LUT:
        if p0 == F.LUT_TABLE8:
            table = unpack(rd.take(64), 256, "8", True)
            x = unpack(rd.take(row_words(m, "8")), m, "8", True)
            return UnitResult(F.ST_OK, pack_u8(table[x + 128]))
        if p0 == F.LUT_SIGMOID:
            x = unpack(rd.take(row_words(m, "16")), m, "16", True)
            idx = (np.clip(x, F.SIG_IN_MIN, F.SIG_IN_MAX) - F.SIG_IN_MIN) >> F.SIG_IDX_SHIFT
            sat += _sat_count(x, F.SIG_IN_MIN, F.SIG_IN_MAX)
            return UnitResult(F.ST_OK, pack_u16(np.asarray(F.SIG_TABLE, dtype=np.int64)[idx]), sat)
        if p0 == F.LUT_EXPF:
            x = unpack(rd.take(row_words(m, "16")), m, "16", True)
            sat += _sat_count(x, F.FC_LOGIT_MIN, F.FC_LOGIT_MAX)
            xi = np.clip(x, F.FC_LOGIT_MIN, F.FC_LOGIT_MAX) - F.FC_LOGIT_MIN
            return UnitResult(F.ST_OK, pack_u32(np.asarray(F.FC_EXP, dtype=np.int64)[xi]), sat)
        if p0 == F.LUT_REQUANT32:
            parts = [unpack(rd.take(m), m, "32", True) for _ in range(k)]
            acc = np.zeros(m, dtype=np.int64)
            for pt in parts:
                acc = acc + pt
            y = rshift_rne_np(acc, p1)
            lo, hi, kind = ((-32768, 32767, "16") if p2 & 2 else ((-127, 127, "8") if p2 & 1 else (-128, 127, "8")))
            sat += _sat_count(y, lo, hi)
            return UnitResult(F.ST_OK, pack_rows(np.clip(y, lo, hi)[None, :], kind), sat)

    if op == F.OP_PERMUTE:
        nout, rw, nin = m, n, k
        idx = unpack(rd.take(row_words(nout, "16")), nout, "16", False)
        rows = [rd.take(rw) for _ in range(nin)]
        out, bad = [], 0
        for i in idx:
            if i == 0xFFFF or i >= nin:
                out += [0] * rw
                bad += int(i != 0xFFFF)
            else:
                out += list(rows[int(i)])
        return UnitResult(F.ST_OK, out, 0, bad)

    if op == F.OP_SEGOPS:
        cells, c, nf = m, n, k
        if p0 == F.SEGOPS_SEGSUM:
            ids = unpack(rd.take(row_words(cells, "16")), cells, "16", False)
            wts = unpack(rd.take(row_words(cells, "8")), cells, "8", False)
            x = unpack_rows(rd.take(cells * row_words(c, "8")), cells, c, "8", True)
            sums = np.zeros((nf, c), dtype=np.int64)
            ws = np.zeros(nf, dtype=np.int64)
            bad = 0
            for i in range(cells):
                f = int(ids[i])
                if f == F.UNASSIGNED:
                    continue
                if f >= nf:
                    bad += 1
                    continue
                sums[f] += wts[i] * x[i]
                ws[f] += wts[i]
            sat += _sat_count(sums, -(1 << 31), (1 << 31) - 1)
            return UnitResult(F.ST_OK, pack_rows(np.clip(sums, -(1 << 31), (1 << 31) - 1), "32") + pack_u32(ws), sat, bad)
        if p0 == F.SEGOPS_SEGMEAN:
            ws = unpack(rd.take(nf), nf, "32", True)
            sums = unpack_rows(rd.take(nf * c), nf, c, "32", True)
            hrows = np.zeros((nf, c), dtype=np.int64)
            for f in range(nf):
                if ws[f] > 0:
                    hrows[f] = [div_rne(int(v), int(ws[f])) for v in sums[f]]
            sat += _sat_count(hrows, -128, 127)
            return UnitResult(F.ST_OK, pack_rows(np.clip(hrows, -128, 127), "8"), sat)
        if p0 in (F.SEGOPS_FILL, F.SEGOPS_DETAIL, F.SEGOPS_SYNTH, F.SEGOPS_CLSBAND):
            hrows = unpack_rows(rd.take(nf * row_words(c, "8")), nf, c, "8", True)
            rz = unpack_rows(rd.take(nf * row_words(c, "8")), nf, c, "8", True) if p0 == F.SEGOPS_CLSBAND else None
            ids = unpack(rd.take(row_words(cells, "16")), cells, "16", False)
            out = []
            bad = 0
            for i in range(cells):
                f = int(ids[i])
                ok = f != F.UNASSIGNED and f < nf
                bad += int(f != F.UNASSIGNED and f >= nf)
                hv = hrows[f] if ok else np.zeros(c, dtype=np.int64)
                if p0 == F.SEGOPS_FILL:
                    out += pack_u8(hv)
                elif p0 == F.SEGOPS_DETAIL:
                    xv = unpack(rd.take(row_words(c, "8")), c, "8", True)
                    out += pack_u16(xv - hv)
                elif p0 == F.SEGOPS_SYNTH:
                    xv = unpack(rd.take(row_words(c, "8")), c, "8", True)
                    rv = unpack(rd.take(row_words(c, "8")), c, "8", True) if p1 & 1 else 0
                    y = xv + hv + rv
                    sat += _sat_count(y, -128, 127)
                    out += pack_u8(np.clip(y, -128, 127))
                else:
                    if ok:
                        y = hrows[f] + rz[f]
                        sat += _sat_count(y, -128, 127)
                        out += pack_u8(np.clip(y, -128, 127))
                    else:
                        out += pack_u8(np.zeros(c, dtype=np.int64))
            return UnitResult(F.ST_OK, out, sat, bad)

    if op == F.OP_TAPMOD:
        cells, c, nf = m, n, k
        if p0 == F.TAPMOD_MODULATE:
            gamma = unpack(rd.take(row_words(c, "16")), c, "16", False)
            aff = unpack(rd.take(row_words(cells, "16")), cells, "16", False)
            out = []
            for i in range(cells):
                xv = unpack(rd.take(row_words(c, "8")), c, "8", True)
                a_code = rshift_rne_np(((1 << 14) - gamma) * (1 << 15) + gamma * int(aff[i]), 15)
                y = rshift_rne_np(a_code * xv, 14)
                sat += _sat_count(y, -128, 127)
                out += pack_u8(np.clip(y, -128, 127))
            return UnitResult(F.ST_OK, out, sat)
        if p0 == F.TAPMOD_ENERGY:
            j = p1
            ids = unpack(rd.take(row_words(cells, "16")), cells, "16", False)
            sums = [[0] * j for _ in range(nf)]
            cnts = [0] * nf
            for jj in range(j):
                for i in range(cells):
                    xv = unpack(rd.take(row_words(c, "8")), c, "8", True)
                    f = int(ids[i])
                    if f != F.UNASSIGNED and f < nf:
                        sums[f][jj] += int(np.dot(xv, xv))
                        if jj == 0:
                            cnts[f] += c
            flat = [sums[f][jj] for f in range(nf) for jj in range(j)]
            return UnitResult(F.ST_OK, pack_i64(flat) + pack_u32(cnts))
        if p0 == F.TAPMOD_NU:
            j = n
            out = []
            for f in range(nf):
                blk = rd.take(3 * j)
                sums = unpack(blk[:2 * j], j, "64", True)
                cnts = unpack(blk[2 * j:], j, "32", True)
                row = []
                for jj in range(j):
                    cnt_ = int(cnts[jj])
                    if cnt_ <= 0:
                        row.append(0)
                    else:
                        v = rshift_rne(div_rne(int(sums[jj]), cnt_), p1)
                        sat += int(v > 127)
                        row.append(min(127, v))
                out += pack_u8(row)
            return UnitResult(F.ST_OK, out, sat)
        if p0 == F.TAPMOD_MIX:
            j = k
            gains = unpack(rd.take(row_words(j, "16")), j, "16", True)
            out = []
            for i in range(cells):
                acc = np.zeros(c, dtype=np.int64)
                for jj in range(j):
                    acc += gains[jj] * unpack(rd.take(row_words(c, "8")), c, "8", True)
                y = rshift_rne_np(acc, p1)
                sat += _sat_count(y, -128, 127)
                out += pack_u8(np.clip(y, -128, 127))
            return UnitResult(F.ST_OK, out, sat)

    if op == F.OP_HEAD:
        if p0 == F.HEAD_FORECAST:
            from .forecast import forecast_row
            rows, nf, du, row0, width = m, k, n, p1, p2
            K = unpack_rows(rd.take(nf * row_words(F.KEY_D, "8")), nf, F.KEY_D, "8", True)
            Uv = unpack_rows(rd.take(nf * row_words(du, "8")), nf, du, "8", True)
            VO = unpack_rows(rd.take(nf * row_words(width, "8")), nf, width, "8", True)
            out = []
            n_abst = 0
            for r in range(rows):
                f = row0 + r
                mask_words = rd.take(F.F_CAP // 32)
                hs = unpack(rd.take(row_words(width, "8")), width, "8", True)
                null_raw = unpack(rd.take(1), 1, "32", True)[0]
                alpha_raw = unpack(rd.take(1), 1, "32", True)[0]
                adm = [bool((int(mask_words[g >> 5]) >> (g & 31)) & 1) for g in range(nf)]
                hs_full = np.zeros((nf, width), dtype=np.int64)
                if f < nf:
                    hs_full[f] = hs
                dh, ab = forecast_row(f, K, Uv, VO, hs_full, int(null_raw), int(alpha_raw), adm)
                n_adm = sum(1 for g in range(nf) if adm[g] and g != f)
                n_abst += int(ab)
                out += pack_u8(dh) + [n_adm | (int(ab) << 16)]
            return UnitResult(F.ST_OK, out, 0, n_abst)
        if p0 == F.HEAD_SCORE:
            cells = m
            lg = unpack(rd.take(row_words(cells, "16")), cells, "16", True)
            gt = unpack(rd.take(row_words(cells, "16")), cells, "16", False)
            from .readout import class_scores
            p = class_scores(lg, np.minimum(gt, 32767))
            bits = [0] * _words_for(cells, 32)
            ncand = 0
            for i in range(cells):
                if p[i] >= p1:
                    bits[i >> 5] |= 1 << (i & 31)
                    ncand += 1
            return UnitResult(F.ST_OK, pack_u16(p) + bits, 0, ncand)
    return UnitResult(F.ST_BAD_SUBOP, [])


def run_stream(words: Sequence[int], body_mask: int = 0xFF) -> List[int]:
    """Process a concatenation of commands exactly as one worker does.

    body_mask: the worker class (bit i enables opcode i, SV_BODY_MASK in the AIE build); a
    body that is not built answers BAD_OPCODE after its payload is consumed."""
    out: List[int] = []
    i = 0
    words = [int(w) & M32 for w in words]
    while i < len(words):
        hdr = words[i:i + F.DESC_WORDS]
        if len(hdr) < F.DESC_WORDS:
            raise ValueError("truncated descriptor")
        i += F.DESC_WORDS
        d = decode_fields(hdr, F.DESC_FIELDS)
        if d["magic"] != F.CMD_MAGIC:
            out += frame(d, F.ST_BAD_MAGIC, [])
            continue
        if d["check"] != check_word(hdr[:F.DESC_WORDS - 1]):
            out += frame(d, F.ST_BAD_CHECK, [])
            continue
        if d["version"] != F.PROTO_VERSION:
            out += frame(d, F.ST_BAD_VERSION, [])
            continue
        payload = words[i:i + d["in_words"]]
        i += d["in_words"]
        if d["opcode"] == F.OP_NOP:
            out += frame(d, F.ST_OK, [])
            continue
        if d["opcode"] <= 7 and not (body_mask >> d["opcode"]) & 1:
            out += frame(d, F.ST_BAD_OPCODE, [])
            continue
        try:
            win, wout, scratch = shape_words(d)
        except KeyError:
            st = F.ST_BAD_OPCODE if d["opcode"] > 7 else F.ST_BAD_SUBOP
            out += frame(d, st, [])
            continue
        except ValueError:
            out += frame(d, F.ST_BAD_SHAPE, [])
            continue
        if win != d["in_words"] or wout != d["out_words"]:
            out += frame(d, F.ST_BAD_SHAPE, [])
            continue
        if scratch > F.SCRATCH_BYTES:
            out += frame(d, F.ST_SCRATCH, [])
            continue
        res = execute(d, payload)
        if len(res.out) & 1:
            res.out = list(res.out) + [0]
        if res.status != F.ST_OK:
            out += frame(d, res.status, [])
            continue
        if len(res.out) != wout:
            raise AssertionError(f"reference produced {len(res.out)} words, expected {wout}")
        out += frame(d, F.ST_OK, res.out, res.sat, res.aux0, res.aux1)
    return out


def split_commands(words: Sequence[int]) -> List[List[int]]:
    """Split a command stream into units exactly as a worker consumes it: a descriptor whose
    magic, check word or version is wrong carries no payload (framing is lost)."""
    out: List[List[int]] = []
    i = 0
    words = [int(w) & M32 for w in words]
    while i < len(words):
        hdr = words[i:i + F.DESC_WORDS]
        if len(hdr) < F.DESC_WORDS:
            raise ValueError("truncated descriptor")
        d = decode_fields(hdr, F.DESC_FIELDS)
        n = 0
        if d["magic"] == F.CMD_MAGIC and d["check"] == check_word(hdr[:F.DESC_WORDS - 1]) and d["version"] == F.PROTO_VERSION:
            n = d["in_words"]
        out.append(words[i:i + F.DESC_WORDS + n])
        i += F.DESC_WORDS + n
    return out


def predict_status(cmd: Sequence[int], body_mask: int = 0xFF) -> int:
    """Status a worker will report for one command, from the pre-execution checks only (no
    body can fail once these pass), so a host can size a counted receive exactly."""
    hdr = [int(w) & M32 for w in cmd[:F.DESC_WORDS]]
    d = decode_fields(hdr, F.DESC_FIELDS)
    if d["magic"] != F.CMD_MAGIC:
        return F.ST_BAD_MAGIC
    if d["check"] != check_word(hdr[:F.DESC_WORDS - 1]):
        return F.ST_BAD_CHECK
    if d["version"] != F.PROTO_VERSION:
        return F.ST_BAD_VERSION
    if d["opcode"] == F.OP_NOP:
        return F.ST_OK
    if d["opcode"] <= 7 and not (body_mask >> d["opcode"]) & 1:
        return F.ST_BAD_OPCODE
    try:
        win, wout, scratch = shape_words(d)
    except KeyError:
        return F.ST_BAD_OPCODE if d["opcode"] > 7 else F.ST_BAD_SUBOP
    except ValueError:
        return F.ST_BAD_SHAPE
    if win != d["in_words"] or wout != d["out_words"]:
        return F.ST_BAD_SHAPE
    if scratch > F.SCRATCH_BYTES:
        return F.ST_SCRATCH
    return F.ST_OK


def response_length(cmd: Sequence[int], body_mask: int = 0xFF) -> int:
    """Words of the response one command produces (header + payload + trailer)."""
    d = decode_fields([int(w) & M32 for w in cmd[:F.DESC_WORDS]], F.DESC_FIELDS)
    ok = predict_status(cmd, body_mask) == F.ST_OK and d["opcode"] != F.OP_NOP
    return F.RSP_WORDS + (d["out_words"] if ok else 0) + F.TRL_WORDS


def command(desc: Desc, payload: Sequence[int]) -> List[int]:
    """Fill in_words/out_words from the shape formulas and return descriptor + payload."""
    d = dict(desc.__dict__)
    d["magic"] = F.CMD_MAGIC
    win, wout, _ = shape_words(d)
    payload = [int(x) & M32 for x in payload]
    if len(payload) & 1:
        payload.append(0)
    if win != len(payload):
        raise ValueError(f"payload has {len(payload)} words, shape needs {win}")
    desc.in_words, desc.out_words = win, wout
    return desc.words() + payload
