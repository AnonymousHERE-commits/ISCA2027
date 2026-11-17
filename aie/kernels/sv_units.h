// SPLICE-V dispatcher: one worker executes one unit per invocation (SPLICE-V §3 shared workers).
//
//   in-stream : 16-word descriptor, then 'in_words' payload words
//   out-stream: 8-word header, 'out_words' payload words, 4-word trailer (counters)
//
// Eight bodies (SPLICE-V Fig. 3(b)): CONV, GEMM, STENCIL, LUT, PERMUTE, SEGOPS, TAPMOD, HEAD.
// This is the scalar, bit-exact path; it is the twin of reference/splicev_ref/units.py and
// compiles both for the AIE-ML core (IO = AIE streams) and on the host (IO = vectors).
// Vectorised AIE API paths must reproduce these outputs exactly (gate G05).
#ifndef SV_UNITS_H
#define SV_UNITS_H

#include <stdint.h>

#include "splicev_gen.h"
#include "sv_aie_int.h"

// Worker classes: a build may include a subset of bodies to fit the 16 KB program memory
// (roadmap 7.1).  Bit i enables opcode i; a disabled opcode answers BAD_OPCODE.
#ifndef SV_BODY_MASK
#define SV_BODY_MASK 0xFFu
#endif

namespace sva {

using namespace svgen;

struct UnitStats {
    uint32_t sat;
    uint32_t aux0;
    uint32_t aux1;
};

// Counts words so the dispatcher can drain/pad to the descriptor's lengths.
template <class IO>
struct CountIO {
    IO& io;
    uint32_t nread;
    uint32_t nwritten;
    explicit CountIO(IO& x) : io(x), nread(0), nwritten(0) {}
    uint32_t read() { ++nread; return io.read(); }
    void write(uint32_t w) { ++nwritten; io.write(w); }
    void read_n(uint32_t* dst, uint32_t n) { for (uint32_t i = 0; i < n; ++i) dst[i] = read(); }
};

// Packs 8/16-bit values into words; flush() closes a row (row-padded layout).
template <class W>
struct Pack8 {
    W& io;
    uint32_t acc;
    int n;
    explicit Pack8(W& x) : io(x), acc(0), n(0) {}
    void put(int32_t v) {
        acc |= ((uint32_t)v & 0xFFu) << (8 * n);
        if (++n == 4) { io.write(acc); acc = 0; n = 0; }
    }
    void flush() { if (n) { io.write(acc); acc = 0; n = 0; } }
};

template <class W>
struct Pack16 {
    W& io;
    uint32_t acc;
    int n;
    explicit Pack16(W& x) : io(x), acc(0), n(0) {}
    void put(int32_t v) {
        acc |= ((uint32_t)v & 0xFFFFu) << (16 * n);
        if (++n == 2) { io.write(acc); acc = 0; n = 0; }
    }
    void flush() { if (n) { io.write(acc); acc = 0; n = 0; } }
};

struct Arena {
    uint32_t* base;
    uint32_t used;
    uint32_t* take(uint64_t bytes) {
        uint32_t* p = base + used;
        used += (uint32_t)(a4(bytes) / 4);
        return p;
    }
};

struct Desc {
    uint32_t w[16];
    uint32_t f(int word, int lsb, int width) const {
        return width == 32 ? w[word] : (w[word] >> lsb) & ((1u << width) - 1u);
    }
    uint32_t magic() const { return f(DESC_MAGIC_W, DESC_MAGIC_LSB, DESC_MAGIC_WID); }
    uint32_t version() const { return f(DESC_VERSION_W, DESC_VERSION_LSB, DESC_VERSION_WID); }
    uint32_t frame_id() const { return f(DESC_FRAME_ID_W, DESC_FRAME_ID_LSB, DESC_FRAME_ID_WID); }
    uint32_t slot() const { return f(DESC_SLOT_W, DESC_SLOT_LSB, DESC_SLOT_WID); }
    uint32_t opcode() const { return f(DESC_OPCODE_W, DESC_OPCODE_LSB, DESC_OPCODE_WID); }
    uint32_t worker() const { return f(DESC_WORKER_W, DESC_WORKER_LSB, DESC_WORKER_WID); }
    uint32_t m() const { return w[DESC_DIM_M_W]; }
    uint32_t n() const { return w[DESC_DIM_N_W]; }
    uint32_t k() const { return w[DESC_DIM_K_W]; }
    uint32_t in_words() const { return w[DESC_IN_WORDS_W]; }
    uint32_t out_words() const { return w[DESC_OUT_WORDS_W]; }
    uint32_t p0() const { return w[DESC_PARAM0_W]; }
    uint32_t p1() const { return w[DESC_PARAM1_W]; }
    uint32_t p2() const { return w[DESC_PARAM2_W]; }
    uint32_t token() const { return w[DESC_TOKEN_W]; }
    uint32_t check() const { return w[DESC_CHECK_W]; }
};

enum ShapeErr { SH_OK = 0, SH_VALUE = 1, SH_KEY = 2 };

static inline uint64_t ev(uint64_t x) { return x + (x & 1); }

// Mirror of units._shape_words: lengths (unpadded) and scratch bytes, or the error class.
static inline int shape_of(const Desc& d, uint64_t& win, uint64_t& wout, uint64_t& scratch) {
    const uint64_t op = d.opcode(), m = d.m(), n = d.n(), k = d.k();
    const uint32_t p0 = d.p0(), p1 = d.p1(), p2 = d.p2();
    win = wout = scratch = 0;
    if (op == (uint64_t)OP_GEMM) {
        if (m < 1 || n < 1 || k < 1) return SH_VALUE;
        bool a16 = p0 & GEMM_F_A16;
        uint64_t rwo = (p0 & GEMM_F_OUT32) ? n : ((p0 & GEMM_F_OUT16) ? rw16(n) : rw8(n));
        win = k * rw8(n) + ((p0 & GEMM_F_BIAS) ? n : 0) + ((p0 & GEMM_F_PERCH) ? rw8(n) : 0) + m * (a16 ? rw16(k) : rw8(k));
        wout = m * rwo;
        scratch = 4 * (k * rw8(n)) + 4 * n + 4 * (uint64_t)rw8(n) + 4 * (uint64_t)rw16(k);
        return SH_OK;
    }
    if (op == (uint64_t)OP_CONV) {
        uint64_t hin = m & 0xFFFF, wi = m >> 16, cout = n, cin = k, kk = p2 & 0xFF, st = (p2 >> 8) & 0xFF;
        if (kk < 1 || st < 1 || hin < kk || wi < kk || cout < 1 || cin < 1) return SH_VALUE;
        bool mod = p0 & CONV_F_MOD;
        if (mod && kk != 3) return SH_VALUE;
        uint64_t hout = (hin - kk) / st + 1, wo = (wi - kk) / st + 1;
        uint64_t w_words = cout * rw8(kk * kk * cin);
        uint64_t per_row = rw8(wi * cin) + (mod ? rw16(wi * 8) : 0);
        win = w_words + ((p0 & CONV_F_BIAS) ? cout : 0) + ((p0 & CONV_F_PERCH) ? rw8(cout) : 0) + (mod ? rw16(cin) : 0) + hin * per_row;
        wout = hout * rw8(wo * cout);
        scratch = 4 * w_words + 4 * cout + 4 * (uint64_t)rw8(cout) + 4 * (uint64_t)rw16(cin) + 4 * kk * per_row + 4 * (uint64_t)rw8(wo * cout);
        return SH_OK;
    }
    if (op == (uint64_t)OP_STENCIL) {
        uint64_t h = m & 0xFFFF, w = m >> 16, c = n, steps = k, halo = p2;
        uint64_t mn = h < w ? h : w;
        if (h < 1 || w < 1 || c < 1 || steps > 15 || 2 * halo >= mn) return SH_VALUE;
        uint64_t ho = h - 2 * halo, wo = w - 2 * halo;
        bool bands = p0 & STENCIL_F_BANDS, in16 = p0 & STENCIL_F_IN16;
        win = h * w * 5 + (bands ? h * w * 2 : 0) + c * (in16 ? rw16(h * w) : rw8(h * w));
        wout = c * (bands ? 5 * (uint64_t)rw8(ho * wo) : rw16(ho * wo));
        scratch = a4(18 * h * w) + (bands ? a4(8 * ho * wo) : 0) + a4(4 * h * w) + a4(2 * (steps + 1) * ho * wo);
        return SH_OK;
    }
    if (op == (uint64_t)OP_LUT) {
        if (m < 1) return SH_VALUE;
        if (p0 == (uint32_t)LUT_TABLE8) { win = 64 + rw8(m); wout = rw8(m); scratch = 256; return SH_OK; }
        if (p0 == (uint32_t)LUT_SIGMOID) { win = rw16(m); wout = rw16(m); return SH_OK; }
        if (p0 == (uint32_t)LUT_EXPF) { win = rw16(m); wout = m; return SH_OK; }
        if (p0 == (uint32_t)LUT_REQUANT32) {
            if (k < 1) return SH_VALUE;
            win = k * m; wout = (p2 & 2) ? rw16(m) : rw8(m); scratch = 8 * m;
            return SH_OK;
        }
        return SH_KEY;
    }
    if (op == (uint64_t)OP_PERMUTE) {
        if (m < 1 || n < 1 || k < 1) return SH_VALUE;
        win = rw16(m) + k * n; wout = m * n; scratch = 4 * (uint64_t)rw16(m) + 4 * k * n;
        return SH_OK;
    }
    if (op == (uint64_t)OP_SEGOPS) {
        uint64_t cells = m, c = n, nf = k;
        if (cells < 1 || c < 1 || nf < 1 || c > 256) return SH_VALUE;
        if (p0 == (uint32_t)SEGOPS_SEGSUM) {
            win = rw16(cells) + rw8(cells) + cells * rw8(c); wout = nf * c + nf;
            scratch = 4 * nf * c + 4 * nf + 4 * (uint64_t)rw16(cells) + 4 * (uint64_t)rw8(cells);
            return SH_OK;
        }
        if (p0 == (uint32_t)SEGOPS_SEGMEAN) { win = nf + nf * c; wout = nf * rw8(c); scratch = 4 * nf; return SH_OK; }
        if (p0 == (uint32_t)SEGOPS_FILL) { win = nf * rw8(c) + rw16(cells); wout = cells * rw8(c); scratch = 4 * nf * rw8(c); return SH_OK; }
        if (p0 == (uint32_t)SEGOPS_DETAIL) {
            win = nf * rw8(c) + rw16(cells) + cells * rw8(c); wout = cells * rw16(c);
            scratch = 4 * nf * rw8(c) + 4 * (uint64_t)rw16(cells);
            return SH_OK;
        }
        if (p0 == (uint32_t)SEGOPS_SYNTH) {
            uint64_t reps = (p1 & 1) ? 2 : 1;
            win = nf * rw8(c) + rw16(cells) + reps * cells * rw8(c); wout = cells * rw8(c);
            scratch = 4 * nf * rw8(c) + 4 * (uint64_t)rw16(cells);
            return SH_OK;
        }
        if (p0 == (uint32_t)SEGOPS_CLSBAND) { win = 2 * nf * rw8(c) + rw16(cells); wout = cells * rw8(c); scratch = 8 * nf * rw8(c); return SH_OK; }
        return SH_KEY;
    }
    if (op == (uint64_t)OP_TAPMOD) {
        uint64_t cells = m, c = n, nf = k;
        if (p0 == (uint32_t)TAPMOD_MODULATE) {
            if (cells < 1 || c < 1 || c > 256) return SH_VALUE;
            win = rw16(c) + rw16(cells) + cells * rw8(c); wout = cells * rw8(c);
            scratch = 4 * (uint64_t)rw16(c) + 4 * (uint64_t)rw16(cells);
            return SH_OK;
        }
        if (p0 == (uint32_t)TAPMOD_ENERGY) {
            uint64_t j = p1;
            if (j < 1 || j > 5 || cells < 1 || c < 1 || nf < 1 || c > 256) return SH_VALUE;
            win = rw16(cells) + j * cells * rw8(c); wout = nf * j * 2 + nf;
            scratch = 8 * nf * j + 4 * nf + 4 * (uint64_t)rw16(cells);
            return SH_OK;
        }
        if (p0 == (uint32_t)TAPMOD_NU) {
            uint64_t j = n;
            if (j < 1 || j > 16 || nf < 1) return SH_VALUE;
            win = nf * 3 * j; wout = nf * rw8(j);
            return SH_OK;
        }
        if (p0 == (uint32_t)TAPMOD_MIX) {
            uint64_t j = k;
            if (j < 1 || j > 5 || cells < 1 || c < 1 || c > 256) return SH_VALUE;
            win = rw16(j) + cells * j * rw8(c); wout = cells * rw8(c); scratch = 4 * (uint64_t)rw16(j) + 4 * c;
            return SH_OK;
        }
        return SH_KEY;
    }
    if (op == (uint64_t)OP_HEAD) {
        if (p0 == (uint32_t)HEAD_FORECAST) {
            uint64_t rows = m, nf = k, du = n, row0 = p1, width = p2;
            if (nf < 1 || nf > (uint64_t)F_CAP || width != (uint64_t)C_TRANSPORT || rows < 1 || du < 1 || row0 + rows > nf) return SH_VALUE;
            uint64_t res = nf * (rw8(KEY_D) + rw8(du) + rw8(width));
            uint64_t per = F_CAP / 32 + rw8(width) + 2;
            win = res + rows * per; wout = rows * (rw8(width) + 1); scratch = 4 * res;
            return SH_OK;
        }
        if (p0 == (uint32_t)HEAD_SCORE) {
            uint64_t cells = m;
            if (cells < 1) return SH_VALUE;
            win = 2 * (uint64_t)rw16(cells); wout = rw16(cells) + (cells + 31) / 32;
            scratch = 4 * (uint64_t)rw16(cells) + 4 * ((cells + 31) / 32);
            return SH_OK;
        }
        return SH_KEY;
    }
    return SH_KEY;
}

// ------------------------------------------------------------------ bodies

template <class C>
static void body_gemm(C& io, const Desc& d, Arena& ar, UnitStats& st) {
    const uint32_t m = d.m(), n = d.n(), k = d.k(), p0 = d.p0();
    const bool a16 = p0 & GEMM_F_A16;
    uint32_t* B = ar.take(4ull * k * rw8(n));
    uint32_t* bias = ar.take(4ull * n);
    uint32_t* sh = ar.take(4ull * rw8(n));
    uint32_t* row = ar.take(4ull * rw16(k));
    io.read_n(B, k * rw8(n));
    for (uint32_t j = 0; j < n; ++j) bias[j] = 0;
    if (p0 & GEMM_F_BIAS) io.read_n(bias, n);
    if (p0 & GEMM_F_PERCH) io.read_n(sh, rw8(n));
    const uint32_t rwb = rw8(n);
    Pack8<C> p8(io);
    Pack16<C> p16(io);
    int64_t lo = (p0 & GEMM_F_SYM) ? -127 : -128, hi = 127;
    for (uint32_t r = 0; r < m; ++r) {
        io.read_n(row, a16 ? rw16(k) : rw8(k));
        for (uint32_t j = 0; j < n; ++j) {
            int64_t acc = (int32_t)bias[j];
            for (uint32_t t = 0; t < k; ++t) {
                int32_t a = a16 ? get_i16(row, t) : get_i8(row, t);
                acc += (int64_t)a * get_i8(B + t * rwb, j);
            }
            if (p0 & GEMM_F_OUT32) {
                int64_t v = clampi(acc, -2147483647ll - 1, 2147483647ll);
                if (v != acc) ++st.sat;
                io.write((uint32_t)(int32_t)v);
                continue;
            }
            int s = (p0 & GEMM_F_PERCH) ? get_u8(sh, j) : (int)d.p1();
            int64_t y = rne(acc, s);
            if (p0 & GEMM_F_OUT16) {
                int64_t v = clampi(y, -32768, 32767);
                if (v != y) ++st.sat;
                p16.put((int32_t)v);
            } else {
                int64_t v = clampi(y, lo, hi);
                if (v != y) ++st.sat;
                p8.put((int32_t)v);
            }
        }
        p8.flush();
        p16.flush();
    }
}

template <class C>
static void body_conv(C& io, const Desc& d, Arena& ar, UnitStats& st) {
    const uint32_t hin = d.m() & 0xFFFF, wi = d.m() >> 16, cout = d.n(), cin = d.k(), p0 = d.p0();
    const uint32_t kk = d.p2() & 0xFF, stride = (d.p2() >> 8) & 0xFF;
    const uint32_t hout = (hin - kk) / stride + 1, wo = (wi - kk) / stride + 1;
    const bool mod = p0 & CONV_F_MOD;
    const uint32_t wrow = rw8(kk * kk * cin);
    const uint32_t xrow = rw8(wi * cin), arow = mod ? rw16(wi * 8) : 0, per_row = xrow + arow;
    uint32_t* W = ar.take(4ull * cout * wrow);
    uint32_t* bias = ar.take(4ull * cout);
    uint32_t* sh = ar.take(4ull * rw8(cout));
    uint32_t* gam = ar.take(4ull * rw16(cin));
    uint32_t* ring = ar.take(4ull * kk * per_row);
    ar.take(4ull * rw8(wo * cout));
    io.read_n(W, cout * wrow);
    for (uint32_t c = 0; c < cout; ++c) bias[c] = 0;
    if (p0 & CONV_F_BIAS) io.read_n(bias, cout);
    if (p0 & CONV_F_PERCH) io.read_n(sh, rw8(cout));
    if (mod) io.read_n(gam, rw16(cin));
    int64_t lo = (p0 & CONV_F_SYM) ? -127 : -128;
    // tap (ky, kx) -> affinity direction of the centre cell (manifest NEIGH order)
    static const int dir_of[9] = {0, 1, 2, 3, -1, 4, 5, 6, 7};
    Pack8<C> p8(io);
    for (uint32_t y = 0; y < hin; ++y) {
        uint32_t* slot = ring + (y % kk) * per_row;
        io.read_n(slot, per_row);
        if (y + 1 < kk || (y + 1 - kk) % stride != 0) continue;
        uint32_t y0 = y + 1 - kk;
        for (uint32_t ox = 0; ox < wo; ++ox) {
            uint32_t x0 = ox * stride;
            for (uint32_t co = 0; co < cout; ++co) {
                int64_t acc = (int32_t)bias[co];
                for (uint32_t ky = 0; ky < kk; ++ky) {
                    const uint32_t* xr = ring + ((y0 + ky) % kk) * per_row;
                    for (uint32_t kx = 0; kx < kk; ++kx) {
                        for (uint32_t ci = 0; ci < cin; ++ci) {
                            int32_t xv = get_i8(xr, (x0 + kx) * cin + ci);
                            if (mod && !(ky == 1 && kx == 1)) {
                                const uint32_t* ar_c = ring + ((y0 + 1) % kk) * per_row + xrow;
                                int32_t wd = get_u16(ar_c, (x0 + 1) * 8 + dir_of[ky * 3 + kx]);
                                int64_t g = get_u16(gam, ci);
                                int64_t a = rne(((1ll << 14) - g) * (1ll << 15) + g * wd, 15);
                                int64_t xm = rne(a * xv, 14);
                                int64_t xc = clampi(xm, -128, 127);
                                if (co == 0 && xc != xm) ++st.sat;
                                xv = (int32_t)xc;
                            }
                            acc += (int64_t)get_i8(W + co * wrow, (ky * kk + kx) * cin + ci) * xv;
                        }
                    }
                }
                int s = (p0 & CONV_F_PERCH) ? get_u8(sh, co) : (int)d.p1();
                int64_t v = rne(acc, s);
                int64_t vc = clampi(v, lo, 127);
                if (vc != v) ++st.sat;
                p8.put((int32_t)vc);
            }
        }
        p8.flush();
        (void)hout;
    }
}

static inline void bank_coef(int64_t beta, int steps, int64_t* a) {
    int64_t pw = 1ll << BETA_FRAC;
    int64_t prev = 0;
    for (int t = 0; t < steps; ++t) {
        pw = rne(pw * beta, BETA_FRAC);
        int64_t r = (1ll << COEF_R) - rne(pw, 1);
        a[t] = r - prev;
        prev = r;
    }
    a[steps] = (1ll << COEF_R) - prev;
}

template <class C>
static void body_stencil(C& io, const Desc& d, Arena& ar, UnitStats& st) {
    const int h = (int)(d.m() & 0xFFFF), w = (int)(d.m() >> 16), c = (int)d.n(), steps = (int)d.k(), halo = (int)d.p2();
    const int ho = h - 2 * halo, wo = w - 2 * halo;
    const bool bands = d.p0() & STENCIL_F_BANDS, in16 = d.p0() & STENCIL_F_IN16;
    int16_t* taps = (int16_t*)ar.take(18ull * h * w);
    uint16_t* betas = bands ? (uint16_t*)ar.take(8ull * ho * wo) : 0;
    int16_t* sA = (int16_t*)ar.take(4ull * h * w);
    int16_t* sB = sA + h * w;
    int16_t* pw = (int16_t*)ar.take(2ull * (steps + 1) * ho * wo);
    uint32_t buf[5];
    for (int i = 0; i < h * w; ++i) {
        io.read_n(buf, 5);
        for (int t = 0; t < 9; ++t) taps[9 * i + t] = (int16_t)get_i16(buf, t);
    }
    if (bands) {
        for (int i = 0; i < h * w; ++i) {
            io.read_n(buf, 2);
            int y = i / w, x = i % w;
            if (y >= halo && y < h - halo && x >= halo && x < w - halo) {
                int o = (y - halo) * wo + (x - halo);
                for (int t = 0; t < 4; ++t) betas[4 * o + t] = (uint16_t)get_u16(buf, t);
            }
        }
    }
    int e[5];
    for (int j = 0; j < 5; ++j) e[j] = (int)((d.p1() >> (4 * j)) & 0xF);
    static const int dy9[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    static const int dx9[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    for (int ch = 0; ch < c; ++ch) {
        int idx = 0;
        uint32_t wbuf = 0;
        for (int i = 0; i < h * w; ++i) {
            if (in16) {
                if ((i & 1) == 0) wbuf = io.read();
                sA[i] = (int16_t)(wbuf >> (16 * (i & 1)));
            } else {
                if ((i & 3) == 0) wbuf = io.read();
                sA[i] = (int16_t)((int32_t)(int8_t)((wbuf >> (8 * (i & 3))) & 0xFF) << 8);
            }
        }
        for (int y = 0; y < ho; ++y)
            for (int x = 0; x < wo; ++x) pw[y * wo + x] = sA[(y + halo) * w + (x + halo)];
        int16_t* cur = sA;
        int16_t* nxt = sB;
        for (int t = 1; t <= steps; ++t) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int64_t acc = 0;
                    const int16_t* tp = taps + 9 * (y * w + x);
                    for (int q = 0; q < 9; ++q) {
                        int yy = y + dy9[q], xx = x + dx9[q];
                        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                        acc += (int64_t)tp[q] * cur[yy * w + xx];
                    }
                    nxt[y * w + x] = (int16_t)rne(acc, TAP_Q);
                }
            }
            int16_t* tmp = cur; cur = nxt; nxt = tmp;
            for (int y = 0; y < ho; ++y)
                for (int x = 0; x < wo; ++x) pw[t * ho * wo + y * wo + x] = cur[(y + halo) * w + (x + halo)];
        }
        (void)idx;
        if (!bands) {
            Pack16<C> p16(io);
            for (int i = 0; i < ho * wo; ++i) p16.put(pw[steps * ho * wo + i]);
            p16.flush();
            continue;
        }
        int64_t a_lo[16], a_hi[16];
        for (int j = 0; j < 5; ++j) {
            Pack8<C> p8(io);
            for (int o = 0; o < ho * wo; ++o) {
                // chain = (0, b1, b2, b3, b4): band j uses chain[j] - chain[j+1]; j = 4 is A(b4)
                if (j < 4) {
                    if (j == 0) { for (int t = 0; t <= steps; ++t) a_lo[t] = t == 0 ? (1ll << COEF_R) : 0; }
                    else bank_coef(betas[4 * o + j - 1], steps, a_lo);
                    bank_coef(betas[4 * o + j], steps, a_hi);
                } else {
                    bank_coef(betas[4 * o + 3], steps, a_hi);
                }
                int64_t acc = 0;
                for (int t = 0; t <= steps; ++t) {
                    int64_t dc = j < 4 ? a_lo[t] - a_hi[t] : a_hi[t];
                    acc += dc * pw[t * ho * wo + o];
                }
                int64_t v = rne(acc, 23 - e[j]);
                int64_t vc = clampi(v, -128, 127);
                if (vc != v) ++st.sat;
                p8.put((int32_t)vc);
            }
            p8.flush();
        }
    }
}

template <class C>
static void body_lut(C& io, const Desc& d, Arena& ar, UnitStats& st) {
    const uint32_t m = d.m(), p0 = d.p0();
    if (p0 == (uint32_t)LUT_TABLE8) {
        uint32_t* tab = ar.take(256);
        io.read_n(tab, 64);
        Pack8<C> p8(io);
        uint32_t wv = 0;
        for (uint32_t i = 0; i < m; ++i) {
            if ((i & 3) == 0) wv = io.read();
            int32_t x = (int32_t)(int8_t)((wv >> (8 * (i & 3))) & 0xFF);
            p8.put(get_i8(tab, (uint32_t)(x + 128)));
        }
        p8.flush();
        return;
    }
    if (p0 == (uint32_t)LUT_SIGMOID || p0 == (uint32_t)LUT_EXPF) {
        Pack16<C> p16(io);
        uint32_t wv = 0;
        for (uint32_t i = 0; i < m; ++i) {
            if ((i & 1) == 0) wv = io.read();
            int32_t x = (int32_t)(int16_t)(wv >> (16 * (i & 1)));
            if (p0 == (uint32_t)LUT_SIGMOID) {
                if (x < SIG_IN_MIN || x > SIG_IN_MAX) ++st.sat;
                int32_t c = (int32_t)clampi(x, SIG_IN_MIN, SIG_IN_MAX);
                p16.put(SIG_TABLE[(c - SIG_IN_MIN) >> SIG_IDX_SHIFT]);
            } else {
                if (x < FC_LOGIT_MIN || x > FC_LOGIT_MAX) ++st.sat;
                int32_t c = (int32_t)clampi(x, FC_LOGIT_MIN, FC_LOGIT_MAX);
                io.write(FC_EXP[c - FC_LOGIT_MIN]);
            }
        }
        if (p0 == (uint32_t)LUT_SIGMOID) p16.flush();
        return;
    }
    // REQUANT32
    const uint32_t parts = d.k();
    int64_t* acc = (int64_t*)ar.take(8ull * m);
    for (uint32_t i = 0; i < m; ++i) acc[i] = 0;
    for (uint32_t p = 0; p < parts; ++p)
        for (uint32_t i = 0; i < m; ++i) acc[i] += (int32_t)io.read();
    const bool out16 = d.p2() & 2, sym = d.p2() & 1;
    int64_t lo = out16 ? -32768 : (sym ? -127 : -128), hi = out16 ? 32767 : 127;
    Pack8<C> p8(io);
    Pack16<C> p16(io);
    for (uint32_t i = 0; i < m; ++i) {
        int64_t y = rne(acc[i], (int)d.p1());
        int64_t v = clampi(y, lo, hi);
        if (v != y) ++st.sat;
        if (out16) p16.put((int32_t)v); else p8.put((int32_t)v);
    }
    p8.flush();
    p16.flush();
}

template <class C>
static void body_permute(C& io, const Desc& d, Arena& ar, UnitStats& st) {
    const uint32_t nout = d.m(), rwd = d.n(), nin = d.k();
    uint32_t* idx = ar.take(4ull * rw16(nout));
    uint32_t* rows = ar.take(4ull * nin * rwd);
    io.read_n(idx, rw16(nout));
    io.read_n(rows, nin * rwd);
    for (uint32_t i = 0; i < nout; ++i) {
        uint32_t s = (uint32_t)get_u16(idx, i);
        bool ok = s != 0xFFFF && s < nin;
        if (s != 0xFFFF && s >= nin) ++st.aux0;
        for (uint32_t q = 0; q < rwd; ++q) io.write(ok ? rows[s * rwd + q] : 0u);
    }
}

template <class C>
static void body_segops(C& io, const Desc& d, Arena& ar, UnitStats& st) {
    const uint32_t cells = d.m(), c = d.n(), nf = d.k(), p0 = d.p0();
    const uint32_t rc = rw8(c);
    if (p0 == (uint32_t)SEGOPS_SEGSUM) {
        int32_t* sums = (int32_t*)ar.take(4ull * nf * c);
        int32_t* ws = (int32_t*)ar.take(4ull * nf);
        uint32_t* ids = ar.take(4ull * rw16(cells));
        uint32_t* wts = ar.take(4ull * rw8(cells));
        for (uint32_t i = 0; i < nf * c; ++i) sums[i] = 0;
        for (uint32_t i = 0; i < nf; ++i) ws[i] = 0;
        io.read_n(ids, rw16(cells));
        io.read_n(wts, rw8(cells));
        uint32_t row[64];
        for (uint32_t i = 0; i < cells; ++i) {
            uint32_t f = (uint32_t)get_u16(ids, i);
            int32_t wv = get_u8(wts, i);
            for (uint32_t q = 0; q < rc; ++q) row[q & 63] = io.read();
            if (f == (uint32_t)UNASSIGNED) continue;
            if (f >= nf) { ++st.aux0; continue; }
            for (uint32_t ch = 0; ch < c; ++ch) sums[f * c + ch] += wv * get_i8(row, ch);
            ws[f] += wv;
        }
        for (uint32_t i = 0; i < nf * c; ++i) io.write((uint32_t)sums[i]);
        for (uint32_t i = 0; i < nf; ++i) io.write((uint32_t)ws[i]);
        return;
    }
    if (p0 == (uint32_t)SEGOPS_SEGMEAN) {
        int32_t* ws = (int32_t*)ar.take(4ull * nf);
        for (uint32_t f = 0; f < nf; ++f) ws[f] = (int32_t)io.read();
        Pack8<C> p8(io);
        for (uint32_t f = 0; f < nf; ++f) {
            for (uint32_t ch = 0; ch < c; ++ch) {
                int64_t s = (int32_t)io.read();
                int64_t v = ws[f] > 0 ? div_rne(s, ws[f]) : 0;
                int64_t vc = clampi(v, -128, 127);
                if (vc != v) ++st.sat;
                p8.put((int32_t)vc);
            }
            p8.flush();
        }
        return;
    }
    // FILL, DETAIL, SYNTH, CLSBAND
    uint32_t* H = ar.take(4ull * nf * rc);
    uint32_t* RZ = (p0 == (uint32_t)SEGOPS_CLSBAND) ? ar.take(4ull * nf * rc) : 0;
    io.read_n(H, nf * rc);
    if (RZ) io.read_n(RZ, nf * rc);
    const bool need_ids = (p0 == (uint32_t)SEGOPS_DETAIL || p0 == (uint32_t)SEGOPS_SYNTH);
    uint32_t* ids = need_ids ? ar.take(4ull * rw16(cells)) : 0;
    if (need_ids) io.read_n(ids, rw16(cells));
    uint32_t idw = 0;
    uint32_t xrow[64], rrow[64];
    for (uint32_t i = 0; i < cells; ++i) {
        uint32_t f;
        if (need_ids) f = (uint32_t)get_u16(ids, i);
        else {
            if ((i & 1) == 0) idw = io.read();
            f = (idw >> (16 * (i & 1))) & 0xFFFF;
        }
        bool ok = f != (uint32_t)UNASSIGNED && f < nf;
        if (f != (uint32_t)UNASSIGNED && f >= nf) ++st.aux0;
        const uint32_t* hr = ok ? H + f * rc : 0;
        if (p0 == (uint32_t)SEGOPS_FILL) {
            for (uint32_t q = 0; q < rc; ++q) io.write(ok ? hr[q] : 0u);
        } else if (p0 == (uint32_t)SEGOPS_DETAIL) {
            for (uint32_t q = 0; q < rc; ++q) xrow[q & 63] = io.read();
            Pack16<C> p16(io);
            for (uint32_t ch = 0; ch < c; ++ch) p16.put(get_i8(xrow, ch) - (ok ? get_i8(hr, ch) : 0));
            p16.flush();
        } else if (p0 == (uint32_t)SEGOPS_SYNTH) {
            for (uint32_t q = 0; q < rc; ++q) xrow[q & 63] = io.read();
            if (d.p1() & 1) for (uint32_t q = 0; q < rc; ++q) rrow[q & 63] = io.read();
            Pack8<C> p8(io);
            for (uint32_t ch = 0; ch < c; ++ch) {
                int64_t y = (int64_t)get_i8(xrow, ch) + (ok ? get_i8(hr, ch) : 0) + ((d.p1() & 1) ? get_i8(rrow, ch) : 0);
                int64_t v = clampi(y, -128, 127);
                if (v != y) ++st.sat;
                p8.put((int32_t)v);
            }
            p8.flush();
        } else {
            Pack8<C> p8(io);
            for (uint32_t ch = 0; ch < c; ++ch) {
                if (!ok) { p8.put(0); continue; }
                int64_t y = (int64_t)get_i8(hr, ch) + get_i8(RZ + f * rc, ch);
                int64_t v = clampi(y, -128, 127);
                if (v != y) ++st.sat;
                p8.put((int32_t)v);
            }
            p8.flush();
        }
    }
}

template <class C>
static void body_tapmod(C& io, const Desc& d, Arena& ar, UnitStats& st) {
    const uint32_t cells = d.m(), c = d.n(), nf = d.k(), p0 = d.p0();
    const uint32_t rc = rw8(c);
    uint32_t xrow[64];
    if (p0 == (uint32_t)TAPMOD_MODULATE) {
        uint32_t* gam = ar.take(4ull * rw16(c));
        uint32_t* aff = ar.take(4ull * rw16(cells));
        io.read_n(gam, rw16(c));
        io.read_n(aff, rw16(cells));
        for (uint32_t i = 0; i < cells; ++i) {
            for (uint32_t q = 0; q < rc; ++q) xrow[q & 63] = io.read();
            Pack8<C> p8(io);
            int64_t wd = get_u16(aff, i);
            for (uint32_t ch = 0; ch < c; ++ch) {
                int64_t g = get_u16(gam, ch);
                int64_t a = rne(((1ll << 14) - g) * (1ll << 15) + g * wd, 15);
                int64_t y = rne(a * get_i8(xrow, ch), 14);
                int64_t v = clampi(y, -128, 127);
                if (v != y) ++st.sat;
                p8.put((int32_t)v);
            }
            p8.flush();
        }
        return;
    }
    if (p0 == (uint32_t)TAPMOD_ENERGY) {
        const uint32_t j = d.p1();
        int64_t* sums = (int64_t*)ar.take(8ull * nf * j);
        int32_t* cnt = (int32_t*)ar.take(4ull * nf);
        uint32_t* ids = ar.take(4ull * rw16(cells));
        for (uint32_t i = 0; i < nf * j; ++i) sums[i] = 0;
        for (uint32_t i = 0; i < nf; ++i) cnt[i] = 0;
        io.read_n(ids, rw16(cells));
        for (uint32_t jj = 0; jj < j; ++jj) {
            for (uint32_t i = 0; i < cells; ++i) {
                for (uint32_t q = 0; q < rc; ++q) xrow[q & 63] = io.read();
                uint32_t f = (uint32_t)get_u16(ids, i);
                if (f == (uint32_t)UNASSIGNED || f >= nf) continue;
                int64_t e2 = 0;
                for (uint32_t ch = 0; ch < c; ++ch) { int64_t v = get_i8(xrow, ch); e2 += v * v; }
                sums[f * j + jj] += e2;
                if (jj == 0) cnt[f] += (int32_t)c;
            }
        }
        for (uint32_t i = 0; i < nf * j; ++i) {
            uint64_t v = (uint64_t)sums[i];
            io.write((uint32_t)(v & 0xFFFFFFFFu));
            io.write((uint32_t)(v >> 32));
        }
        for (uint32_t i = 0; i < nf; ++i) io.write((uint32_t)cnt[i]);
        return;
    }
    if (p0 == (uint32_t)TAPMOD_NU) {
        const uint32_t j = d.n();
        int64_t s[16];
        int32_t k[16];
        for (uint32_t f = 0; f < nf; ++f) {
            for (uint32_t jj = 0; jj < j; ++jj) {
                uint64_t lo = io.read(), hi = io.read();
                s[jj] = (int64_t)(lo | (hi << 32));
            }
            for (uint32_t jj = 0; jj < j; ++jj) k[jj] = (int32_t)io.read();
            Pack8<C> p8(io);
            for (uint32_t jj = 0; jj < j; ++jj) {
                if (k[jj] <= 0) { p8.put(0); continue; }
                int64_t v = rne(div_rne(s[jj], k[jj]), (int)d.p1());
                if (v > 127) { ++st.sat; v = 127; }
                p8.put((int32_t)v);
            }
            p8.flush();
        }
        return;
    }
    // MIX
    const uint32_t j = d.k();
    uint32_t* gains = ar.take(4ull * rw16(j));
    int32_t* acc = (int32_t*)ar.take(4ull * c);
    io.read_n(gains, rw16(j));
    for (uint32_t i = 0; i < cells; ++i) {
        for (uint32_t ch = 0; ch < c; ++ch) acc[ch] = 0;
        for (uint32_t jj = 0; jj < j; ++jj) {
            for (uint32_t q = 0; q < rc; ++q) xrow[q & 63] = io.read();
            int32_t g = get_i16(gains, jj);
            for (uint32_t ch = 0; ch < c; ++ch) acc[ch] += g * get_i8(xrow, ch);
        }
        Pack8<C> p8(io);
        for (uint32_t ch = 0; ch < c; ++ch) {
            int64_t y = rne(acc[ch], (int)d.p1());
            int64_t v = clampi(y, -128, 127);
            if (v != y) ++st.sat;
            p8.put((int32_t)v);
        }
        p8.flush();
    }
    (void)nf;
}

static inline int32_t fc_logit(int64_t raw, int64_t mult, int shift) {
    return (int32_t)clampi(rne(raw * mult, shift), FC_LOGIT_MIN, FC_LOGIT_MAX);
}

static inline int32_t sigmoid_q15(int64_t code) {
    int32_t c = (int32_t)clampi(code, SIG_IN_MIN, SIG_IN_MAX);
    return SIG_TABLE[(c - SIG_IN_MIN) >> SIG_IDX_SHIFT];
}

template <class C>
static void body_head(C& io, const Desc& d, Arena& ar, UnitStats& st) {
    if (d.p0() == (uint32_t)HEAD_FORECAST) {
        const uint32_t rows = d.m(), du = d.n(), nf = d.k(), row0 = d.p1(), width = d.p2();
        const uint32_t rk = rw8(KEY_D), ru = rw8(du), rv = rw8(width);
        uint32_t* K = ar.take(4ull * nf * (rk + ru + rv));
        uint32_t* U = K + nf * rk;
        uint32_t* V = U + nf * ru;
        io.read_n(K, nf * rk);
        io.read_n(U, nf * ru);
        io.read_n(V, nf * rv);
        uint32_t mask[F_CAP / 32];
        uint32_t hs[C_TRANSPORT / 4];
        int64_t num[C_TRANSPORT];
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t f = row0 + r;
            io.read_n(mask, F_CAP / 32);
            io.read_n(hs, rv);
            int32_t null_raw = (int32_t)io.read();
            int32_t alpha_raw = (int32_t)io.read();
            int32_t l0 = fc_logit(null_raw, FC_NULL_MULT, FC_NULL_SHIFT);
            int64_t w0 = FC_EXP[l0 - FC_LOGIT_MIN];
            int64_t den = w0;
            for (uint32_t ch = 0; ch < width; ++ch) num[ch] = 0;
            uint32_t n_adm = 0;
            for (uint32_t g = 0; g < nf; ++g) {
                if (!((mask[g >> 5] >> (g & 31)) & 1u) || g == f) continue;
                ++n_adm;
                int64_t kk = 0, uu = 0;
                for (uint32_t t = 0; t < (uint32_t)KEY_D; ++t) kk += (int64_t)get_i8(K + f * rk, t) * get_i8(K + g * rk, t);
                for (uint32_t t = 0; t < du; ++t) uu += (int64_t)get_i8(U + f * ru, t) * get_i8(U + g * ru, t);
                int32_t l = fc_logit(kk * FC_KK_MULT + uu * FC_UU_MULT, 1, FC_SHIFT);
                int64_t wg = FC_EXP[l - FC_LOGIT_MIN];
                den += wg;
                for (uint32_t ch = 0; ch < width; ++ch) {
                    int64_t df = (int64_t)get_i8(V + g * rv, ch) - get_i8(hs, ch);
                    df = clampi(df, -FC_B, FC_B);
                    num[ch] += wg * df;
                }
            }
            int64_t sig = sigmoid_q15(rne((int64_t)alpha_raw * FC_ALPHA_MULT, FC_ALPHA_SHIFT));
            int64_t alpha = (sig * FC_SHRINK_NUM) >> FC_SHRINK_SHIFT;
            int64_t scale = (alpha << FC_SCALE_SHIFT) / den;
            Pack8<C> p8(io);
            for (uint32_t ch = 0; ch < width; ++ch) p8.put((int32_t)clampi(rne(num[ch] * scale, FC_OUT_SHIFT), -127, 127));
            p8.flush();
            bool abst = n_adm == 0 || w0 >= den - w0;
            if (abst) ++st.aux0;
            io.write(n_adm | ((abst ? 1u : 0u) << 16));
        }
        return;
    }
    // SCORE
    const uint32_t cells = d.m(), tau = d.p1();
    uint32_t* lg = ar.take(4ull * rw16(cells));
    uint32_t* bits = ar.take(4ull * ((cells + 31) / 32));
    io.read_n(lg, rw16(cells));
    for (uint32_t i = 0; i < (cells + 31) / 32; ++i) bits[i] = 0;
    Pack16<C> p16(io);
    uint32_t gw = 0;
    for (uint32_t i = 0; i < cells; ++i) {
        if ((i & 1) == 0) gw = io.read();
        int64_t g = (gw >> (16 * (i & 1))) & 0xFFFF;
        if (g > 32767) g = 32767;
        int64_t p = ((int64_t)sigmoid_q15(get_i16(lg, i)) * g) >> SCORE_FRAC;
        p16.put((int32_t)p);
        if (p >= (int64_t)tau) { bits[i >> 5] |= 1u << (i & 31); ++st.aux0; }
    }
    p16.flush();
    for (uint32_t i = 0; i < (cells + 31) / 32; ++i) io.write(bits[i]);
}

// ------------------------------------------------------------------ dispatcher

template <class IO>
static void send_header(IO& io, const Desc& d, uint32_t status, uint32_t out_words, uint32_t* hdr) {
    hdr[0] = RSP_MAGIC;
    hdr[1] = d.frame_id();
    hdr[2] = (d.opcode() & 0xFF) | ((status & 0xFF) << 8) | ((d.slot() & 0xFF) << 16) | ((d.worker() & 0xFF) << 24);
    hdr[3] = out_words;
    hdr[4] = d.token();
    hdr[5] = d.in_words();
    hdr[6] = 0;
    hdr[7] = check_word(hdr, 7, CHECK_SEED);
    for (int i = 0; i < 8; ++i) io.write(hdr[i]);
}

template <class IO>
static void send_trailer(IO& io, const uint32_t* hdr, const UnitStats& st) {
    uint32_t t[4] = {st.sat, st.aux0, st.aux1, 0};
    t[3] = check_word(t, 3, hdr[7]);
    for (int i = 0; i < 4; ++i) io.write(t[i]);
}

template <class IO>
static void error_frame(IO& io, const Desc& d, uint32_t status) {
    uint32_t hdr[8];
    UnitStats z = {0, 0, 0};
    send_header(io, d, status, 0, hdr);
    send_trailer(io, hdr, z);
}

// Executes exactly one unit: reads its descriptor and payload, writes its response.
template <class IO>
static void dispatch_one(IO& io, uint32_t* scratch) {
    Desc d;
    for (int i = 0; i < 16; ++i) d.w[i] = io.read();
    if (d.magic() != CMD_MAGIC) { error_frame(io, d, ST_BAD_MAGIC); return; }
    if (d.check() != check_word(d.w, 15, CHECK_SEED)) { error_frame(io, d, ST_BAD_CHECK); return; }
    if (d.version() != (uint32_t)PROTO_VERSION) { error_frame(io, d, ST_BAD_VERSION); return; }
    const uint32_t in_words = d.in_words(), out_words = d.out_words();
    if (d.opcode() == (uint32_t)OP_NOP) {
        for (uint32_t i = 0; i < in_words; ++i) io.read();
        error_frame(io, d, ST_OK);
        return;
    }
    uint64_t win, wout, scr;
    int sh = shape_of(d, win, wout, scr);
    uint32_t status = ST_OK;
    if (d.opcode() <= 7 && !((SV_BODY_MASK >> d.opcode()) & 1u)) status = ST_BAD_OPCODE;   // body not built into this worker class
    else if (sh == SH_KEY) status = d.opcode() > 7 ? ST_BAD_OPCODE : ST_BAD_SUBOP;
    else if (sh == SH_VALUE) status = ST_BAD_SHAPE;
    else if (ev(win) != in_words || ev(wout) != out_words) status = ST_BAD_SHAPE;
    else if (scr > (uint64_t)SCRATCH_BYTES) status = ST_SCRATCH;
    if (status != ST_OK) {
        for (uint32_t i = 0; i < in_words; ++i) io.read();
        error_frame(io, d, status);
        return;
    }
    uint32_t hdr[8];
    send_header(io, d, ST_OK, out_words, hdr);
    CountIO<IO> cio(io);
    UnitStats st = {0, 0, 0};
    Arena ar = {scratch, 0};
    switch (d.opcode()) {
#if SV_BODY_MASK & 0x01u
        case OP_CONV: body_conv(cio, d, ar, st); break;
#endif
#if SV_BODY_MASK & 0x02u
        case OP_GEMM: body_gemm(cio, d, ar, st); break;
#endif
#if SV_BODY_MASK & 0x04u
        case OP_STENCIL: body_stencil(cio, d, ar, st); break;
#endif
#if SV_BODY_MASK & 0x08u
        case OP_LUT: body_lut(cio, d, ar, st); break;
#endif
#if SV_BODY_MASK & 0x10u
        case OP_PERMUTE: body_permute(cio, d, ar, st); break;
#endif
#if SV_BODY_MASK & 0x20u
        case OP_SEGOPS: body_segops(cio, d, ar, st); break;
#endif
#if SV_BODY_MASK & 0x40u
        case OP_TAPMOD: body_tapmod(cio, d, ar, st); break;
#endif
#if SV_BODY_MASK & 0x80u
        case OP_HEAD: body_head(cio, d, ar, st); break;
#endif
        default: break;
    }
    while (cio.nread < in_words) cio.read();
    while (cio.nwritten < out_words) cio.write(0u);
    send_trailer(io, hdr, st);
}

}  // namespace sva

#endif  // SV_UNITS_H
