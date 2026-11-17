// Nonnegative block-floating arithmetic with 24-bit mantissas (posterior and gate lanes).
// Twin of reference/splicev_ref/bf24.py: every operation is the exact result rounded half
// to even to 24 bits, except recip(), the manifest's 12-bit seed + two Newton steps unit.
#ifndef SV_BF24_H
#define SV_BF24_H

#include "sv_int.h"
#include "splicev_gen.h"

namespace sv {

struct BF {
    uint32_t m;  // 0, or in [2^23, 2^24)
    int32_t e;   // value = m * 2^e
};

static const uint32_t BF_LO = 1u << 23;
static const uint32_t BF_HI = 1u << 24;

static inline BF bf_zero() { BF z; z.m = 0; z.e = 0; return z; }
static inline bool bf_is_zero(const BF& a) { return a.m == 0; }

// exact nonnegative x * 2^e -> nearest BF (x < 2^63)
static inline BF bf_norm(uint64_t x, int32_t e) {
    BF r;
    if (x == 0) return bf_zero();
    int n = bitlen64(x);
    if (n > 24) {
        int s = n - 24;
        int64_t m = rshift_rne64((int64_t)x, s);
        e += s;
        if (m == (int64_t)BF_HI) { m >>= 1; e += 1; }
        r.m = (uint32_t)m;
        r.e = e;
        return r;
    }
    r.m = (uint32_t)(x << (24 - n));
    r.e = e - (24 - n);
    return r;
}

static inline BF bf_from_int(uint64_t x) { return bf_norm(x, 0); }

static inline BF bf_mul(const BF& a, const BF& b) {
    if (a.m == 0 || b.m == 0) return bf_zero();
    return bf_norm((uint64_t)a.m * (uint64_t)b.m, a.e + b.e);
}

static inline BF bf_add(BF a, BF b) {
    if (a.m == 0) return b;
    if (b.m == 0) return a;
    if (a.e < b.e) { BF t = a; a = b; b = t; }
    int32_t d = a.e - b.e;
    if (d >= 26) return a;                       // b < 1/4 ulp(a): RNE result is a
    return bf_norm(((uint64_t)a.m << d) + (uint64_t)b.m, b.e);
}

static inline BF bf_recip(const BF& a) {
    uint64_t m = a.m;                            // caller guarantees a != 0
    uint64_t y = svgen::RECIP_SEED[(m >> 12) - 2048];
    for (int it = 0; it < 2; ++it) {
        uint64_t t = m * y;                      // Q2.54
        uint64_t err = (1ull << 55) - t;         // 2 - x y, Q2.54 (positive)
        y = (y * (err >> 24)) >> 30;             // Q2.30
    }
    return bf_norm(y, -54 - a.e);
}

static inline int bf_cmp(const BF& a, const BF& b) {
    if (a.m == 0 || b.m == 0) return (a.m > 0) - (b.m > 0);
    if (a.e != b.e) return a.e > b.e ? 1 : -1;
    return (a.m > b.m) - (a.m < b.m);
}

// a * 2^frac rounded half-even (result must fit int64)
static inline int64_t bf_to_fixed(const BF& a, int frac) {
    int s = a.e + frac;
    if (s >= 0) return (int64_t)a.m << s;
    if (-s >= 40) return 0;
    return rshift_rne64((int64_t)a.m, -s);
}

// e^(s_code / 256), relative error below 2^-20 (two-level 2^f table)
static inline BF bf_exp_q78(int32_t s_code) {
    int64_t t = (int64_t)s_code * svgen::LOG2E_Q30;          // units 2^-38
    int64_t k = t >> 38;                                     // floor
    int64_t f = t - (k << 38);                               // [0, 2^38)
    int hi = (int)(f >> 28);
    int64_t lo = f & ((1ll << 28) - 1);
    int64_t c = (lo * svgen::LN2_Q30) >> 44;                 // Q0.24
    int64_t mm = rshift_rne64((int64_t)svgen::EXP2_HI[hi] * ((1ll << 24) + c), 24);
    return bf_norm((uint64_t)mm, (int32_t)(k - 23));
}

}  // namespace sv

#endif  // SV_BF24_H
