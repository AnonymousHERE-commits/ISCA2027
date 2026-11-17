// SPLICE-V integer primitives for the PL engines (Vitis HLS C++) and their host builds.
//
// The arithmetic is the manifest's (Definition 1): exact integer products held in
// containers wide enough for every bound, one explicit rounding at each manifest
// site, explicit saturation.  Containers:
//   - host builds (g++/clang):  i128 = __int128
//   - Vitis HLS (csim/synth):   i128 = ap_int<128>  (define SPLICEV_USE_AP_INT, set by pl/configs)
// Both hold the same mathematical integers, so results are identical bit for bit.
#ifndef SV_INT_H
#define SV_INT_H

#include <stdint.h>

#if defined(SPLICEV_USE_AP_INT) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_int<128> i128;
typedef ap_uint<128> u128;
static inline int64_t sv_to_i64(const i128& x) { return x.to_int64(); }
static inline uint64_t sv_to_u64(const u128& x) { return x.to_uint64(); }
#else
typedef __int128 i128;
typedef unsigned __int128 u128;
static inline int64_t sv_to_i64(i128 x) { return (int64_t)x; }
static inline uint64_t sv_to_u64(u128 x) { return (uint64_t)x; }
#endif

namespace sv {

// x * 2^-s rounded half to even; s <= 0 is an exact left shift.  Arithmetic (floor) right shifts.
static inline int64_t rshift_rne64(int64_t x, int s) {
    if (s <= 0) return x << (-s);
    int64_t q = x >> s;
    int64_t r = x - (q << s);
    int64_t half = (int64_t)1 << (s - 1);
    if (r > half || (r == half && (q & 1))) q += 1;
    return q;
}

static inline i128 rshift_rne128(i128 x, int s) {
    if (s <= 0) return x << (-s);
    i128 q = x >> s;
    i128 r = x - (q << s);
    i128 half = ((i128)1) << (s - 1);
    if (r > half || (r == half && (q & 1) != 0)) q += 1;
    return q;
}

// floor(n / d) for d > 0 (C++ division truncates toward zero).
static inline int64_t floor_div64(int64_t n, int64_t d) {
    int64_t q = n / d;
    int64_t r = n % d;
    if (r < 0) q -= 1;
    return q;
}

static inline i128 floor_div128(i128 n, i128 d) {
    i128 q = n / d;
    i128 r = n - q * d;
    if (r < 0) q -= 1;
    return q;
}

// n / d rounded half to even, d > 0.
static inline int64_t div_rne64(int64_t n, int64_t d) {
    int64_t q = floor_div64(n, d);
    int64_t r = n - q * d;
    int64_t two_r = 2 * r;
    if (two_r > d || (two_r == d && (q & 1))) q += 1;
    return q;
}

static inline i128 div_rne128(i128 n, i128 d) {
    i128 q = floor_div128(n, d);
    i128 r = n - q * d;
    i128 two_r = r * 2;
    if (two_r > d || (two_r == d && (q & 1) != 0)) q += 1;
    return q;
}

static inline int64_t clamp64(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }

static inline int bitlen64(uint64_t x) {
    int n = 0;
    for (int i = 0; i < 64; ++i) {
#pragma HLS UNROLL
        if (x >> i) n = i + 1;
    }
    return n;
}

// floor(sqrt(n)) for n < 2^126, bit by bit (no floating point).
static inline u128 isqrt_floor(u128 n) {
    u128 res = 0;
    u128 bit = ((u128)1) << 124;
    for (int i = 0; i < 63; ++i) {
        if (bit > n) bit >>= 2;
    }
    for (int i = 0; i < 63; ++i) {
        if (bit != 0) {
            if (n >= res + bit) {
                n -= res + bit;
                res = (res >> 1) + bit;
            } else {
                res >>= 1;
            }
            bit >>= 2;
        }
    }
    return res;
}

static inline u128 isqrt_ceil(u128 n) {
    u128 r = isqrt_floor(n);
    return (r * r == n) ? r : r + 1;
}

static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

static inline uint32_t check_word(const uint32_t* w, int n, uint32_t seed) {
    uint32_t c = seed;
    for (int i = 0; i < n; ++i) c = rotl32(c, 5) ^ w[i];
    return c;
}

}  // namespace sv

#endif  // SV_INT_H
