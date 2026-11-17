// Integer helpers for the AIE-ML dispatcher bodies (scalar path).  int64 only: the AIE
// compiler supports 64-bit integers on the scalar unit; no 128-bit type is needed here.
#ifndef SV_AIE_INT_H
#define SV_AIE_INT_H

#include <stdint.h>

namespace sva {

static inline int64_t rne(int64_t x, int s) {
    if (s <= 0) return x << (-s);
    int64_t q = x >> s;
    int64_t r = x - (q << s);
    int64_t half = (int64_t)1 << (s - 1);
    if (r > half || (r == half && (q & 1))) q += 1;
    return q;
}

static inline int64_t floor_div(int64_t n, int64_t d) {
    int64_t q = n / d;
    if ((n % d) < 0) q -= 1;
    return q;
}

static inline int64_t div_rne(int64_t n, int64_t d) {
    int64_t q = floor_div(n, d);
    int64_t r = n - q * d;
    int64_t two_r = 2 * r;
    if (two_r > d || (two_r == d && (q & 1))) q += 1;
    return q;
}

static inline int64_t clampi(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }

static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

static inline uint32_t check_word(const uint32_t* w, int n, uint32_t seed) {
    uint32_t c = seed;
    for (int i = 0; i < n; ++i) c = rotl32(c, 5) ^ w[i];
    return c;
}

static inline uint32_t rw8(uint32_t n) { return (n + 3) / 4; }
static inline uint32_t rw16(uint32_t n) { return (n + 1) / 2; }
static inline uint64_t a4(uint64_t b) { return (b + 3) & ~(uint64_t)3; }

static inline int32_t get_i8(const uint32_t* w, uint32_t i) { return (int32_t)(int8_t)((w[i >> 2] >> (8 * (i & 3))) & 0xFF); }
static inline int32_t get_u8(const uint32_t* w, uint32_t i) { return (int32_t)((w[i >> 2] >> (8 * (i & 3))) & 0xFF); }
static inline int32_t get_i16(const uint32_t* w, uint32_t i) { return (int32_t)(int16_t)((w[i >> 1] >> (16 * (i & 1))) & 0xFFFF); }
static inline int32_t get_u16(const uint32_t* w, uint32_t i) { return (int32_t)((w[i >> 1] >> (16 * (i & 1))) & 0xFFFF); }

}  // namespace sva

#endif
