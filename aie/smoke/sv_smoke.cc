// G02 smoke kernel (roadmap 3.3): integer add on an AIE-ML tile, framed so the host can
// check lengths and contents exactly.  Per invocation (one graph iteration):
//   in : 0x534D4B30 'SMK0', n (even), add, 0, then n words                 (4 + n words)
//   out: 0x534D4B31 'SMK1', n, then n words (in + add, mod 2^32), then the
//        xor of the n input words and 0x534D4B32 'SMK2'                     (4 + n words)
// A wrong magic sets n = 0 (the payload is not consumed: the host must restart the stream).
#include <adf.h>
#include <stdint.h>

#include "sv_smoke.h"

void sv_smoke(input_stream<int32>* __restrict in, output_stream<int32>* __restrict out) {
    uint32_t magic = (uint32_t)readincr(in);
    uint32_t n = (uint32_t)readincr(in);
    uint32_t add = (uint32_t)readincr(in);
    (void)readincr(in);
    if (magic != 0x534D4B30u) n = 0;
    uint32_t x = 0;
    writeincr(out, (int32)0x534D4B31u);
    writeincr(out, (int32)n);
    // the checksum is known only after the payload; it is sent in a 2-word trailer
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t v = (uint32_t)readincr(in);
        x ^= v;
        writeincr(out, (int32)(v + add));
    }
    writeincr(out, (int32)x);
    writeincr(out, (int32)0x534D4B32u);
}
