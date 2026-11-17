// sv_reserve: P3 reserve anchors by thin-aware farthest-point sampling (starts at H0).
#include "sv_kernels.h"
#include "sv_reserve.h"

extern "C" void sv_reserve(const uint32_t* cells, uint32_t* picks, uint32_t* stats, int n, int width, int count) {
#pragma HLS INTERFACE m_axi port = cells offset = slave bundle = gmem0 depth = 6400
#pragma HLS INTERFACE m_axi port = picks offset = slave bundle = gmem1 depth = 256
#pragma HLS INTERFACE m_axi port = stats offset = slave bundle = gmem1 depth = 4
#pragma HLS INTERFACE s_axilite port = n
#pragma HLS INTERFACE s_axilite port = width
#pragma HLS INTERFACE s_axilite port = count
#pragma HLS INTERFACE s_axilite port = return
    static uint8_t e[SV_RESERVE_NMAX];
    static uint8_t t[SV_RESERVE_NMAX];
    static uint16_t out[svgen::N_RESERVE];
    static sv::ReserveState<SV_RESERVE_NMAX> st;
    if (n > SV_RESERVE_NMAX) n = SV_RESERVE_NMAX;
    if (count > svgen::N_RESERVE) count = svgen::N_RESERVE;
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
        uint32_t w = cells[i];
        e[i] = (uint8_t)(w & 0xFF);
        t[i] = (uint8_t)((w >> 8) & 0xFF);
    }
    uint64_t scans = 0;
    int np = sv::reserve_select<SV_RESERVE_NMAX>(e, t, n, width, count, out, st, scans);
    for (int i = 0; i < np; ++i) picks[i] = out[i];
    stats[0] = (uint32_t)np;
    stats[1] = (uint32_t)(scans & 0xFFFFFFFFu);
    stats[2] = (uint32_t)(scans >> 32);
    stats[3] = 0;
}
