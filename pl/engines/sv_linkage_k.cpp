// sv_linkage: certified MNN linkage with the counted work budget, fragment selection by
// eligibility mass, and the compact anchor order (publishes ids and order at H2).
#include "sv_kernels.h"
#include "sv_linkage.h"

extern "C" void sv_linkage(const uint32_t* edges, const uint32_t* anchors, uint32_t* ids, uint32_t* order, uint32_t* stats,
                           int n, int n_edges, int budget, int f_cap) {
#pragma HLS INTERFACE m_axi port = edges offset = slave bundle = gmem0 depth = 98304
#pragma HLS INTERFACE m_axi port = anchors offset = slave bundle = gmem0 depth = 1856
#pragma HLS INTERFACE m_axi port = ids offset = slave bundle = gmem1 depth = 1856
#pragma HLS INTERFACE m_axi port = order offset = slave bundle = gmem1 depth = 1856
#pragma HLS INTERFACE m_axi port = stats offset = slave bundle = gmem1 depth = 8
#pragma HLS INTERFACE s_axilite port = n
#pragma HLS INTERFACE s_axilite port = n_edges
#pragma HLS INTERFACE s_axilite port = budget
#pragma HLS INTERFACE s_axilite port = f_cap
#pragma HLS INTERFACE s_axilite port = return
    static int32_t ed[3 * SV_LINK_EMAX];
    static uint8_t act[SV_LINK_NMAX];
    static uint8_t elig[SV_LINK_NMAX];
    static uint16_t id16[SV_LINK_NMAX];
    static uint16_t ord16[SV_LINK_NMAX];
    static int counts[svgen::F_CAP + 2];
    static sv::LinkState<SV_LINK_NMAX, SV_LINK_NBCAP> st;
    sv::LinkResult r;
    r.rounds = 0; r.work = 0; r.budget_hit = false; r.overflow = false; r.dup_error = false; r.n_frag = 0; r.cap_hit = false;
    // capacities exceeded: fail closed (no fragments), flagged as overflow
    bool fatal = n > SV_LINK_NMAX || n_edges > SV_LINK_EMAX || n < 0 || n_edges < 0;
    int nn = fatal ? (n > SV_LINK_NMAX ? SV_LINK_NMAX : (n < 0 ? 0 : n)) : n;
    int ne = fatal ? 0 : n_edges;
    for (int i = 0; i < 3 * ne; ++i) {
#pragma HLS PIPELINE II = 1
        ed[i] = (int32_t)edges[i];
    }
    for (int i = 0; i < nn; ++i) {
#pragma HLS PIPELINE II = 1
        uint32_t w = anchors[i];
        act[i] = (uint8_t)(w & 0xFF);
        elig[i] = (uint8_t)((w >> 8) & 0xFF);
    }
    if (f_cap > svgen::F_CAP) f_cap = svgen::F_CAP;
    sv::Linker<SV_LINK_NMAX, SV_LINK_NBCAP> lk(st);
    int F = 0;
    if (fatal) {
        r.overflow = true;
        for (int i = 0; i < nn; ++i) id16[i] = (uint16_t)svgen::UNASSIGNED;
    } else {
        // a list overflow stops the linkage and keeps the clusters formed so far (as the reference does);
        // a duplicate window pair is a protocol error and yields no fragments
        lk.init(nn, ed, ne, act, r);
        lk.run(budget < 0 ? -1 : (int64_t)budget, r);
        if (r.dup_error) {
            for (int i = 0; i < nn; ++i) id16[i] = (uint16_t)svgen::UNASSIGNED;
        } else {
            F = lk.select(act, elig, f_cap, id16, r);
        }
    }
    sv::anchor_order(id16, nn, F, ord16, counts);
    for (int i = 0; i < nn; ++i) {
#pragma HLS PIPELINE II = 1
        ids[i] = id16[i];
        order[i] = ord16[i];
    }
    stats[0] = (uint32_t)F;
    stats[1] = (uint32_t)r.rounds;
    stats[2] = (uint32_t)((uint64_t)r.work & 0xFFFFFFFFu);
    stats[3] = (uint32_t)((uint64_t)r.work >> 32);
    stats[4] = r.budget_hit ? 1u : 0u;
    stats[5] = r.overflow ? 1u : 0u;
    stats[6] = r.dup_error ? 1u : 0u;
    stats[7] = r.cap_hit ? 1u : 0u;
}
