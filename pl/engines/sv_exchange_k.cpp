// sv_exchange1: posterior (Eq. 1) per receiver row, mutual posterior, screened trust solve
//               (Eq. 2) -> source lists (H3') and the trusts of each list slot.
// sv_exchange2: one-hop innovation dV from the translated values of the priority unit (H3'').
#include "sv_kernels.h"
#include "sv_posterior.h"
#include "sv_trust.h"

static int find_partner(const uint16_t* row, int np, int f) {
    for (int k = 0; k < np; ++k)
        if (row[k] == f) return k;
    return -1;
}

extern "C" void sv_exchange1(const uint32_t* partners, const uint32_t* evidence, const uint32_t* npart, const uint32_t* need,
                             const uint32_t* rho, const uint32_t* phi, uint32_t* sources, uint32_t* lam, uint32_t* status,
                             int n_frag) {
#pragma HLS INTERFACE m_axi port = partners offset = slave bundle = gmem0 depth = 32768
#pragma HLS INTERFACE m_axi port = evidence offset = slave bundle = gmem0 depth = 32768
#pragma HLS INTERFACE m_axi port = npart offset = slave bundle = gmem0 depth = 256
#pragma HLS INTERFACE m_axi port = need offset = slave bundle = gmem0 depth = 256
#pragma HLS INTERFACE m_axi port = rho offset = slave bundle = gmem0 depth = 256
#pragma HLS INTERFACE m_axi port = phi offset = slave bundle = gmem0 depth = 1024
#pragma HLS INTERFACE m_axi port = sources offset = slave bundle = gmem1 depth = 2048
#pragma HLS INTERFACE m_axi port = lam offset = slave bundle = gmem1 depth = 4096
#pragma HLS INTERFACE m_axi port = status offset = slave bundle = gmem1 depth = 256
#pragma HLS INTERFACE s_axilite port = n_frag
#pragma HLS INTERFACE s_axilite port = return
    const int N = svgen::N_CAP;
    static uint16_t part[SV_FMAX][svgen::N_CAP];
    static int16_t ev[SV_FMAX][svgen::N_CAP];
    static int np[SV_FMAX];
    static sv::BF marg[SV_FMAX][svgen::N_CAP];
    static sv::PostState<svgen::N_CAP> pst;
    if (n_frag > SV_FMAX) n_frag = SV_FMAX;
    for (int f = 0; f < n_frag; ++f) {
        np[f] = (int)npart[f];
        if (np[f] > N) np[f] = N;
        for (int k = 0; k < N; ++k) {
#pragma HLS PIPELINE II = 1
            part[f][k] = (uint16_t)(partners[f * N + k] & 0xFFFF);
            ev[f][k] = (int16_t)(evidence[f * N + k] & 0xFFFF);
        }
    }
    for (int f = 0; f < n_frag; ++f) sv::esp_marginals<svgen::N_CAP>(ev[f], np[f], 0, marg[f], pst);
    for (int f = 0; f < n_frag; ++f) {
        int32_t cand[svgen::N_CAP];
        int64_t p39[svgen::N_CAP];
        int64_t rho39[svgen::N_CAP];
        int64_t phi39[4 * svgen::N_CAP];
        for (int k = 0; k < np[f]; ++k) {
            int g = part[f][k];
            cand[k] = g;
            sv::BF a = marg[f][k];
            sv::BF b = sv::bf_zero();
            if (g < n_frag) {
                int kk = find_partner(part[g], np[g], f);
                if (kk >= 0) b = marg[g][kk];
            }
            sv::BF mu = sv::bf_cmp(a, b) <= 0 ? a : b;
            int64_t q = sv::bf_to_fixed(mu, 39);
            p39[k] = q > ((int64_t)1 << 39) ? ((int64_t)1 << 39) : q;
            rho39[k] = (int64_t)(int32_t)rho[g] << 16;
            for (int t = 0; t < 4; ++t) phi39[4 * k + t] = (int64_t)(int32_t)phi[4 * g + t] << 16;
        }
        sv::TrustRowOut<svgen::N_CAP> out;
        sv::trust_solve_row<svgen::N_CAP>((int64_t)(int32_t)need[f] << 16, cand, p39, rho39, phi39, np[f], svgen::KQP, out);
        uint16_t list[svgen::MS];
        sv::trust_source_list<svgen::N_CAP>(out, list);
        for (int s = 0; s < svgen::MS; ++s) {
            int64_t l = 0;
            for (int i = 0; i < out.m; ++i)
                if (list[s] != 0xFFFF && out.sources[i] == list[s]) l = out.lam[i];
            sources[f * svgen::MS + s] = list[s];
            lam[2 * (f * svgen::MS + s)] = (uint32_t)((uint64_t)l & 0xFFFFFFFFu);
            lam[2 * (f * svgen::MS + s) + 1] = (uint32_t)((uint64_t)l >> 32);
        }
        status[f] = (uint32_t)out.status | ((uint32_t)out.iterations << 8) | ((uint32_t)out.screened << 16);
    }
}

extern "C" void sv_exchange2(const uint32_t* sources, const uint32_t* lam, const uint32_t* tv, const uint32_t* h, uint32_t* dv,
                             uint32_t* stats, int n_frag) {
#pragma HLS INTERFACE m_axi port = sources offset = slave bundle = gmem0 depth = 2048
#pragma HLS INTERFACE m_axi port = lam offset = slave bundle = gmem0 depth = 4096
#pragma HLS INTERFACE m_axi port = tv offset = slave bundle = gmem0 depth = 24576
#pragma HLS INTERFACE m_axi port = h offset = slave bundle = gmem0 depth = 3072
#pragma HLS INTERFACE m_axi port = dv offset = slave bundle = gmem1 depth = 3072
#pragma HLS INTERFACE m_axi port = stats offset = slave bundle = gmem1 depth = 2
#pragma HLS INTERFACE s_axilite port = n_frag
#pragma HLS INTERFACE s_axilite port = return
    const int W = svgen::C_TRANSPORT;
    const int WW = W / 4;
    uint32_t sat = 0;
    if (n_frag > SV_FMAX) n_frag = SV_FMAX;
    for (int f = 0; f < n_frag; ++f) {
        int8_t hv[svgen::C_TRANSPORT];
        int8_t out[svgen::C_TRANSPORT];
        for (int q = 0; q < WW; ++q) {
            uint32_t w = h[f * WW + q];
            for (int b = 0; b < 4; ++b) hv[4 * q + b] = (int8_t)((w >> (8 * b)) & 0xFF);
        }
        for (int c = 0; c < W; ++c) {
            i128 acc = 0;
            for (int s = 0; s < svgen::MS; ++s) {
                if ((sources[f * svgen::MS + s] & 0xFFFF) == 0xFFFF) continue;
                uint64_t lo = lam[2 * (f * svgen::MS + s)], hi = lam[2 * (f * svgen::MS + s) + 1];
                int64_t l = (int64_t)(lo | (hi << 32));
                uint32_t w = tv[(f * svgen::MS + s) * WW + c / 4];
                int8_t t8 = (int8_t)((w >> (8 * (c % 4))) & 0xFF);
                acc += (i128)l * ((int64_t)t8 - (int64_t)hv[c]);
            }
            int64_t v = sv_to_i64(sv::rshift_rne128(acc, svgen::TRUST_FRAC));
            if (v > 127 || v < -127) { ++sat; v = v > 127 ? 127 : -127; }
            out[c] = (int8_t)v;
        }
        for (int q = 0; q < WW; ++q) {
            uint32_t w = 0;
            for (int b = 0; b < 4; ++b) w |= ((uint32_t)(uint8_t)out[4 * q + b]) << (8 * b);
            dv[f * WW + q] = w;
        }
    }
    stats[0] = sat;
    stats[1] = 0;
}
