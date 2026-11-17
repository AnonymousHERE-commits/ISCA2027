// sv_gate: precision-weighted fusion of anchor hypotheses and the admissible mask G,
// run once as soon as the linkage names the fragments; the forecast and the exchange
// read the same mask.
#include "sv_kernels.h"
#include "sv_gate.h"

extern "C" void sv_gate(const uint32_t* anchors, uint32_t* mask, uint32_t* fused, uint32_t* stats, int n, int n_frag) {
#pragma HLS INTERFACE m_axi port = anchors offset = slave bundle = gmem0 depth = 9280
#pragma HLS INTERFACE m_axi port = mask offset = slave bundle = gmem1 depth = 2048
#pragma HLS INTERFACE m_axi port = fused offset = slave bundle = gmem1 depth = 3072
#pragma HLS INTERFACE m_axi port = stats offset = slave bundle = gmem1 depth = 2
#pragma HLS INTERFACE s_axilite port = n
#pragma HLS INTERFACE s_axilite port = n_frag
#pragma HLS INTERFACE s_axilite port = return
    static int16_t mu[4 * SV_GATE_NMAX];
    static uint16_t prec[4 * SV_GATE_NMAX];
    static uint16_t ids[SV_GATE_NMAX];
    static uint32_t m[svgen::F_CAP * sv::G_ROW_WORDS];
    static sv::GateState<SV_FMAX> st;
    if (n > SV_GATE_NMAX) n = SV_GATE_NMAX;
    if (n_frag > SV_FMAX) n_frag = SV_FMAX;
    for (int a = 0; a < n; ++a) {
#pragma HLS PIPELINE II = 5
        uint32_t w0 = anchors[5 * a], w1 = anchors[5 * a + 1], w2 = anchors[5 * a + 2], w3 = anchors[5 * a + 3];
        mu[4 * a] = (int16_t)(w0 & 0xFFFF); mu[4 * a + 1] = (int16_t)(w0 >> 16);
        mu[4 * a + 2] = (int16_t)(w1 & 0xFFFF); mu[4 * a + 3] = (int16_t)(w1 >> 16);
        prec[4 * a] = (uint16_t)(w2 & 0xFFFF); prec[4 * a + 1] = (uint16_t)(w2 >> 16);
        prec[4 * a + 2] = (uint16_t)(w3 & 0xFFFF); prec[4 * a + 3] = (uint16_t)(w3 >> 16);
        ids[a] = (uint16_t)(anchors[5 * a + 4] & 0xFFFF);
    }
    sv::gate_fuse<SV_FMAX>(mu, prec, ids, n, n_frag, st);
    int pairs = sv::gate_mask<SV_FMAX>(st, n_frag, m);
    for (int i = 0; i < svgen::F_CAP * sv::G_ROW_WORDS; ++i) {
#pragma HLS PIPELINE II = 1
        mask[i] = m[i];
    }
    for (int f = 0; f < svgen::F_CAP; ++f) {
        for (int d = 0; d < 4; ++d) {
            bool v = f < n_frag;
            fused[12 * f + d] = v ? (uint32_t)(int32_t)st.mu_f[f][d] : 0u;
            fused[12 * f + 4 + d] = v ? st.var_f[f][d].m : 0u;
            fused[12 * f + 8 + d] = v ? (uint32_t)st.var_f[f][d].e : 0u;
        }
    }
    stats[0] = (uint32_t)pairs;
    stats[1] = (uint32_t)n_frag;
}
