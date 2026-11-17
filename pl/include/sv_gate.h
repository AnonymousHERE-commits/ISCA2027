// Early conformal gate: precision-weighted fusion of anchor hypotheses and the shared
// admissible mask G (DD-12).  Twin of reference/splicev_ref/gate.py.
#ifndef SV_GATE_H
#define SV_GATE_H

#include "sv_bf24.h"

namespace sv {

static const int G_ROW_WORDS = svgen::F_CAP / 32;

template <int FMAX>
struct GateState {
    int64_t psum[FMAX][4];
    int64_t msum[FMAX][4];
    int64_t mu_f[FMAX][4];
    BF var_f[FMAX][4];
};

static inline int gate_stratum(int n_frag) {
    int s = 0;
    if (n_frag >= svgen::GATE_STRATUM_BOUND_0) ++s;
    if (n_frag >= svgen::GATE_STRATUM_BOUND_1) ++s;
    if (n_frag >= svgen::GATE_STRATUM_BOUND_2) ++s;
    return s;
}

// mu: n x 4 int16 codes, prec: n x 4 u16 precision codes, ids: n fragment ids
template <int FMAX>
static void gate_fuse(const int16_t* mu, const uint16_t* prec, const uint16_t* ids, int n, int n_frag, GateState<FMAX>& st) {
    for (int f = 0; f < n_frag; ++f)
        for (int d = 0; d < 4; ++d) { st.psum[f][d] = 0; st.msum[f][d] = 0; }
    for (int a = 0; a < n; ++a) {
        int f = ids[a];
        if (f == svgen::UNASSIGNED || f >= n_frag) continue;
        for (int d = 0; d < 4; ++d) {
            st.psum[f][d] += prec[4 * a + d];
            st.msum[f][d] += (int64_t)prec[4 * a + d] * mu[4 * a + d];
        }
    }
    for (int f = 0; f < n_frag; ++f) {
        for (int d = 0; d < 4; ++d) {
            if (st.psum[f][d] > 0) {
                st.mu_f[f][d] = div_rne64(st.msum[f][d], st.psum[f][d]);
                st.var_f[f][d] = bf_recip(bf_from_int((uint64_t)st.psum[f][d]));
            } else {
                st.mu_f[f][d] = 0;
                st.var_f[f][d] = bf_zero();
            }
        }
    }
}

// mask: F_CAP * G_ROW_WORDS words, symmetric, zero diagonal; returns the admissible pair count
template <int FMAX>
static int gate_mask(const GateState<FMAX>& st, int n_frag, uint32_t* mask) {
    for (int i = 0; i < svgen::F_CAP * G_ROW_WORDS; ++i) mask[i] = 0;
    int s = gate_stratum(n_frag);
    BF tau;
    tau.m = svgen::GATE_TAU_BF_M[s];
    tau.e = svgen::GATE_TAU_BF_E[s];
    int count = 0;
    for (int f = 0; f < n_frag; ++f) {
        for (int g = f + 1; g < n_frag; ++g) {
            bool ok = true;
            BF acc = bf_zero();
            for (int d = 0; d < 4; ++d) {
                if (bf_is_zero(st.var_f[f][d]) || bf_is_zero(st.var_f[g][d])) { ok = false; break; }
                BF vs = bf_add(st.var_f[f][d], st.var_f[g][d]);
                int64_t diff = st.mu_f[f][d] - st.mu_f[g][d];
                acc = bf_add(acc, bf_mul(bf_from_int((uint64_t)(diff * diff)), bf_recip(vs)));
            }
            if (ok && bf_cmp(acc, tau) <= 0) {
                mask[f * G_ROW_WORDS + (g >> 5)] |= 1u << (g & 31);
                mask[g * G_ROW_WORDS + (f >> 5)] |= 1u << (f & 31);
                ++count;
            }
        }
    }
    return count;
}

}  // namespace sv

#endif  // SV_GATE_H
