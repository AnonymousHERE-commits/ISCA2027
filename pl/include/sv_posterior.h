// Partner posterior by elementary symmetric polynomials (Eq. (1), Prop. 2) in BF24.
// Twin of reference/splicev_ref/posterior.py: identical operation order.
#ifndef SV_POSTERIOR_H
#define SV_POSTERIOR_H

#include "sv_bf24.h"

namespace sv {

static const int KH_MAX = svgen::KMAX;

static inline uint64_t comb_small(int n, int k) {
    uint64_t r = 1;
    for (int i = 0; i < k; ++i) r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);
    return r;
}

template <int NMAX>
struct PostState {
    BF w[NMAX];
    BF pre[NMAX + 1][KH_MAX + 1];
    BF suf[NMAX + 2][KH_MAX + 1];
};

// evidence: n codes (Q7.8); out: n marginals.  stratum selects the prior row.
template <int NMAX>
static void esp_marginals(const int16_t* evidence, int n, int stratum, BF* out, PostState<NMAX>& st) {
    if (n <= 0) return;
    BF one = bf_from_int(1);
    int kh = n < svgen::KMAX ? n : svgen::KMAX;
    for (int u = 0; u < n; ++u) st.w[u] = bf_exp_q78(evidence[u]);
    st.pre[0][0] = one;
    for (int k = 1; k <= kh; ++k) st.pre[0][k] = bf_zero();
    for (int u = 1; u <= n; ++u) {
        st.pre[u][0] = one;
        for (int k = 1; k <= kh; ++k) st.pre[u][k] = bf_add(st.pre[u - 1][k], bf_mul(st.w[u - 1], st.pre[u - 1][k - 1]));
    }
    st.suf[n + 1][0] = one;
    for (int k = 1; k <= kh; ++k) st.suf[n + 1][k] = bf_zero();
    for (int u = n; u >= 1; --u) {
        st.suf[u][0] = one;
        for (int k = 1; k <= kh; ++k) st.suf[u][k] = bf_add(st.suf[u + 1][k], bf_mul(st.w[u - 1], st.suf[u + 1][k - 1]));
    }
    BF coef[KH_MAX + 1];
    for (int k = 1; k <= kh; ++k) {
        BF pr;
        pr.m = svgen::PRIOR_BF_M[stratum * svgen::KMAX + k - 1];
        pr.e = svgen::PRIOR_BF_E[stratum * svgen::KMAX + k - 1];
        coef[k] = bf_mul(pr, bf_recip(bf_from_int(comb_small(n, k))));
    }
    BF z = bf_zero();
    for (int k = 1; k <= kh; ++k) z = bf_add(z, bf_mul(coef[k], st.pre[n][k]));
    BF rz = bf_recip(z);
    for (int u = 1; u <= n; ++u) {
        BF acc = bf_zero();
        for (int k = 1; k <= kh; ++k) {
            BF e = bf_zero();
            for (int i = 0; i < k; ++i) e = bf_add(e, bf_mul(st.pre[u - 1][i], st.suf[u + 1][k - 1 - i]));
            acc = bf_add(acc, bf_mul(coef[k], e));
        }
        out[u - 1] = bf_mul(bf_mul(st.w[u - 1], acc), rz);
    }
}

}  // namespace sv

#endif  // SV_POSTERIOR_H
