// Screened trust solve of Eq. (2) in Q8.39 (Definition 1, Prop. 2) and the dev-profile
// one-hop innovation.  Twin of reference/splicev_ref/trust.py: identical operation order;
// every product is exact in 128 bits and rounded half-even once at its manifest site.
#ifndef SV_TRUST_H
#define SV_TRUST_H

#include "sv_int.h"
#include "splicev_gen.h"

namespace sv {

static const int TQ = svgen::TRUST_FRAC;
static const int64_t TONE = (int64_t)1 << TQ;

enum TrustStatus { T_EMPTY = 0, T_CONVERGED = 1, T_CAPPED = 2, T_SINGULAR = 3 };

struct TrustCounters {
    uint32_t screened, ms_cap, iteration_cap, clamp_lo, clamp_hi, D_clamp, phi_scaled, q_sat, singular, rho_nonpos, need_clamp;
};

static inline int64_t tq_mul(int64_t a, int64_t b) { return sv_to_i64(rshift_rne128((i128)a * (i128)b, TQ)); }
static inline int64_t tq_div(int64_t a, int64_t b) { return sv_to_i64(div_rne128(((i128)a) << TQ, (i128)b)); }
static inline int64_t tq_sat(i128 x, TrustCounters& c) {
    if (x > (i128)svgen::Q39_MAX) { c.q_sat++; return svgen::Q39_MAX; }
    if (x < (i128)svgen::Q39_MIN) { c.q_sat++; return svgen::Q39_MIN; }
    return sv_to_i64(x);
}

// Solve A x = rhs for SPD A (m x m, Q8.39) with rhs scaled by 2^s; false if a pivot is not positive.
template <int MS>
static bool ldl_solve(const int64_t a[MS][MS], int m, const i128* rhs, i128* x) {
    i128 L[MS][MS];
    i128 T[MS][MS];
    i128 D[MS];
    i128 y[MS];
    i128 z[MS];
    for (int j = 0; j < m; ++j) {
        i128 s = ((i128)a[j][j]) << TQ;
        for (int k = 0; k < j; ++k) s -= L[j][k] * T[j][k];
        i128 dj = rshift_rne128(s, TQ);
        if (dj <= 0) return false;
        D[j] = dj;
        for (int i = j + 1; i < m; ++i) {
            i128 t = 0;
            for (int k = 0; k < j; ++k) t += L[i][k] * T[j][k];
            L[i][j] = div_rne128((((i128)a[i][j]) - rshift_rne128(t, TQ)) << TQ, dj);
            T[i][j] = rshift_rne128(L[i][j] * dj, TQ);
        }
    }
    for (int i = 0; i < m; ++i) {
        i128 t = 0;
        for (int k = 0; k < i; ++k) t += L[i][k] * y[k];
        y[i] = rhs[i] - rshift_rne128(t, TQ);
    }
    for (int i = 0; i < m; ++i) z[i] = div_rne128(y[i] << TQ, D[i]);
    for (int i = m - 1; i >= 0; --i) {
        i128 t = 0;
        for (int k = i + 1; k < m; ++k) t += L[k][i] * x[k];
        x[i] = z[i] - rshift_rne128(t, TQ);
    }
    return true;
}

template <int NMAX>
struct TrustRowOut {
    int status;
    int iterations;
    int screened;
    int m;                       // survivors
    int32_t sources[svgen::MS];  // survivor ids in survivor order
    int64_t lam[svgen::MS];      // Q8.39
    TrustCounters c;
};

// need (Q8.39), cand[n] source ids, p/rho (Q8.39), phi n x 4 (Q8.39)
template <int NMAX>
static void trust_solve_row(int64_t need, const int32_t* cand, const int64_t* p, const int64_t* rho,
                            const int64_t* phi, int n, int kqp, TrustRowOut<NMAX>& out) {
    const int MS = svgen::MS;
    TrustCounters& c = out.c;
    c.screened = c.ms_cap = c.iteration_cap = c.clamp_lo = c.clamp_hi = c.D_clamp = 0;
    c.phi_scaled = c.q_sat = c.singular = c.rho_nonpos = c.need_clamp = 0;
    out.status = T_EMPTY;
    out.iterations = 0;
    out.screened = 0;
    out.m = 0;
    if (n == 0) return;
    if (need < 0) { need = 0; c.need_clamp++; }
    int64_t dval[NMAX];
    int64_t ph[NMAX][4];
    for (int k = 0; k < n; ++k) {
        int64_t dk;
        if (rho[k] > 0) dk = tq_div(svgen::A_R_Q39, rho[k]) + tq_mul(svgen::B_R_Q39, TONE - p[k]);
        else { dk = svgen::D_MAX_Q39; c.rho_nonpos++; }
        if (dk < svgen::D_MIN_Q39) { dk = svgen::D_MIN_Q39; c.D_clamp++; }
        else if (dk > svgen::D_MAX_Q39) { dk = svgen::D_MAX_Q39; c.D_clamp++; }
        dval[k] = dk;
        for (int t = 0; t < 4; ++t) ph[k][t] = phi[4 * k + t];
    }
    const i128 limit = ((i128)1) << (svgen::PHI_FRO2_MAX_LOG2 + 2 * TQ);
    bool scaled = false;
    for (int guard = 0; guard < 64; ++guard) {
        i128 fro = 0;
        for (int k = 0; k < n; ++k)
            for (int t = 0; t < 4; ++t) fro += (i128)ph[k][t] * ph[k][t];
        if (fro <= limit) break;
        for (int k = 0; k < n; ++k)
            for (int t = 0; t < 4; ++t) ph[k][t] = rshift_rne64(ph[k][t], 1);
        scaled = true;
    }
    if (scaled) c.phi_scaled++;
    int64_t norm[NMAX];
    int64_t nmax = 0;
    for (int k = 0; k < n; ++k) {
        u128 s2 = 0;
        for (int t = 0; t < 4; ++t) s2 += (u128)((i128)ph[k][t] * ph[k][t]);
        norm[k] = (int64_t)sv_to_u64(isqrt_ceil(s2));
        if (norm[k] > nmax) nmax = norm[k];
    }
    int keep[NMAX];
    int nk = 0;
    for (int k = 0; k < n; ++k) {
        i128 lhs = (i128)svgen::C_R_Q39 * (i128)(TONE - p[k]);
        i128 rhs = (i128)need * TONE + (i128)norm[k] * nmax;
        if (lhs > rhs) out.screened++;
        else keep[nk++] = k;
    }
    c.screened = (uint32_t)out.screened;
    for (int i = 1; i < nk; ++i) {                 // insertion sort: p descending, id ascending
        int v = keep[i];
        int j = i - 1;
        while (j >= 0 && (p[keep[j]] < p[v] || (p[keep[j]] == p[v] && cand[keep[j]] > cand[v]))) {
            keep[j + 1] = keep[j];
            --j;
        }
        keep[j + 1] = v;
    }
    if (nk > MS) { c.ms_cap++; nk = MS; }
    out.m = nk;
    for (int i = 0; i < nk; ++i) { out.sources[i] = cand[keep[i]]; out.lam[i] = 0; }
    if (nk == 0) { out.status = T_CONVERGED; return; }
    int m = nk;
    int64_t A[svgen::MS][svgen::MS];
    int64_t b[svgen::MS];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            i128 dot = 0;
            for (int t = 0; t < 4; ++t) dot += (i128)ph[keep[i]][t] * ph[keep[j]][t];
            i128 v = (i128)need + rshift_rne128(dot, TQ);
            if (i == j) v += dval[keep[i]];
            A[i][j] = tq_sat(v, c);
        }
        b[i] = tq_sat((i128)(-need) + tq_mul(svgen::C_R_Q39, TONE - p[keep[i]]), c);
    }
    int64_t lam[svgen::MS];
    bool active[svgen::MS];
    for (int i = 0; i < m; ++i) { lam[i] = 0; active[i] = true; }
    int status = T_CAPPED;
    int steps = 0;
    const int64_t tol = svgen::KKT_TOL_Q39;
    while (true) {
        int64_t grad[svgen::MS];
        int64_t resid = 0;
        for (int i = 0; i < m; ++i) {
            i128 s = 0;
            for (int j = 0; j < m; ++j) s += (i128)A[i][j] * lam[j];
            grad[i] = sv_to_i64(rshift_rne128(s, TQ)) + b[i];
            int64_t r = active[i] ? (grad[i] < 0 ? -grad[i] : 0) : (grad[i] < 0 ? -grad[i] : grad[i]);
            if (r > resid) resid = r;
        }
        if (resid <= tol) { status = T_CONVERGED; break; }
        if (steps >= kqp) { status = T_CAPPED; c.iteration_cap++; break; }
        steps += 1;
        int fr[svgen::MS];
        int nf = 0;
        int64_t gm = 0;
        for (int i = 0; i < m; ++i) {
            if (!active[i]) {
                fr[nf++] = i;
                int64_t ag = grad[i] < 0 ? -grad[i] : grad[i];
                if (ag > gm) gm = ag;
            }
        }
        if (nf == 0 || gm <= tol) {
            int j = -1;
            for (int i = 0; i < m; ++i) {
                if (!active[i]) continue;
                if (j < 0 || grad[i] < grad[j]) j = i;
            }
            active[j] = false;
            continue;
        }
        int s = 46 - bitlen64((uint64_t)gm);
        if (s > 40) s = 40;
        if (s < 0) s = 0;
        int64_t sub[svgen::MS][svgen::MS];
        i128 rhs[svgen::MS];
        i128 ps[svgen::MS];
        for (int u = 0; u < nf; ++u) {
            for (int v = 0; v < nf; ++v) sub[u][v] = A[fr[u]][fr[v]];
            rhs[u] = ((i128)(-grad[fr[u]])) << s;
        }
        if (!ldl_solve<svgen::MS>(sub, nf, rhs, ps)) { status = T_SINGULAR; c.singular++; break; }
        int bt = -1;
        for (int t = 0; t < nf; ++t) {
            if (ps[t] >= 0) continue;
            if (bt < 0) { bt = t; continue; }
            i128 lhs = (i128)lam[fr[t]] * (-ps[bt]);
            i128 rhs2 = (i128)lam[fr[bt]] * (-ps[t]);
            if (lhs < rhs2 || (lhs == rhs2 && fr[t] < fr[bt])) bt = t;
        }
        bool full = (bt < 0) || ((((i128)lam[fr[bt]]) << s) >= -ps[bt]);
        i128 nl[svgen::MS];
        for (int i = 0; i < m; ++i) nl[i] = lam[i];
        if (full) {
            for (int t = 0; t < nf; ++t) nl[fr[t]] += rshift_rne128(ps[t], s);
        } else {
            i128 lj = lam[fr[bt]];
            for (int t = 0; t < nf; ++t)
                if (t != bt) nl[fr[t]] += div_rne128(lj * ps[t], -ps[bt]);
            nl[fr[bt]] = 0;
            active[fr[bt]] = true;
        }
        for (int i = 0; i < m; ++i) {
            if (nl[i] < 0) { nl[i] = 0; c.clamp_lo++; }
            else if (nl[i] > (i128)svgen::LAMBDA_MAX_Q39) { nl[i] = svgen::LAMBDA_MAX_Q39; c.clamp_hi++; }
            lam[i] = sv_to_i64(nl[i]);
        }
    }
    for (int i = 0; i < m; ++i) out.lam[i] = lam[i];
    out.status = status;
    out.iterations = steps;
}

// source list: ids with positive trust, ascending, padded with 0xFFFF
template <int NMAX>
static void trust_source_list(const TrustRowOut<NMAX>& r, uint16_t* list) {
    int cnt = 0;
    int32_t ids[svgen::MS];
    for (int i = 0; i < r.m; ++i)
        if (r.lam[i] > 0) ids[cnt++] = r.sources[i];
    for (int i = 1; i < cnt; ++i) {
        int32_t v = ids[i];
        int j = i - 1;
        while (j >= 0 && ids[j] > v) { ids[j + 1] = ids[j]; --j; }
        ids[j + 1] = v;
    }
    for (int i = 0; i < svgen::MS; ++i) list[i] = i < cnt ? (uint16_t)ids[i] : (uint16_t)0xFFFF;
}

// dev-profile innovation: dv_c = sat_sym(rne(sum_slot lam_slot (tv_slot,c - h_c) >> 39))
template <int NMAX>
static void trust_innovation(const TrustRowOut<NMAX>& r, const uint16_t* list, const int8_t* tv /* MS x width */,
                             const int8_t* h, int width, int8_t* dv, uint32_t& sat) {
    for (int ch = 0; ch < width; ++ch) {
        i128 acc = 0;
        for (int slot = 0; slot < svgen::MS; ++slot) {
            if (list[slot] == 0xFFFF) continue;
            int64_t l = 0;
            for (int i = 0; i < r.m; ++i) if (r.sources[i] == list[slot]) l = r.lam[i];
            acc += (i128)l * ((int64_t)tv[slot * width + ch] - (int64_t)h[ch]);
        }
        int64_t v = sv_to_i64(rshift_rne128(acc, TQ));
        if (v > 127 || v < -127) { sat++; v = v > 127 ? 127 : -127; }
        dv[ch] = (int8_t)v;
    }
}

}  // namespace sv

#endif  // SV_TRUST_H
