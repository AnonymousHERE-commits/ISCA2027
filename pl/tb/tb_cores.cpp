// Host conformance test of the PL engine cores against the integer reference vectors.
// Build and run: make host-pl   (or: SPLICEV_VECTORS=<dir> build/host/tb_cores)
// Every comparison is bit exact; the program exits non-zero on the first mismatching case
// after printing counts and the first mismatch position.
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "svb.h"
#include "sv_bf24.h"
#include "sv_gate.h"
#include "sv_linkage.h"
#include "sv_posterior.h"
#include "sv_reserve.h"
#include "sv_trust.h"

static int g_fail = 0;

static void report(const char* what, int ok, int total) {
    std::printf("%-12s %d/%d cases bit-exact\n", what, ok, total);
    if (ok != total) g_fail = 1;
}

static int test_reserve(const std::string& dir) {
    svb::Bundle b;
    if (!b.load(dir + "/engines/reserve.svb")) return 2;
    int n = (int)b.scalar("n_cases"), ok = 0;
    static sv::ReserveState<6400> st;
    for (int c = 0; c < n; ++c) {
        std::string p = "c" + std::to_string(c) + ".";
        const auto& e = b[p + "e"];
        const auto& t = b[p + "t"];
        std::vector<uint8_t> e8(e.size()), t8(t.size());
        for (size_t i = 0; i < e.size(); ++i) { e8[i] = (uint8_t)e.at(i); t8[i] = (uint8_t)t.at(i); }
        std::vector<uint16_t> picks(1024);
        uint64_t scans = 0;
        int np = sv::reserve_select<6400>(e8.data(), t8.data(), (int)e.size(), (int)b.scalar(p + "width"),
                                          (int)b.scalar(p + "count"), picks.data(), st, scans);
        const auto& want = b[p + "picks"];
        bool good = np == (int)want.size();
        for (int i = 0; good && i < np; ++i) good = picks[i] == want.at(i);
        if (!good) std::printf("  reserve case %d: mismatch (got %d picks, want %zu)\n", c, np, want.size());
        ok += good;
    }
    report("reserve", ok, n);
    return 0;
}

static int test_linkage(const std::string& dir) {
    svb::Bundle b;
    if (!b.load(dir + "/engines/linkage.svb")) return 2;
    int n = (int)b.scalar("n_cases"), ok = 0;
    const int NMAX = 1856, NBCAP = 2 * svgen::LINK_NBMAX - 2;
    std::unique_ptr<sv::LinkState<NMAX, NBCAP>> st(new sv::LinkState<NMAX, NBCAP>());
    for (int c = 0; c < n; ++c) {
        std::string p = "c" + std::to_string(c) + ".";
        int na = (int)b.scalar(p + "n");
        const auto& E = b[p + "edges"];
        std::vector<int32_t> edges(E.size());
        for (size_t i = 0; i < E.size(); ++i) edges[i] = (int32_t)E.at(i);
        std::vector<uint8_t> active(na), elig(na);
        for (int i = 0; i < na; ++i) { active[i] = (uint8_t)b[p + "active"].at(i); elig[i] = (uint8_t)b[p + "elig"].at(i); }
        sv::Linker<NMAX, NBCAP> lk(*st);
        sv::LinkResult r;
        std::memset(&r, 0, sizeof(r));
        lk.init(na, edges.data(), (int)(E.size() / 3), active.data(), r);
        lk.run(b.scalar(p + "budget"), r);
        std::vector<uint16_t> ids(na), order(na);
        std::vector<int> counts(svgen::F_CAP + 2);
        int F = lk.select(active.data(), elig.data(), svgen::F_CAP, ids.data(), r);
        sv::anchor_order(ids.data(), na, F, order.data(), counts.data());
        bool good = F == b.scalar(p + "F") && r.rounds == b.scalar(p + "rounds") && r.work == b.scalar(p + "work")
                    && (int)r.budget_hit == b.scalar(p + "budget_hit") && (int)r.overflow == b.scalar(p + "overflow");
        int first_bad = -1;
        for (int i = 0; i < na; ++i) {
            if (ids[i] != b[p + "ids"].at(i) || order[i] != b[p + "order"].at(i)) { if (first_bad < 0) first_bad = i; good = false; }
        }
        if (!good)
            std::printf("  linkage case %d: F %d/%lld rounds %d/%lld work %lld/%lld budget %d/%lld first id mismatch %d\n", c, F,
                        (long long)b.scalar(p + "F"), r.rounds, (long long)b.scalar(p + "rounds"), (long long)r.work,
                        (long long)b.scalar(p + "work"), (int)r.budget_hit, (long long)b.scalar(p + "budget_hit"), first_bad);
        ok += good;
    }
    report("linkage", ok, n);
    return 0;
}

static int test_gate(const std::string& dir) {
    svb::Bundle b;
    if (!b.load(dir + "/engines/gate.svb")) return 2;
    int n = (int)b.scalar("n_cases"), ok = 0;
    std::unique_ptr<sv::GateState<256>> st(new sv::GateState<256>());
    std::vector<uint32_t> mask(svgen::F_CAP * sv::G_ROW_WORDS);
    for (int c = 0; c < n; ++c) {
        std::string p = "c" + std::to_string(c) + ".";
        const auto& mu = b[p + "mu"];
        const auto& pr = b[p + "prec"];
        const auto& ids = b[p + "ids"];
        int na = (int)ids.size(), F = (int)b.scalar(p + "F");
        std::vector<int16_t> mu16(mu.size());
        std::vector<uint16_t> pr16(pr.size()), id16(na);
        for (size_t i = 0; i < mu.size(); ++i) { mu16[i] = (int16_t)mu.at(i); pr16[i] = (uint16_t)pr.at(i); }
        for (int i = 0; i < na; ++i) id16[i] = (uint16_t)ids.at(i);
        sv::gate_fuse<256>(mu16.data(), pr16.data(), id16.data(), na, F, *st);
        sv::gate_mask<256>(*st, F, mask.data());
        bool good = true;
        for (int f = 0; f < F; ++f)
            for (int d = 0; d < 4; ++d) {
                good &= st->mu_f[f][d] == b[p + "mu_f"].at(4 * f + d);
                good &= st->var_f[f][d].m == (uint32_t)b[p + "var_m"].at(4 * f + d);
                good &= st->var_f[f][d].e == (int32_t)b[p + "var_e"].at(4 * f + d);
            }
        for (size_t i = 0; i < mask.size(); ++i) good &= mask[i] == (uint32_t)b[p + "mask"].at(i);
        if (!good) std::printf("  gate case %d mismatch\n", c);
        ok += good;
    }
    report("gate", ok, n);
    return 0;
}

static int test_posterior(const std::string& dir) {
    svb::Bundle b;
    if (!b.load(dir + "/engines/posterior.svb")) return 2;
    int n = (int)b.scalar("n_rows"), ok = 0;
    std::unique_ptr<sv::PostState<128>> st(new sv::PostState<128>());
    for (int r = 0; r < n; ++r) {
        std::string p = "r" + std::to_string(r) + ".";
        const auto& ev = b[p + "evidence"];
        std::vector<int16_t> e16(ev.size());
        for (size_t i = 0; i < ev.size(); ++i) e16[i] = (int16_t)ev.at(i);
        std::vector<sv::BF> out(ev.size() + 1);
        sv::esp_marginals<128>(e16.data(), (int)ev.size(), 0, out.data(), *st);
        bool good = true;
        for (size_t i = 0; i < ev.size(); ++i)
            good &= out[i].m == (uint32_t)b[p + "m"].at(i) && out[i].e == (int32_t)b[p + "e"].at(i);
        if (!good) std::printf("  posterior row %d mismatch\n", r);
        ok += good;
    }
    report("posterior", ok, n);
    return 0;
}

static int test_trust(const std::string& dir) {
    svb::Bundle b;
    if (!b.load(dir + "/engines/trust.svb")) return 2;
    int n = (int)b.scalar("n_rows"), ok = 0;
    for (int r = 0; r < n; ++r) {
        std::string p = "r" + std::to_string(r) + ".";
        const auto& cand = b[p + "cand"];
        int m = (int)cand.size();
        std::vector<int32_t> c32(m);
        std::vector<int64_t> pp(m), rho(m), phi(4 * m);
        for (int i = 0; i < m; ++i) { c32[i] = (int32_t)cand.at(i); pp[i] = b[p + "p"].at(i); rho[i] = b[p + "rho"].at(i); }
        for (int i = 0; i < 4 * m; ++i) phi[i] = b[p + "phi"].at(i);
        sv::TrustRowOut<128> out;
        sv::trust_solve_row<128>(b.scalar(p + "need"), c32.data(), pp.data(), rho.data(), phi.data(), m, (int)b.scalar(p + "kqp"), out);
        uint16_t list[svgen::MS];
        sv::trust_source_list<128>(out, list);
        bool good = out.status == b.scalar(p + "status") && out.iterations == b.scalar(p + "iterations")
                    && out.screened == b.scalar(p + "screened") && out.m == (int)b[p + "sources"].size();
        for (int i = 0; good && i < out.m; ++i) good = out.sources[i] == b[p + "sources"].at(i) && out.lam[i] == b[p + "lam"].at(i);
        for (int i = 0; good && i < svgen::MS; ++i) good = list[i] == b[p + "source_list"].at(i);
        if (!good) std::printf("  trust row %d mismatch (status %d/%lld, it %d/%lld)\n", r, out.status,
                               (long long)b.scalar(p + "status"), out.iterations, (long long)b.scalar(p + "iterations"));
        ok += good;
    }
    report("trust", ok, n);
    return 0;
}

int main() {
    std::string dir = svb::vectors_dir();
    std::printf("vectors: %s\n", dir.c_str());
    if (test_reserve(dir) || test_linkage(dir) || test_gate(dir) || test_posterior(dir) || test_trust(dir)) {
        std::printf("FAIL: could not read vectors (run: python3 tools/make_vectors.py)\n");
        return 2;
    }
    std::printf(g_fail ? "FAIL\n" : "PASS: all PL engine cores are bit-exact against the integer reference\n");
    return g_fail;
}
