// C-simulation testbench of the PL kernel tops (Vitis HLS csim and host g++).
// Define one of TB_RESERVE, TB_LINKAGE, TB_GATE, TB_EXCHANGE (HLS component builds compile only
// that kernel) or TB_ALL (host build).  Vectors: $SPLICEV_VECTORS/engines/pl_kernels.svb.
#include <cstdio>
#include <vector>

#include "svb.h"
#include "sv_kernels.h"

#ifdef TB_ALL
#define TB_RESERVE
#define TB_LINKAGE
#define TB_GATE
#define TB_EXCHANGE
#endif

static int g_bad = 0;

static bool same(const char* what, int c, const std::vector<uint32_t>& got, const svb::Tensor& want, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != (uint32_t)want.at(i)) {
            std::printf("  case %d %s: first mismatch at %zu: got %u want %lld\n", c, what, i, got[i], (long long)want.at(i));
            g_bad = 1;
            return false;
        }
    }
    return true;
}

static std::vector<uint32_t> words(const svb::Tensor& t, size_t pad_to = 0) {
    std::vector<uint32_t> v(t.size() > pad_to ? t.size() : pad_to, 0);
    for (size_t i = 0; i < t.size(); ++i) v[i] = (uint32_t)t.at(i);
    return v;
}

int main() {
    svb::Bundle b;
    std::string path = svb::vectors_dir() + "/engines/pl_kernels.svb";
    if (!b.load(path)) { std::printf("FAIL: cannot read %s (run python3 tools/make_vectors.py)\n", path.c_str()); return 2; }
    int n = (int)b.scalar("n_cases");
    int ok[4] = {0, 0, 0, 0};
    for (int c = 0; c < n; ++c) {
        std::string p = "c" + std::to_string(c) + ".";
#ifdef TB_RESERVE
        {
            const auto& a = b[p + "reserve.args"];
            std::vector<uint32_t> cells = words(b[p + "reserve.cells"]), picks(svgen::N_RESERVE), stats(4);
            sv_reserve(cells.data(), picks.data(), stats.data(), (int)a.at(0), (int)a.at(1), (int)a.at(2));
            const auto& w = b[p + "reserve.picks"];
            ok[0] += stats[0] == w.size() && same("reserve.picks", c, picks, w, w.size());
        }
#endif
#ifdef TB_LINKAGE
        {
            const auto& a = b[p + "linkage.args"];
            int na = (int)a.at(0);
            std::vector<uint32_t> ed = words(b[p + "linkage.edges"]), an = words(b[p + "linkage.anchors"]);
            std::vector<uint32_t> ids(na), order(na), stats(8);
            sv_linkage(ed.data(), an.data(), ids.data(), order.data(), stats.data(), na, (int)a.at(1), (int)a.at(2), (int)a.at(3));
            bool g = same("linkage.ids", c, ids, b[p + "linkage.ids"], na) && same("linkage.order", c, order, b[p + "linkage.order"], na)
                     && same("linkage.stats", c, stats, b[p + "linkage.stats"], 8);
            ok[1] += g;
        }
#endif
#ifdef TB_GATE
        {
            const auto& a = b[p + "gate.args"];
            std::vector<uint32_t> an = words(b[p + "gate.anchors"]), mask(svgen::F_CAP * svgen::F_CAP / 32), fused(12 * svgen::F_CAP), stats(2);
            sv_gate(an.data(), mask.data(), fused.data(), stats.data(), (int)a.at(0), (int)a.at(1));
            ok[2] += same("gate.mask", c, mask, b[p + "gate.mask"], mask.size()) && stats[0] == (uint32_t)b.scalar(p + "gate.pairs");
        }
#endif
#ifdef TB_EXCHANGE
        {
            int F = (int)b.scalar(p + "ex1.args");
            std::vector<uint32_t> pa = words(b[p + "ex1.partners"]), ev = words(b[p + "ex1.evidence"]), np = words(b[p + "ex1.npart"]);
            std::vector<uint32_t> nd = words(b[p + "ex1.need"]), rh = words(b[p + "ex1.rho"]), ph = words(b[p + "ex1.phi"]);
            std::vector<uint32_t> src(svgen::F_CAP * svgen::MS), lam(2 * svgen::F_CAP * svgen::MS), st(svgen::F_CAP);
            sv_exchange1(pa.data(), ev.data(), np.data(), nd.data(), rh.data(), ph.data(), src.data(), lam.data(), st.data(), F);
            bool g = same("ex1.sources", c, src, b[p + "ex1.sources"], (size_t)F * svgen::MS)
                     && same("ex1.lam", c, lam, b[p + "ex1.lam"], (size_t)2 * F * svgen::MS) && same("ex1.status", c, st, b[p + "ex1.status"], F);
            std::vector<uint32_t> s2 = words(b[p + "ex2.sources"]), l2 = words(b[p + "ex2.lam"]), tv = words(b[p + "ex2.tv"]), h = words(b[p + "ex2.h"]);
            std::vector<uint32_t> dv(svgen::F_CAP * svgen::C_TRANSPORT / 4), st2(2);
            sv_exchange2(s2.data(), l2.data(), tv.data(), h.data(), dv.data(), st2.data(), F);
            g = g && same("ex2.dv", c, dv, b[p + "ex2.dv"], (size_t)F * svgen::C_TRANSPORT / 4);
            ok[3] += g;
        }
#endif
    }
    const char* names[4] = {"sv_reserve", "sv_linkage", "sv_gate", "sv_exchange1/2"};
    bool enabled[4] = {
#ifdef TB_RESERVE
        true,
#else
        false,
#endif
#ifdef TB_LINKAGE
        true,
#else
        false,
#endif
#ifdef TB_GATE
        true,
#else
        false,
#endif
#ifdef TB_EXCHANGE
        true,
#else
        false,
#endif
    };
    for (int k = 0; k < 4; ++k)
        if (enabled[k]) std::printf("%-15s %d/%d cases bit-exact\n", names[k], ok[k], n);
    std::printf(g_bad ? "FAIL\n" : "PASS\n");
    return g_bad;
}
