// Guarded average linkage by certified mutual-nearest-neighbour rounds (Lemma 2),
// with the counted work budget (DD-11), fragment selection by eligibility mass and the
// compact anchor order.  Twin of reference/splicev_ref/linkage.py (certified, select_fragments,
// anchor_order).
//
// Capacity: NMAX anchors, NBCAP entries per cluster list.  Python checks list lengths
// against LINK_NBMAX at the start and after every round; inside a round a merged list can
// reach 2*NBMAX - 2, so NBCAP = 2*LINK_NBMAX - 2 reproduces the reference exactly.
#ifndef SV_LINKAGE_H
#define SV_LINKAGE_H

#include "sv_int.h"
#include "splicev_gen.h"

namespace sv {

struct LinkStats {
    uint32_t sum, cnt, hi, vhi;
};

struct LinkResult {
    int rounds;
    int64_t work;
    bool budget_hit;
    bool overflow;
    bool dup_error;
    int n_frag;
    bool cap_hit;
};

static inline bool lk_above(const LinkStats& s) { return 2ull * s.sum > (uint64_t)svgen::LINK_SCORE_DEN * s.cnt; }
static inline bool lk_guard(const LinkStats& s) { return s.hi >= (uint32_t)svgen::LINK_MIN_HI || s.vhi >= 1; }
static inline bool lk_eligible(const LinkStats& s) { return lk_above(s) && lk_guard(s); }
static inline bool lk_gfail(const LinkStats& s) { return lk_above(s) && !lk_guard(s); }

// (sa, ka) precedes (sb, kb): higher mean first, then smaller pair key (kmin, kmax)
static inline bool lk_precedes(const LinkStats& sa, uint32_t ka_lo, uint32_t ka_hi,
                               const LinkStats& sb, uint32_t kb_lo, uint32_t kb_hi) {
    uint64_t lhs = (uint64_t)sa.sum * sb.cnt;
    uint64_t rhs = (uint64_t)sb.sum * sa.cnt;
    if (lhs != rhs) return lhs > rhs;
    if (ka_lo != kb_lo) return ka_lo < kb_lo;
    return ka_hi < kb_hi;
}

template <int NMAX, int NBCAP>
struct LinkState {
    bool alive[NMAX];
    uint16_t key[NMAX];
    uint16_t parent[NMAX];
    uint16_t nbn[NMAX];
    uint16_t nb_c[NMAX][NBCAP];
    LinkStats nb_s[NMAX][NBCAP];
    int fe[NMAX];
    int fg[NMAX];
    uint16_t ca[NMAX];
    uint16_t cb[NMAX];
    int64_t mass[NMAX];
    int rank_tmp[NMAX];
    uint16_t cluster_of_key[NMAX];
};

template <int NMAX, int NBCAP>
class Linker {
public:
    LinkState<NMAX, NBCAP>& s;
    int n;
    explicit Linker(LinkState<NMAX, NBCAP>& st) : s(st), n(0) {}

    int find(int a) {
        while (s.parent[a] != a) {
            s.parent[a] = s.parent[s.parent[a]];
            a = s.parent[a];
        }
        return a;
    }

    int find_entry(int a, int c) {
        for (int j = 0; j < s.nbn[a]; ++j)
            if (s.nb_c[a][j] == c) return j;
        return -1;
    }

    void remove_entry(int a, int j) {
        int last = s.nbn[a] - 1;
        s.nb_c[a][j] = s.nb_c[a][last];
        s.nb_s[a][j] = s.nb_s[a][last];
        s.nbn[a] = (uint16_t)last;
    }

    bool append(int a, int c, const LinkStats& st) {
        if (s.nbn[a] >= NBCAP) return false;
        s.nb_c[a][s.nbn[a]] = (uint16_t)c;
        s.nb_s[a][s.nbn[a]] = st;
        s.nbn[a] += 1;
        return true;
    }

    void pair_key(int a, int b, uint32_t& lo, uint32_t& hi) {
        uint32_t ka = s.key[a], kb = s.key[b];
        lo = ka < kb ? ka : kb;
        hi = ka < kb ? kb : ka;
    }

    // edges: E x (a, b, q); active: n flags.  Returns false on a duplicate pair or capacity overflow.
    bool init(int n_anchor, const int32_t* edges, int n_edges, const uint8_t* active, LinkResult& r) {
        n = n_anchor;
        for (int i = 0; i < n; ++i) {
            s.alive[i] = active[i] != 0;
            s.key[i] = (uint16_t)i;
            s.parent[i] = (uint16_t)i;
            s.nbn[i] = 0;
        }
        for (int e = 0; e < n_edges; ++e) {
            int a = edges[3 * e], b = edges[3 * e + 1], q = edges[3 * e + 2];
            if (a == b || !active[a] || !active[b]) continue;
            if (find_entry(a, b) >= 0) { r.dup_error = true; return false; }
            LinkStats st;
            st.sum = (uint32_t)q; st.cnt = 1;
            st.hi = q >= svgen::LINK_HI ? 1u : 0u;
            st.vhi = q >= svgen::LINK_VHI ? 1u : 0u;
            if (!append(a, b, st) || !append(b, a, st)) { r.overflow = true; return false; }
        }
        for (int i = 0; i < n; ++i)
            if (s.alive[i] && s.nbn[i] > svgen::LINK_NBMAX) { r.overflow = true; return false; }
        return true;
    }

    void merge(int a, int b, LinkResult& r) {
        int keep = s.key[a] > s.key[b] ? a : b;
        int drop = keep == a ? b : a;
        for (int j = 0; j < s.nbn[drop]; ++j) {
            int c = s.nb_c[drop][j];
            if (c == keep) continue;
            LinkStats st = s.nb_s[drop][j];
            int jk = find_entry(keep, c);
            LinkStats merged;
            if (jk >= 0) {
                merged = s.nb_s[keep][jk];
                merged.sum += st.sum; merged.cnt += st.cnt; merged.hi += st.hi; merged.vhi += st.vhi;
                s.nb_s[keep][jk] = merged;
            } else {
                merged = st;
                if (!append(keep, c, merged)) { r.overflow = true; return; }
            }
            int jd = find_entry(c, drop);
            if (jd >= 0) remove_entry(c, jd);
            int jc = find_entry(c, keep);
            if (jc >= 0) s.nb_s[c][jc] = merged;
            else if (!append(c, keep, merged)) { r.overflow = true; return; }
        }
        int jdk = find_entry(keep, drop);
        if (jdk >= 0) remove_entry(keep, jdk);
        s.nbn[drop] = 0;
        s.alive[drop] = false;
        s.parent[drop] = (uint16_t)keep;
    }

    static int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

    // budget < 0 disables the budget
    void run(int64_t budget, LinkResult& r) {
        r.rounds = 0;
        r.work = 0;
        r.budget_hit = false;
        if (r.overflow || r.dup_error) return;
        while (true) {
            bool any = false;
            int64_t entries = 0, clusters = 0;
            for (int a = 0; a < n; ++a) {
                s.fe[a] = -1;
                s.fg[a] = -1;
                if (!s.alive[a]) continue;
                clusters += 1;
                entries += s.nbn[a];
                for (int j = 0; j < s.nbn[a]; ++j) {
                    const LinkStats& st = s.nb_s[a][j];
                    uint32_t lo, hi;
                    pair_key(a, s.nb_c[a][j], lo, hi);
                    if (lk_eligible(st)) {
                        if (s.fe[a] < 0) { s.fe[a] = j; continue; }
                        uint32_t blo, bhi;
                        pair_key(a, s.nb_c[a][s.fe[a]], blo, bhi);
                        if (lk_precedes(st, lo, hi, s.nb_s[a][s.fe[a]], blo, bhi)) s.fe[a] = j;
                    } else if (lk_gfail(st)) {
                        if (s.fg[a] < 0) { s.fg[a] = j; continue; }
                        uint32_t blo, bhi;
                        pair_key(a, s.nb_c[a][s.fg[a]], blo, bhi);
                        if (lk_precedes(st, lo, hi, s.nb_s[a][s.fg[a]], blo, bhi)) s.fg[a] = j;
                    }
                }
                if (s.fe[a] >= 0) any = true;
            }
            if (!any) break;
            if (budget >= 0 && r.work >= budget) { r.budget_hit = true; break; }
            r.rounds += 1;
            int ga = -1;
            uint32_t glo = 0, ghi = 0;
            for (int a = 0; a < n; ++a) {
                if (!s.alive[a] || s.fe[a] < 0) continue;
                uint32_t lo, hi;
                pair_key(a, s.nb_c[a][s.fe[a]], lo, hi);
                if (ga < 0) { ga = a; glo = lo; ghi = hi; continue; }
                if (lk_precedes(s.nb_s[a][s.fe[a]], lo, hi, s.nb_s[ga][s.fe[ga]], glo, ghi)) { ga = a; glo = lo; ghi = hi; }
            }
            int nc = 0;
            for (int a = 0; a < n; ++a) {
                if (!s.alive[a] || s.fe[a] < 0) continue;
                int b = s.nb_c[a][s.fe[a]];
                if (a >= b || s.fe[b] < 0 || s.nb_c[b][s.fe[b]] != a) continue;
                const LinkStats& st = s.nb_s[a][s.fe[a]];
                uint32_t lo, hi;
                pair_key(a, b, lo, hi);
                bool ok = (lo == glo && hi == ghi);
                if (!ok) {
                    ok = true;
                    int cs[2] = {a, b};
                    for (int u = 0; u < 2; ++u) {
                        int c = cs[u];
                        if (s.fg[c] < 0) continue;
                        uint32_t flo, fhi;
                        pair_key(c, s.nb_c[c][s.fg[c]], flo, fhi);
                        if (lk_precedes(s.nb_s[c][s.fg[c]], flo, fhi, st, lo, hi)) ok = false;
                    }
                }
                if (ok) { s.ca[nc] = (uint16_t)a; s.cb[nc] = (uint16_t)b; ++nc; }
            }
            int64_t merge_entries = 0;
            for (int i = 0; i < nc; ++i) merge_entries += s.nbn[s.ca[i]] + s.nbn[s.cb[i]];
            r.work += ceil_div(entries, svgen::LINK_SCAN_LANES) + svgen::LINK_PHASE_FILL
                    + ceil_div(clusters, svgen::LINK_COMMIT_LANES) + svgen::LINK_PHASE_FILL
                    + ceil_div(merge_entries, svgen::LINK_MERGE_UNITS) + svgen::LINK_PHASE_FILL;
            for (int i = 0; i < nc; ++i) {
                merge(s.ca[i], s.cb[i], r);
                if (r.overflow) return;
            }
            for (int a = 0; a < n; ++a) {
                if (s.alive[a] && s.nbn[a] > svgen::LINK_NBMAX) { r.overflow = true; return; }
            }
        }
    }

    // Fragment ids per anchor (UNASSIGNED for inactive or dropped) and the number of fragments.
    int select(const uint8_t* active, const uint8_t* elig, int f_cap, uint16_t* ids, LinkResult& r) {
        for (int a = 0; a < n; ++a) s.mass[a] = 0;
        for (int a = 0; a < n; ++a)
            if (active[a]) s.mass[find(a)] += elig[a];
        int clusters = 0;
        for (int a = 0; a < n; ++a) {
            s.cluster_of_key[a] = 0xFFFF;
            if (active[a] && find(a) == a) ++clusters;
        }
        for (int a = 0; a < n; ++a) {
            s.rank_tmp[a] = 0;
            if (!(active[a] && find(a) == a)) continue;
            int rk = 0;
            for (int b = 0; b < n; ++b) {
                if (b == a || !(active[b] && find(b) == b)) continue;
                if (s.mass[b] > s.mass[a] || (s.mass[b] == s.mass[a] && s.key[b] < s.key[a])) ++rk;
            }
            s.rank_tmp[a] = rk;
            if (rk < f_cap) s.cluster_of_key[s.key[a]] = (uint16_t)a;
        }
        r.cap_hit = clusters > f_cap;
        for (int a = 0; a < n; ++a) s.ca[a] = 0xFFFF;   // reuse as cluster -> fragment id
        int fid = 0;
        for (int k = 0; k < n; ++k) {
            if (s.cluster_of_key[k] != 0xFFFF) s.ca[s.cluster_of_key[k]] = (uint16_t)(fid++);
        }
        for (int a = 0; a < n; ++a) ids[a] = active[a] ? s.ca[find(a)] : (uint16_t)svgen::UNASSIGNED;
        r.n_frag = fid;
        return fid;
    }
};

// anchors sorted by (fragment id, anchor index); unassigned last (counting sort)
static inline void anchor_order(const uint16_t* ids, int n, int n_frag, uint16_t* order, int* counts /* n_frag+1 */) {
    for (int f = 0; f <= n_frag; ++f) counts[f] = 0;
    for (int a = 0; a < n; ++a) counts[ids[a] == svgen::UNASSIGNED ? n_frag : ids[a]] += 1;
    int acc = 0;
    for (int f = 0; f <= n_frag; ++f) { int c = counts[f]; counts[f] = acc; acc += c; }
    for (int a = 0; a < n; ++a) {
        int f = ids[a] == svgen::UNASSIGNED ? n_frag : ids[a];
        order[counts[f]++] = (uint16_t)a;
    }
}

}  // namespace sv

#endif  // SV_LINKAGE_H
