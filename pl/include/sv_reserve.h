// P3 reserve anchors: thin-aware farthest-point sampling (DD-09).  Twin of reserve.py.
#ifndef SV_RESERVE_H
#define SV_RESERVE_H

#include "sv_int.h"
#include "splicev_gen.h"

namespace sv {

template <int NMAX>
struct ReserveState {
    uint16_t cand[NMAX];
    uint8_t tc[NMAX];
    int64_t dmin[NMAX];
    bool picked[NMAX];
};

// e, t: n P3 cells (u8 codes); width: P3 lattice width; picks receives up to 'count' cell indices.
// Returns the number of picks.  'scans' accumulates the counted work (candidate scans).
template <int NMAX>
static int reserve_select(const uint8_t* e, const uint8_t* t, int n, int width, int count,
                          uint16_t* picks, ReserveState<NMAX>& st, uint64_t& scans) {
    int nc = 0;
    for (int i = 0; i < n; ++i) {
        if ((int)e[i] >= svgen::RESERVE_MIN_E && nc < NMAX) {
            st.cand[nc] = (uint16_t)i;
            st.tc[nc] = t[i];
            st.dmin[nc] = (int64_t)1 << 30;
            st.picked[nc] = false;
            ++nc;
        }
    }
    int np = 0;
    int limit = count < nc ? count : nc;
    for (int k = 0; k < limit; ++k) {
        int64_t best = -2;
        int bi = 0;
        for (int c = 0; c < nc; ++c) {
            int64_t score = (k == 0) ? (int64_t)st.tc[c] : st.dmin[c] * (256 + (int64_t)st.tc[c]);
            if (st.picked[c]) score = -1;
            if (score > best) { best = score; bi = c; }       // strict: first maximum wins
        }
        picks[np++] = st.cand[bi];
        st.picked[bi] = true;
        int by = st.cand[bi] / width, bx = st.cand[bi] % width;
        for (int c = 0; c < nc; ++c) {
            int y = st.cand[c] / width, x = st.cand[c] % width;
            int64_t d2 = (int64_t)(y - by) * (y - by) + (int64_t)(x - bx) * (x - bx);
            if (d2 < st.dmin[c]) st.dmin[c] = d2;
        }
        scans += (uint64_t)nc;
    }
    return np;
}

}  // namespace sv

#endif  // SV_RESERVE_H
