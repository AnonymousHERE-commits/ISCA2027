// Declarations and capacity macros of the SPLICE-V PL kernels (Vitis HLS C++).
// Capacities are compile-time: the defaults are the manifest caps (full frame);
// the "small" capacity set of pl/kernels.json lowers them for the toy-slice bring-up build.
#ifndef SV_KERNELS_H
#define SV_KERNELS_H

#include <stdint.h>
#include "splicev_gen.h"

#ifndef SV_RESERVE_NMAX
#define SV_RESERVE_NMAX 6400
#endif
#ifndef SV_LINK_NMAX
#define SV_LINK_NMAX 1856
#endif
#ifndef SV_LINK_NBCAP
#define SV_LINK_NBCAP (2 * 128 - 2)
#endif
#ifndef SV_LINK_EMAX
#define SV_LINK_EMAX 32768
#endif
#ifndef SV_GATE_NMAX
#define SV_GATE_NMAX 1856
#endif
#ifndef SV_FMAX
#define SV_FMAX 256
#endif

extern "C" {
void sv_reserve(const uint32_t* cells, uint32_t* picks, uint32_t* stats, int n, int width, int count);
void sv_linkage(const uint32_t* edges, const uint32_t* anchors, uint32_t* ids, uint32_t* order, uint32_t* stats,
                int n, int n_edges, int budget, int f_cap);
void sv_gate(const uint32_t* anchors, uint32_t* mask, uint32_t* fused, uint32_t* stats, int n, int n_frag);
void sv_exchange1(const uint32_t* partners, const uint32_t* evidence, const uint32_t* npart, const uint32_t* need,
                  const uint32_t* rho, const uint32_t* phi, uint32_t* sources, uint32_t* lam, uint32_t* status, int n_frag);
void sv_exchange2(const uint32_t* sources, const uint32_t* lam, const uint32_t* tv, const uint32_t* h, uint32_t* dv,
                  uint32_t* stats, int n_frag);
}

#endif
