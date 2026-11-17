"""PL exchange in two phases around the AIE priority unit (SPLICE-V Fig. 3(a), Prop. 4).

phase1 (after H3):  posterior per receiver row -> mutual posterior -> screened trust
                    solve -> source lists (H3') and the trusts of each list slot.
phase2 (after H3''): one-hop innovation dV from the translated values (dev profile;
                    checks, consensus and hop two need the SPLICE model spec).

Word formats (one element per 32-bit word unless noted) are shared with the HLS
kernels sv_exchange1 / sv_exchange2 and the PS host:
  partners  F_cap x N_cap   partner id or 0xFFFF (admissible partners, ascending id)
  evidence  F_cap x N_cap   int16 Q7.8 sign-extended
  npart     F_cap
  need, rho F_cap           Q8.23;   phi F_cap x 4 Q8.23
  sources   F_cap x MS      u16 or 0xFFFF;   lam F_cap x MS x 2 (int64 lo, hi) Q8.39
  status    F_cap           status | iterations << 8 | screened << 16
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from . import bf24 as bf
from . import gen
from . import posterior as post
from . import trust as trust_mod
from .arith import Counters

STATUS_CODE = {"empty": 0, "converged": 1, "capped": 2, "singular": 3}


def phase1(partners: Sequence[Sequence[int]], evidence: Sequence[Sequence[int]], need23, rho23, phi23,
           cnt: Counters | None = None, stratum: int = 0) -> Dict:
    n_frag = len(partners)
    marg = [post.row_marginals(evidence[f], stratum) for f in range(n_frag)]
    mutual = post.mutual_posteriors(partners, marg)
    sources = np.full((n_frag, gen.MS), 0xFFFF, dtype=np.int64)
    lam_slot = np.zeros((n_frag, gen.MS), dtype=np.int64)
    status = np.zeros(n_frag, dtype=np.int64)
    rows: List[trust_mod.TrustRow] = []
    for f in range(n_frag):
        cand = list(partners[f])
        p39 = [min(1 << 39, bf.to_fixed(mutual[f][k], 39)) for k in range(len(cand))]
        rho39 = [int(rho23[g]) << 16 for g in cand]
        phi39 = [[int(v) << 16 for v in phi23[g]] for g in cand]
        row = trust_mod.solve_row(int(need23[f]) << 16, cand, p39, rho39, phi39)
        if cnt is not None:
            cnt.merge(row.counters)
        rows.append(row)
        lst = row.source_list()
        sources[f] = lst
        lam_of = dict(zip(row.sources, row.lam))
        lam_slot[f] = [lam_of.get(s, 0) if s != 0xFFFF else 0 for s in lst]
        status[f] = STATUS_CODE[row.status] | (row.iterations << 8) | (row.screened << 16)
    return {"marginals": marg, "mutual": mutual, "rows": rows, "sources": sources, "lam_slot": lam_slot, "status": status}


def phase2(sources: np.ndarray, lam_slot: np.ndarray, tv: np.ndarray, h: np.ndarray, cnt: Counters | None = None) -> np.ndarray:
    """dv_f,c = sat_sym(rne(sum_slot lam (tv - h) >> 39)) over the slots of the source list."""
    n_frag, width = h.shape
    dv = np.zeros((n_frag, width), dtype=np.int64)
    for f in range(n_frag):
        for c in range(width):
            acc = 0
            for s in range(gen.MS):
                if sources[f, s] != 0xFFFF:
                    acc += int(lam_slot[f, s]) * (int(tv[f, s, c]) - int(h[f, c]))
            v = trust_mod.rshift_rne(acc, trust_mod.Q)
            if v > 127 or v < -127:
                if cnt is not None:
                    cnt.add("sat_dv")
                v = max(-127, min(127, v))
            dv[f, c] = v
    return dv


def pack_phase1_inputs(partners, evidence, need23, rho23, phi23):
    """Word arrays for sv_exchange1 (F rows padded to F_cap, N_cap partners per row)."""
    F, N = gen.F_CAP, gen.N_CAP
    pw = np.full((F, N), 0xFFFF, dtype=np.int64)
    ew = np.zeros((F, N), dtype=np.int64)
    npart = np.zeros(F, dtype=np.int64)
    for f, row in enumerate(partners):
        npart[f] = len(row)
        pw[f, :len(row)] = row
        ew[f, :len(row)] = evidence[f]
    nf = len(partners)
    need = np.zeros(F, dtype=np.int64); need[:nf] = need23
    rho = np.zeros(F, dtype=np.int64); rho[:nf] = rho23
    phi = np.zeros((F, 4), dtype=np.int64); phi[:nf] = phi23
    to_u32 = lambda a: (np.asarray(a, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint32)
    return {"partners": to_u32(pw), "evidence": to_u32(ew), "npart": to_u32(npart), "need": to_u32(need),
            "rho": to_u32(rho), "phi": to_u32(phi)}
