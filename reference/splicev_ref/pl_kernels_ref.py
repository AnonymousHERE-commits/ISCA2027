"""Reference semantics of the PL kernel tops at the word ABI (pl_abi.py).

Each function takes and returns the same 32-bit word arrays as the HLS kernel of the
same name, so a board run, an HLS C-simulation and the reference compare bytes.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from . import exchange, gate as gate_mod, gen, linkage as lk, pl_abi, reserve as res_mod
from .arith import Counters


def sv_reserve(cells: np.ndarray, n: int, width: int, count: int) -> Tuple[np.ndarray, np.ndarray]:
    w = np.asarray(cells, dtype=np.int64)[:n]
    picks = res_mod.select_reserve(w & 0xFF, (w >> 8) & 0xFF, min(count, gen.N_RESERVE), width)
    cand = int(np.count_nonzero((w & 0xFF) >= gen.RESERVE_MIN_E))
    scans = cand * len(picks)
    return pl_abi.u32(picks), pl_abi.u32([len(picks), scans & 0xFFFFFFFF, scans >> 32, 0])


def sv_linkage(edges: np.ndarray, anchors: np.ndarray, n: int, n_edges: int, budget: int, f_cap: int):
    e = np.asarray(edges, dtype=np.int64)[: 3 * n_edges].reshape(-1, 3)
    a = np.asarray(anchors, dtype=np.int64)[:n]
    active = (a & 0xFF) != 0
    elig = (a >> 8) & 0xFF
    res = lk.certified(n, [tuple(int(x) for x in row) for row in e], active.tolist(), None if budget < 0 else budget)
    cnt = Counters()
    ids, F = lk.select_fragments(res, n, elig, min(f_cap, gen.F_CAP), cnt)
    order = lk.anchor_order(ids)
    stats = [F, res.rounds, res.work & 0xFFFFFFFF, res.work >> 32, int(res.budget_hit), int(res.overflow), 0,
             int(cnt.get("fragment_cap_hit") > 0)]
    return pl_abi.u32(ids), pl_abi.u32(order), pl_abi.u32(stats)


def sv_gate(anchors: np.ndarray, n: int, n_frag: int):
    w = np.asarray(anchors, dtype=np.int64)[: 5 * n].reshape(n, 5)
    def s16(x):
        x = x & 0xFFFF
        return np.where(x >= 32768, x - 65536, x)
    mu = np.stack([s16(w[:, 0]), s16(w[:, 0] >> 16), s16(w[:, 1]), s16(w[:, 1] >> 16)], axis=1)
    prec = np.stack([w[:, 2] & 0xFFFF, (w[:, 2] >> 16) & 0xFFFF, w[:, 3] & 0xFFFF, (w[:, 3] >> 16) & 0xFFFF], axis=1)
    ids = w[:, 4] & 0xFFFF
    mu_f, var_f = gate_mod.fuse(mu, prec, ids, n_frag)
    adm = gate_mod.admissible_mask(mu_f, var_f, n_frag)
    fused = np.zeros((gen.F_CAP, 12), dtype=np.int64)
    for f in range(n_frag):
        fused[f, 0:4] = mu_f[f]
        fused[f, 4:8] = [v[0] for v in var_f[f]]
        fused[f, 8:12] = [v[1] for v in var_f[f]]
    return pl_abi.u32(gate_mod.pack_mask(adm)), pl_abi.u32(fused.ravel()), pl_abi.u32([int(adm.sum()) // 2, n_frag])


def unpack_exchange1(packed: Dict[str, np.ndarray], n_frag: int):
    N = gen.N_CAP
    pw = np.asarray(packed["partners"], dtype=np.int64).reshape(gen.F_CAP, N)
    ew = pl_abi.s32(packed["evidence"]).reshape(gen.F_CAP, N)
    npart = np.asarray(packed["npart"], dtype=np.int64)
    partners = [[int(x) for x in pw[f, :min(int(npart[f]), N)]] for f in range(n_frag)]
    evidence = [[int(((x & 0xFFFF) ^ 0x8000) - 0x8000) for x in ew[f, :min(int(npart[f]), N)]] for f in range(n_frag)]
    need = pl_abi.s32(packed["need"])[:n_frag]
    rho = pl_abi.s32(packed["rho"])[:n_frag]
    phi = pl_abi.s32(packed["phi"]).reshape(gen.F_CAP, 4)[:n_frag]
    return partners, evidence, need, rho, phi


def sv_exchange1(packed: Dict[str, np.ndarray], n_frag: int):
    partners, evidence, need, rho, phi = unpack_exchange1(packed, n_frag)
    ex = exchange.phase1(partners, evidence, need, rho, phi)
    src = pl_abi.pad_rows(ex["sources"], gen.F_CAP, 0)
    lam = pl_abi.pad_rows(ex["lam_slot"], gen.F_CAP, 0)
    st = pl_abi.pad_rows(ex["status"][:, None], gen.F_CAP, 0).ravel()
    return pl_abi.u32(src.ravel()), pl_abi.lam_words(lam), pl_abi.u32(st)


def sv_exchange2(packed: Dict[str, np.ndarray], n_frag: int):
    src = np.asarray(packed["sources"], dtype=np.int64).reshape(gen.F_CAP, gen.MS)[:n_frag]
    lam = pl_abi.lam_from_words(packed["lam"]).reshape(gen.F_CAP, gen.MS)[:n_frag]
    tvw = np.asarray(packed["tv"], dtype=np.int64).reshape(gen.F_CAP * gen.MS, gen.C_TRANSPORT // 4)
    tv = pl_abi.i8_rows_from(tvw.ravel(), gen.F_CAP * gen.MS).reshape(gen.F_CAP, gen.MS, gen.C_TRANSPORT)[:n_frag]
    h = pl_abi.i8_rows_from(packed["h"], n_frag)
    cnt = Counters()
    dv = exchange.phase2(src, lam, tv, h, cnt)
    return pl_abi.i8_rows(dv, gen.F_CAP).ravel(), pl_abi.u32([cnt.get("sat_dv"), 0])
