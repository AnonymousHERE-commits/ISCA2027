"""Host-orchestrated toy vertical slice (gate G07): every arithmetic step runs as an AIE
dispatcher unit or a PL engine call; Python only sequences them and assembles packets.

Backends
  RefBackend    units by units.run_stream (optionally routed onto worker classes), engines
                by pl_kernels_ref; tests/test_slice_runner.py proves the orchestration
                equals toy.run bit for bit
  BoardBackend  units through the AIE worker pool (sv_mm2s/sv_s2mm per worker, routed by
                body mask), engines through the HLS kernels; on fakexrt.FakeDevice the same
                code runs against the emulated device (tests/test_ps_board_paths.py)

Host steps (labelled 'host' in the trace): walk-tap export, P/H1 packet assembly into the
edge list, level extension of the memberships, the synthetic need/rho/phi scaling and the
residual subtraction.  The device-scheduled version (G09) moves them to the array/PL.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from splicev_ref import gen, pl_abi, pl_kernels_ref, routing, toy, units
from splicev_ref import unit_builders as ub
from splicev_ref.gate import unpack_mask
from splicev_ref.linkage import window_edges
from splicev_ref.membership import extend_levels, level_weights
from splicev_ref.walk import export_taps

HOST_STEPS = ["walk tap export", "P/H1 assembly into the edge list", "level extension", "synthetic trust-head scaling",
              "residual subtraction"]


class RefBackend:
    """Integer reference behind the same interface.  With masks, units are routed onto
    worker classes exactly as WorkerPool routes them on the board."""
    name = "reference"

    def __init__(self, masks: Sequence[int] = (routing.FULL_MASK,)):
        self.masks = list(masks)

    def units(self, cmds: Sequence[Sequence[int]]):
        per = routing.group(cmds, self.masks)
        by_cmd: Dict[int, Dict] = {}
        for w, idx in enumerate(per):
            if not idx:
                continue
            stream = [x for i in idx for x in cmds[i]]
            for i, r in zip(idx, units.split_responses(units.run_stream(stream, self.masks[w]))):
                by_cmd[i] = r
        return [by_cmd[i] for i in range(len(cmds))]

    def reserve(self, cells, n, width, count):
        return pl_kernels_ref.sv_reserve(cells, n, width, count)

    def linkage(self, edges, anchors, n, n_edges, budget, f_cap):
        return pl_kernels_ref.sv_linkage(edges, anchors, n, n_edges, budget, f_cap)

    def gate(self, anchors, n, n_frag):
        return pl_kernels_ref.sv_gate(anchors, n, n_frag)

    def exchange1(self, packed, n_frag):
        return pl_kernels_ref.sv_exchange1(packed, n_frag)

    def exchange2(self, packed, n_frag):
        return pl_kernels_ref.sv_exchange2(packed, n_frag)


class BoardBackend:
    """Units through the AIE worker pool (sv_mm2s/sv_s2mm), decisions through the HLS kernels."""
    name = "board"

    def __init__(self, dev, workers: Optional[Sequence[Dict]] = None, timeout_ms: int = 5000):
        from .engines import Engines
        from .workers import WorkerPool, load_workers_json
        self.pool = WorkerPool(dev, workers or load_workers_json(None), timeout_ms)
        self.eng = Engines(dev)
        self.timing: List[Dict] = []

    def units(self, cmds):
        rsps, meta = self.pool.submit(cmds)
        self.timing.append({"step": "units", "count": len(cmds), "t_s": meta["t_submit_s"]})
        return units.split_responses([w for r in rsps for w in r])

    def reserve(self, cells, n, width, count):
        picks, stats, rec = self.eng.reserve(cells, n, width, count)
        self.timing.append({"step": "sv_reserve", "t_s": rec.seconds})
        return picks, stats

    def linkage(self, edges, anchors, n, n_edges, budget, f_cap):
        ids, order, stats, rec = self.eng.linkage(edges, anchors, n, n_edges, budget, f_cap)
        self.timing.append({"step": "sv_linkage", "t_s": rec.seconds})
        return ids, order, stats

    def gate(self, anchors, n, n_frag):
        mask, fused, stats, rec = self.eng.gate(anchors, n, n_frag)
        self.timing.append({"step": "sv_gate", "t_s": rec.seconds})
        return mask, fused, stats

    def exchange1(self, packed, n_frag):
        src, lam, st, rec = self.eng.exchange1(packed, n_frag)
        self.timing.append({"step": "sv_exchange1", "t_s": rec.seconds})
        return src, lam, st

    def exchange2(self, packed, n_frag):
        dv, st, rec = self.eng.exchange2(packed, n_frag)
        self.timing.append({"step": "sv_exchange2", "t_s": rec.seconds})
        return dv, st


def _ok(rsps) -> List[List[int]]:
    for r in rsps:
        if r["hdr"]["status"] != gen.ST_OK:
            raise RuntimeError(f"unit opcode {r['hdr']['opcode']} slot {r['hdr']['slot']} returned status {r['hdr']['status']}")
    return [r["payload"] for r in rsps]


def _rows(words, rows, cols, kind, signed=True):
    return units.unpack_rows(words, rows, cols, kind, signed)


def run_slice(scene: Dict, be, frame_id: int = 1) -> Dict:
    geo = scene["geo"]
    lv = scene["levels"]
    hd = scene["heads"]
    tr: Dict = {"backend": be.name, "host_steps": HOST_STEPS}
    fid = dict(frame_id=frame_id)

    # A1: walks and bands (one STENCIL unit per level, whole lattice, halo 0)
    cmds, shapes = [], []
    for slot, name in enumerate(("P3", "P4", "P5")):
        L = lv[name]
        taps = export_taps(L["w"], L["inside"])                                   # host: tap export
        exps = tuple(gen.BAND_EXP[name]) + (gen.BAND_EXP_LOW,)
        cmds.append(ub.stencil(taps, L["x"], gen.LEVELS[name]["steps"], 0, L["betas"], exps, slot=slot, **fid))
        shapes.append(L["x"].shape)
    outs = _ok(be.units(cmds))
    bands = {}
    for name, (h, w, c), words in zip(("P3", "P4", "P5"), shapes, outs):
        per = units.row_words(h * w, "8")
        arr = np.zeros((5, h, w, c), dtype=np.int64)
        for ch in range(c):
            for j in range(5):
                blk = words[(ch * 5 + j) * per:(ch * 5 + j + 1) * per]
                arr[j, :, :, ch] = units.unpack(blk, h * w, "8", True).reshape(h, w)
        bands[name] = arr
    tr["bands"] = bands

    # PL at H0: reserve
    e3, t3 = lv["P3"]["e"].ravel(), lv["P3"]["t"].ravel()
    picks, _ = be.reserve(pl_abi.reserve_in(e3, t3), e3.size, geo.w3, geo.n_reserve)
    reserve = np.asarray(picks, dtype=np.int64)
    tr["reserve"] = reserve

    # PL A2a: linkage on the P and H1 packets (host assembles the edge list)
    rs, rr_p, rr_s = toy.reserve_window_scores(scene, reserve)
    edges = window_edges(scene["scores_p4"], reserve, rs, rr_p, rr_s, geo.h4, geo.w4, geo.w3)
    n4 = geo.h4 * geo.w4
    e_anchor = np.concatenate([lv["P4"]["e"].ravel(), e3[reserve]])
    active = e_anchor >= gen.RECV_MIN_E
    n_anchor = n4 + reserve.size
    ew, aw = pl_abi.linkage_in(edges, active, e_anchor)
    ids_w, order_w, lstats = be.linkage(ew, aw, n_anchor, len(edges), gen.LINK_BUDGET, gen.F_CAP)
    ids_anchor = np.asarray(ids_w, dtype=np.int64)[:n_anchor]
    order = np.asarray(order_w, dtype=np.int64)[:n_anchor]
    F = int(lstats[0])
    ids3, ids4, ids5 = extend_levels(ids_anchor[:n4], reserve, ids_anchor[n4:], (geo.h3, geo.w3), (geo.h4, geo.w4), (geo.h5, geo.w5))
    tr.update(ids_anchor=ids_anchor, order=order, F=F, link_stats=np.asarray(lstats), ids={"P3": ids3, "P4": ids4, "P5": ids5})

    # PL right after the ids: early gate -> G
    anchor_obj = np.concatenate([lv["P4"]["obj"].ravel(), lv["P3"]["obj"].ravel()[reserve]])
    mu, prec = toy._anchor_hypotheses(scene, anchor_obj)
    mask_w, _fused, _gst = be.gate(pl_abi.gate_in(mu, prec, ids_anchor), n_anchor, F)
    mask_words = np.asarray(mask_w, dtype=np.int64)
    adm = unpack_mask(mask_words, F)
    tr.update(mask_words=mask_words, adm=adm)

    # A2b first part: restriction, descriptor, per-fragment heads
    x_anchor = np.concatenate([lv["P4"]["x"].reshape(n4, -1), lv["P3"]["x"].reshape(-1, geo.channels)[reserve]])
    omega = np.where(ids_anchor != gen.UNASSIGNED, e_anchor, 0)
    seg = _ok(be.units([ub.segsum(ids_anchor, omega, x_anchor, F, slot=10, **fid)]))[0]
    sums = _rows(seg[:F * geo.channels], F, geo.channels, "32")
    ws = units.unpack(seg[F * geo.channels:F * geo.channels + F], F, "32", True)
    H = _rows(_ok(be.units([ub.segmean(sums, ws, slot=11, **fid)]))[0], F, geo.channels, "8")
    cmds = []
    for slot, name in enumerate(("P3", "P4", "P5")):
        z = bands[name][:4, ..., :gen.C_TRANSPORT].reshape(4, -1, gen.C_TRANSPORT)
        cmds.append(ub.energy(tr["ids"][name], z, F, slot=20 + slot, **fid))
    en = _ok(be.units(cmds))
    nu_sums = np.zeros((F, 12), dtype=object)
    nu_cnts = np.zeros((F, 12), dtype=np.int64)
    for li, words in enumerate(en):
        s64 = units.unpack(words[:F * 4 * 2], F * 4, "64", True).reshape(F, 4)
        cnt = units.unpack(words[F * 4 * 2:F * 4 * 2 + F], F, "32", True)
        for f in range(F):
            for j in range(4):
                nu_sums[f, li * 4 + j] = int(s64[f, j])
                nu_cnts[f, li * 4 + j] = int(cnt[f])
    nu = _rows(_ok(be.units([ub.nu(nu_sums, nu_cnts, slot=23, **fid)]))[0], F, 12, "8")
    heads = _ok(be.units([
        ub.gemm(H, hd["W_K"], shift=7, slot=30, **fid),
        ub.gemm(nu, hd["W_U"], shift=6, slot=31, **fid),
        ub.gemm(nu, hd["u0"][:, None], out="32", slot=32, **fid),
        ub.gemm(nu, hd["ua"][:, None], out="32", slot=33, **fid),
        ub.gemm(H, hd["O_F"], shift=7, slot=34, **fid),
        ub.gemm(H, hd["w_need"][:, None], out="32", slot=35, **fid),
        ub.gemm(H, hd["w_rho"][:, None], out="32", slot=36, **fid),
        ub.gemm(H, hd["w_phi"], out="32", slot=37, **fid),
    ]))
    extra = []
    if "W_S" in hd:
        extra.append(("S", ub.gemm(H, hd["W_S"], shift=7, slot=38, **fid), gen.KEY_D if hd["W_S"].shape[1] == gen.KEY_D else hd["W_S"].shape[1]))
    if "O_T" in hd:
        extra.append(("VO_T", ub.gemm(H, hd["O_T"], shift=7, slot=39, **fid), gen.C_TRANSPORT))
    ex_out = _ok(be.units([c for _, c, _ in extra])) if extra and F else []
    K = _rows(heads[0], F, gen.KEY_D, "8")
    Uv = _rows(heads[1], F, gen.FC_UDIM, "8")
    null_raw = _rows(heads[2], F, 1, "32").ravel()
    alpha_raw = _rows(heads[3], F, 1, "32").ravel()
    VO = _rows(heads[4], F, gen.C_TRANSPORT, "8")
    named = {name: _rows(words, F, width, "8") for (name, _, width), words in zip(extra, ex_out)}
    S = named.get("S", K)
    VO_T = named.get("VO_T", VO)
    hn = _rows(heads[5], F, 1, "32").ravel()
    hr = _rows(heads[6], F, 1, "32").ravel()
    hp = _rows(heads[7], F, 4, "32")
    need = np.clip((1 << 23) + hn * 64, 1 << 21, 3 << 22)                          # host: synthetic head scaling
    rho = np.clip((1 << 23) + hr * 64, 1 << 21, 3 << 22)
    phi = np.clip(hp * 16, -(1 << 21), 1 << 21)
    Hs = H[:, :gen.C_TRANSPORT]
    tr.update(H=H, nu=nu, K=K, Uv=Uv, VO=VO)

    # A2b second part: gated forecast reading the same mask G
    mrows = mask_words.reshape(gen.F_CAP, -1)
    fc_words = _ok(be.units([ub.forecast(K, Uv, VO, mrows[:F], Hs, null_raw, alpha_raw, 0, slot=40, **fid)]))[0]
    per = units.row_words(gen.C_TRANSPORT, "8") + 1
    dh_hat = np.stack([units.unpack(fc_words[r * per:r * per + per - 1], gen.C_TRANSPORT, "8", True) for r in range(F)]) if F else np.zeros((0, 48), dtype=np.int64)
    tr["dh_hat"] = dh_hat

    # exchange: evidence (array) -> posterior, trust (PL) -> H3' -> priority unit -> H3'' -> dV
    kk = _rows(_ok(be.units([ub.gemm(S, S.T, shift=10, out="16", slot=50, **fid)]))[0], F, F, "16") if F else np.zeros((0, 0), dtype=np.int64)
    partners = [[int(g) for g in np.flatnonzero(adm[f]) if g != f][:gen.N_CAP] for f in range(F)]
    evid = [[int(kk[f, g]) for g in partners[f]] for f in range(F)]
    from splicev_ref.exchange import pack_phase1_inputs
    src_w, lam_w, st_w = be.exchange1(pack_phase1_inputs(partners, evid, need, rho, phi), F)
    sources = np.asarray(src_w, dtype=np.int64).reshape(gen.F_CAP, gen.MS)[:F]
    lam_slot = pl_abi.lam_from_words(lam_w).reshape(gen.F_CAP, gen.MS)[:F]
    vo_rows = [units.pack_u8(VO_T[g]) for g in range(F)] or [[0] * 12]
    idx = [int(s) for s in sources.ravel()]
    tvw = _ok(be.units([ub.permute(idx, vo_rows, slot=51, **fid)]))[0] if F else []
    tv = np.asarray([units.unpack(tvw[i * 12:(i + 1) * 12], 48, "8", True) for i in range(F * gen.MS)]).reshape(F, gen.MS, 48) if F else np.zeros((0, gen.MS, 48), dtype=np.int64)
    dv_w, _ = be.exchange2(pl_abi.exchange2_in(sources, lam_slot, tv, Hs), F)
    dV = pl_abi.i8_rows_from(dv_w, F)
    dH = dV.copy()
    tr.update(sources=sources, lam_slot=lam_slot, tv=tv, dV=dV, partners=partners, evidence=evid)

    # A3a synthesis with the forecast; A3b class band with the exact residual (at P4)
    x4 = lv["P4"]["x"].reshape(n4, geo.channels)
    xs_new = _rows(_ok(be.units([ub.synth(dh_hat, ids4, x4[:, :gen.C_TRANSPORT], slot=60, **fid)]))[0], n4, gen.C_TRANSPORT, "8")
    n_feat = np.concatenate([xs_new, x4[:, gen.C_TRANSPORT:]], axis=1)
    r = dH - dh_hat                                                                   # host: residual (int16)
    rz = _rows(_ok(be.units([ub.gemm(r, hd["Z"], shift=gen.RZ_SHIFT, a16=True, slot=61, **fid)]))[0], F, geo.channels, "8") if F else np.zeros((0, geo.channels), dtype=np.int64)
    w4 = level_weights(lv["P4"]["e"].ravel(), ids4, F)
    seg = _ok(be.units([ub.segsum(ids4, w4, n_feat, max(F, 1), slot=62, **fid)]))[0]
    sums = _rows(seg[:max(F, 1) * geo.channels], max(F, 1), geo.channels, "32")
    ws = units.unpack(seg[max(F, 1) * geo.channels:max(F, 1) * geo.channels + max(F, 1)], max(F, 1), "32", True)
    PiN = _rows(_ok(be.units([ub.segmean(sums, ws, slot=63, **fid)]))[0], max(F, 1), geo.channels, "8")
    rz_full = rz if F else np.zeros((1, geo.channels), dtype=np.int64)
    out = _ok(be.units([ub.clsband(PiN, rz_full, ids4, slot=64, **fid), ub.detail(PiN, ids4, n_feat, slot=65, **fid)]))
    cls_in = _rows(out[0], n4, geo.channels, "8")
    box_in = _rows(out[1], n4, geo.channels, "16")
    logits = _rows(_ok(be.units([ub.gemm(cls_in, hd["w_cls"][:, None], shift=2, out="16", slot=66, **fid)]))[0], n4, 1, "16").ravel()
    sc = _ok(be.units([ub.score(logits, lv["P4"]["gate"].ravel(), slot=67, **fid)]))[0]
    p = units.unpack(sc, n4, "16", False)
    bits = sc[units.row_words(n4, "16"):]
    cands = np.asarray([i for i in range(n4) if (bits[i >> 5] >> (i & 31)) & 1], dtype=np.int64)
    tr.update(r=r, rz=rz, cls_in=cls_in, box_in=box_in, logits=logits, scores=p, candidates=cands)
    tr["boundaries"] = {
        "ids": np.concatenate([ids3, ids4, ids5, order]).astype(np.int64),
        "mask": mask_words.astype(np.int64),
        "sources": sources.ravel(),
        "dV": dV.ravel(),
        "logits": logits.astype(np.int64),
        "candidates": cands,
    }
    return tr


def compare(got: Dict, want: Dict) -> Dict:
    """P1-style comparison at the six boundaries: sizes, mismatch counts, first mismatch."""
    out = {}
    for k, w in want["boundaries"].items():
        g = got["boundaries"][k]
        w = np.asarray(w, dtype=np.int64).ravel()
        g = np.asarray(g, dtype=np.int64).ravel()
        n = min(g.size, w.size)
        diff = np.flatnonzero(g[:n] != w[:n])
        out[k] = {"expected_size": int(w.size), "actual_size": int(g.size), "mismatches": int(diff.size) + abs(int(g.size) - int(w.size)),
                  "first_mismatch": int(diff[0]) if diff.size else (n if g.size != w.size else None)}
    out["all_zero_mismatch"] = all(v["mismatches"] == 0 for v in out.values() if isinstance(v, dict))
    return out
