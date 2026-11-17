"""Toy vertical slice (roadmap §12): synthetic features -> walk/bands -> memberships ->
early mask -> gated forecast -> PL trust -> late residual -> class output.

Everything is synthetic and labelled so: the scene, the head weights and the
calibration constants come from a seeded generator, the exchange is the
dev-profile one-hop innovation, and the neck is the identity.  Its purpose is an
exact cross-engine conformance target for gates G05-G07, not accuracy.

The six P1 boundaries are returned under trace["boundaries"]:
ids (H2 ids + order), mask (G), sources (H3'), dV (H4), logits, candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from . import gen
from . import gate as gate_mod
from . import exchange
from .arith import Counters, div_rne, rshift_rne, rshift_rne_np, sat_np
from .forecast import forecast
from .linkage import anchor_order, certified, select_fragments, window_edges, LINK_FWD
from .membership import UNASSIGNED, class_band, extend_levels, fill, level_weights, restrict, synthesize
from .readout import candidates, class_scores, residual, rz_product
from .reserve import select_reserve
from .walk import NEIGH, export_taps, inside_mask, walk_bands


@dataclass
class ToyGeometry:
    h4: int = 12
    w4: int = 12
    n_reserve: int = 16
    channels: int = 64
    du: int = gen.FC_UDIM

    @property
    def h3(self): return 2 * self.h4

    @property
    def w3(self): return 2 * self.w4

    @property
    def h5(self): return self.h4 // 2

    @property
    def w5(self): return self.w4 // 2


def _objects(rng, geo: ToyGeometry):
    """Rectangles on the P4 grid (x0, y0, x1, y1, parts); an object with parts > 1 is cut by
    occluders into horizontal pieces that the linkage keeps apart (Fig. 2(a))."""
    objs = []
    n = rng.integers(4, 7)
    for _ in range(n):
        w = int(rng.integers(2, 5))
        h = int(rng.integers(3, 7))
        x0 = int(rng.integers(0, geo.w4 - w + 1))
        y0 = int(rng.integers(0, geo.h4 - h + 1))
        parts = int(rng.integers(1, 4)) if h >= 3 else 1
        objs.append((x0, y0, x0 + w, y0 + h, min(parts, h)))
    return objs


def _ownership(objs, h, w, scale):
    """(object id, part id) maps at a level whose cells are 'scale/2' per P4 cell."""
    obj = np.full((h, w), -1, dtype=np.int64)
    part = np.full((h, w), -1, dtype=np.int64)
    pid = 0
    for k, (x0, y0, x1, y1, parts) in enumerate(objs):
        ya, yb = y0 * scale // 2, y1 * scale // 2
        xa, xb = x0 * scale // 2, x1 * scale // 2
        obj[ya:yb, xa:xb] = k
        cuts = np.linspace(ya, yb, parts + 1).round().astype(int)
        for q in range(parts):
            part[cuts[q]:max(cuts[q + 1], cuts[q] + 1), xa:xb] = pid + q
        pid += parts
    return obj, part


def _affinities(rng, own):
    h, w = own.shape
    inside = inside_mask(h, w)
    wc = np.zeros((h, w, 8), dtype=np.int64)
    for d, (dy, dx) in enumerate(NEIGH):
        ys, xs = np.mgrid[0:h, 0:w]
        yy, xx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
        same = (own == own[yy, xx]) & (own >= 0)
        base = np.where(same, 29000, np.where((own < 0) & (own[yy, xx] < 0), 12000, 1500))
        wc[..., d] = np.clip(base + rng.integers(-1200, 1200, size=(h, w)), 0, 1 << 15)
    return np.where(inside, wc, 0), inside


def _features(rng, own, channels):
    h, w = own.shape
    proto = rng.integers(-60, 60, size=(16, channels))
    x = rng.integers(-25, 25, size=(h, w, channels))
    for k in range(own.max() + 1):
        x[own == k] += proto[k % 16]
    return np.clip(x, -128, 127)


def _betas(rng, h, w):
    base = np.array([16000, 32000, 46000, 58000], dtype=np.int64)
    jit = rng.integers(-1500, 1500, size=(h, w, 4))
    b = np.clip(base + jit, 1, 65535 - 4)
    return np.sort(b, axis=-1) + np.arange(4)


def make_scene(seed: int = 0, geo: ToyGeometry | None = None) -> Dict:
    geo = geo or ToyGeometry()
    rng = np.random.default_rng(seed)
    objs = _objects(rng, geo)
    obj4, own4 = _ownership(objs, geo.h4, geo.w4, 2)
    obj3, own3 = _ownership(objs, geo.h3, geo.w3, 4)
    obj5, own5 = _ownership(objs, geo.h5, geo.w5, 1)
    lv = {}
    for name, own, obj in (("P3", own3, obj3), ("P4", own4, obj4), ("P5", own5, obj5)):
        w, inside = _affinities(rng, own)
        lv[name] = {
            "own": own, "obj": obj, "w": w, "inside": inside,
            "x": _features(rng, own, geo.channels),
            "betas": _betas(rng, *own.shape),
            "e": np.where(own >= 0, rng.integers(170, 250, size=own.shape), rng.integers(10, 60, size=own.shape)),
            "t": rng.integers(0, 255, size=own.shape),
            "gate": np.where(own >= 0, rng.integers(12000, 32000, size=own.shape), rng.integers(0, 6000, size=own.shape)),
        }
    # window scores of the P4 lattice (forward half window)
    n4 = geo.h4 * geo.w4
    sc = np.zeros((n4, 12), dtype=np.int64)
    for i in range(n4):
        y, x = divmod(i, geo.w4)
        for k, (dy, dx) in enumerate(LINK_FWD):
            yy, xx = y + dy, x + dx
            if 0 <= yy < geo.h4 and 0 <= xx < geo.w4:
                same = own4[y, x] == own4[yy, xx] and own4[y, x] >= 0
                sc[i, k] = int(np.clip((215 if same else 60) + rng.integers(-35, 35), 0, 255))
    # anchor box hypotheses: object box in P4 units * 16 plus noise; precision codes
    boxes = np.array([[x0, y0, x1, y1] for (x0, y0, x1, y1, _p) in objs], dtype=np.int64) * 16
    heads = {
        "W_K": rng.integers(-40, 40, size=(geo.channels, gen.KEY_D)),
        "W_U": rng.integers(-40, 40, size=(gen.NU_DIM, geo.du)),
        "u0": rng.integers(-30, 30, size=(gen.NU_DIM,)),
        "ua": rng.integers(-30, 30, size=(gen.NU_DIM,)),
        "O_F": rng.integers(-30, 30, size=(geo.channels, gen.C_TRANSPORT)),
        "Z": rng.integers(-30, 30, size=(gen.C_TRANSPORT, geo.channels)),
        "w_cls": rng.integers(-20, 20, size=(geo.channels,)),
        "w_need": rng.integers(-20, 20, size=(geo.channels,)),
        "w_rho": rng.integers(-20, 20, size=(geo.channels,)),
        "w_phi": rng.integers(-20, 20, size=(geo.channels, 4)),
    }
    return {"geo": geo, "objs": objs, "levels": lv, "scores_p4": sc, "boxes": boxes, "heads": heads, "seed": seed, "rng": rng}


def _anchor_hypotheses(scene, anchor_obj: np.ndarray):
    rng = np.random.default_rng(scene["seed"] + 1000)
    n = anchor_obj.size
    mu = np.zeros((n, 4), dtype=np.int64)
    prec = np.zeros((n, 4), dtype=np.int64)
    for a in range(n):
        k = anchor_obj[a]
        if k >= 0:
            mu[a] = scene["boxes"][k] + rng.integers(-6, 7, size=4)      # sigma ~ 3.7 code units
            prec[a] = rng.integers(16, 26, size=4)                        # 256 / sigma^2
        else:
            mu[a] = rng.integers(0, 16 * scene["geo"].w4, size=4)
            prec[a] = rng.integers(1, 3, size=4)
    return mu, prec


def reserve_window_scores(scene: Dict, reserve: np.ndarray):
    """Synthetic H1 reserve scores: 5x5 P4 window of each reserve's parent and reserve pairs."""
    geo: ToyGeometry = scene["geo"]
    own4, own3 = scene["levels"]["P4"]["own"].ravel(), scene["levels"]["P3"]["own"].ravel()
    rng = np.random.default_rng(scene["seed"] + 2000)
    rx, ry = (reserve % geo.w3) >> 1, (reserve // geo.w3) >> 1
    rs = np.zeros((reserve.size, 25), dtype=np.int64)
    for r, cell in enumerate(reserve):
        for k in range(25):
            dy, dx = k // 5 - 2, k % 5 - 2
            yy, xx = ry[r] + dy, rx[r] + dx
            if 0 <= yy < geo.h4 and 0 <= xx < geo.w4:
                same = own3[cell] >= 0 and own3[cell] == own4[yy * geo.w4 + xx]
                rs[r, k] = int(np.clip((210 if same else 55) + rng.integers(-30, 30), 0, 255))
    rr_p = np.full((reserve.size, 8), 0xFFFF, dtype=np.int64)
    rr_s = np.zeros((reserve.size, 8), dtype=np.int64)
    for r in range(reserve.size):
        slot = 0
        for p in range(r + 1, reserve.size):
            if slot < 8 and max(abs(int(rx[p] - rx[r])), abs(int(ry[p] - ry[r]))) <= gen.LINK_WINDOW:
                same = own3[reserve[r]] >= 0 and own3[reserve[r]] == own3[reserve[p]]
                rr_p[r, slot] = p
                rr_s[r, slot] = int(np.clip((205 if same else 50) + rng.integers(-25, 25), 0, 255))
                slot += 1
    return rs, rr_p, rr_s


def trust_heads(H: np.ndarray, hd: Dict):
    """Synthetic Q8.23 heads of order one: need ~ 1 +- 0.5, reliability ~ 1 +- 0.5, moments ~ +-0.25."""
    need = np.clip((1 << 23) + (H @ hd["w_need"]) * 64, 1 << 21, 3 << 22)
    rho = np.clip((1 << 23) + (H @ hd["w_rho"]) * 64, 1 << 21, 3 << 22)
    phi = np.clip((H @ hd["w_phi"]) * 16, -(1 << 21), 1 << 21)
    return need, rho, phi


def run(scene: Dict, budget: int | None = gen.LINK_BUDGET) -> Dict:
    geo: ToyGeometry = scene["geo"]
    lv = scene["levels"]
    hd = scene["heads"]
    cnt = Counters()
    tr: Dict = {"counters": cnt}

    # 1. walks and bands per level (A1)
    bands = {}
    taps = {}
    for name in ("P3", "P4", "P5"):
        L = lv[name]
        taps[name] = export_taps(L["w"], L["inside"])
        exps = gen.BAND_EXP[name]
        bands[name] = walk_bands(L["x"], taps[name], L["betas"], gen.LEVELS[name]["steps"], exps, gen.BAND_EXP_LOW, 0, cnt)
    tr["taps"], tr["bands"] = taps, bands

    # 2. reserve (PL, at H0)
    e3, t3 = lv["P3"]["e"].ravel(), lv["P3"]["t"].ravel()
    reserve = select_reserve(e3, t3, geo.n_reserve, geo.w3, cnt)
    tr["reserve"] = reserve

    # 3. linkage (PL, A2a): window edges from P and H1
    n4 = geo.h4 * geo.w4
    rs, rr_p, rr_s = reserve_window_scores(scene, reserve)
    edges = window_edges(scene["scores_p4"], reserve, rs, rr_p, rr_s, geo.h4, geo.w4, geo.w3, cnt)
    e_anchor = np.concatenate([lv["P4"]["e"].ravel(), e3[reserve]])
    active = e_anchor >= gen.RECV_MIN_E
    n_anchor = n4 + reserve.size
    link = certified(n_anchor, edges, active, budget)
    ids_anchor, n_frag = select_fragments(link, n_anchor, e_anchor, gen.F_CAP, cnt)
    order = anchor_order(ids_anchor)
    tr.update(edges=edges, active=active, e_anchor=e_anchor, link=link, ids_anchor=ids_anchor, F=n_frag, order=order,
              reserve_scores=rs, rr_partner=rr_p, rr_score=rr_s)
    cnt.merge(link.counters)

    # 4. memberships of every level (H2)
    ids3, ids4, ids5 = extend_levels(ids_anchor[:n4], reserve, ids_anchor[n4:], (geo.h3, geo.w3), (geo.h4, geo.w4), (geo.h5, geo.w5))
    tr["ids"] = {"P3": ids3, "P4": ids4, "P5": ids5}

    # 5. restriction at the anchors and per-fragment heads (A2b first part)
    x_anchor = np.concatenate([lv["P4"]["x"].reshape(n4, -1), lv["P3"]["x"].reshape(-1, geo.channels)[reserve]])
    omega = np.where(ids_anchor != UNASSIGNED, e_anchor, 0)
    H = restrict(x_anchor, ids_anchor, omega, n_frag, cnt)
    # descriptor nu from band energies of the transported channels at each level
    nu = np.zeros((n_frag, gen.NU_DIM), dtype=np.int64)
    for li, name in enumerate(("P3", "P4", "P5")):
        ids_l = tr["ids"][name]
        z = bands[name][:4, ..., :gen.C_TRANSPORT].reshape(4, -1, gen.C_TRANSPORT)
        for f in range(n_frag):
            sel = ids_l == f
            cells = int(sel.sum())
            for j in range(4):
                if cells:
                    energy = int((z[j][sel] ** 2).sum())
                    mean = div_rne(energy, cells * gen.C_TRANSPORT)
                    nu[f, li * 4 + j] = min(127, rshift_rne(mean, gen.NU_ENERGY_SHIFT))
    # optional heads of a trained model (absent in the synthetic default, which reuses K and O_F):
    # W_S = the symmetric evidence scorer, O_T = the exchange's translation map
    K = sat_np(rshift_rne_np(H @ hd["W_K"], 7), -128, 127, cnt, "sat_K")
    Uv = sat_np(rshift_rne_np(nu @ hd["W_U"], 6), -128, 127, cnt, "sat_Uv")
    null_raw = nu @ hd["u0"]
    alpha_raw = nu @ hd["ua"]
    VO = sat_np(rshift_rne_np(H @ hd["O_F"], 7), -128, 127, cnt, "sat_VO")
    S = sat_np(rshift_rne_np(H @ hd["W_S"], 7), -128, 127, cnt, "sat_S") if "W_S" in hd else K
    VO_T = sat_np(rshift_rne_np(H @ hd["O_T"], 7), -128, 127, cnt, "sat_VOT") if "O_T" in hd else VO
    Hs = H[:, :gen.C_TRANSPORT]
    tr.update(H=H, nu=nu, K=K, Uv=Uv, null_raw=null_raw, alpha_raw=alpha_raw, VO=VO, S=S, VO_T=VO_T)

    # 6. early gate (PL, right after the ids): fused hypotheses -> mask G
    anchor_obj = np.concatenate([lv["P4"]["obj"].ravel(), lv["P3"]["obj"].ravel()[reserve]])
    mu, prec = _anchor_hypotheses(scene, anchor_obj)
    mu_f, var_f = gate_mod.fuse(mu, prec, ids_anchor, n_frag)
    adm = gate_mod.admissible_mask(mu_f, var_f, n_frag, cnt)
    mask_words = gate_mod.pack_mask(adm)
    tr.update(hyp_mu=mu, hyp_prec=prec, mu_f=mu_f, var_f=var_f, adm=adm, mask_words=mask_words)

    # 7. gated forecast (A2b second part)
    dh_hat, abst = forecast(K, Uv, VO, Hs, null_raw, alpha_raw, adm, None, cnt)
    tr.update(dh_hat=dh_hat, abstain=abst)

    # 8. evidence on admissible pairs (array) -> posterior, trust (PL)
    partners = [[int(g) for g in np.flatnonzero(adm[f]) if g != f][:gen.N_CAP] for f in range(n_frag)]
    evid = [[int(np.clip(rshift_rne(int(np.dot(S[f], S[g])), 10), -32768, 32767)) for g in partners[f]] for f in range(n_frag)]
    need, rho, phi = trust_heads(H, hd)
    ex1 = exchange.phase1(partners, evid, need, rho, phi, cnt)
    sources = ex1["sources"]
    tr.update(partners=partners, evidence=evid, marginals=ex1["marginals"], mutual=ex1["mutual"], need=need, rho=rho,
              phi=phi, trust_rows=ex1["rows"], sources=sources, lam_slot=ex1["lam_slot"], trust_status=ex1["status"])

    # 9. priority unit (array): translated values per (receiver, source slot), toy identity translation
    tv = np.zeros((n_frag, gen.MS, gen.C_TRANSPORT), dtype=np.int64)
    for f in range(n_frag):
        for slot, g in enumerate(sources[f]):
            if g != 0xFFFF:
                tv[f, slot] = VO_T[g]
    # 10. exchange phase 2 (PL): one-hop innovation dV -> dH = dV O_T with toy O_T = I
    dV = exchange.phase2(sources, ex1["lam_slot"], tv, Hs, cnt)
    dH = dV.copy()
    tr.update(tv=tv, dV=dV, dH=dH)

    # 11. synthesis with the forecast (A3a) and the readout with the late exact residual (A3b), at P4
    x4 = lv["P4"]["x"].reshape(n4, geo.channels)
    xs_new = synthesize(x4[:, :gen.C_TRANSPORT], ids4, dh_hat, None, cnt)
    n_feat = np.concatenate([xs_new, x4[:, gen.C_TRANSPORT:]], axis=1)
    r = residual(dH, dh_hat, cnt)
    rz = rz_product(r, hd["Z"], cnt)
    w4 = level_weights(lv["P4"]["e"].ravel(), ids4, n_frag)
    cls_in = class_band(n_feat, ids4, w4, n_frag, rz, cnt)
    box_in = n_feat - fill(restrict(n_feat, ids4, w4, n_frag), ids4)
    logits = np.clip(rshift_rne_np(cls_in @ hd["w_cls"], 2), -32768, 32767)        # Q8.8 codes
    p = class_scores(logits, lv["P4"]["gate"].ravel())
    cands = candidates(p)
    tr.update(r=r, rz=rz, cls_in=cls_in, box_in=box_in, logits=logits, scores=p, candidates=cands)

    tr["boundaries"] = {
        "ids": np.concatenate([ids3, ids4, ids5, order]).astype(np.int64),
        "mask": mask_words.astype(np.int64),
        "sources": sources.ravel(),
        "dV": dV.ravel(),
        "logits": logits.astype(np.int64),
        "candidates": cands.astype(np.int64),
    }
    return tr
