#!/usr/bin/env python3
"""Generate bit-exact test vectors from the integer reference for the C++ engines.

Outputs (deterministic for a given manifest and seed):
  <out>/units/<suite>.cmd.bin   uint32 LE command stream (descriptors + payloads)
  <out>/units/<suite>.rsp.bin   expected uint32 LE response stream
  <out>/engines/<engine>.svb    tensor bundles: inputs and expected outputs per case

Tensor bundle (.svb): u32 magic 0x31425653 'SVB1', u32 count, then per tensor:
u32 name length, name bytes padded to 4, u32 ndim, u32 dims[ndim], i64 data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference"))

from splicev_ref import bf24 as bf  # noqa: E402
from splicev_ref import svb  # noqa: E402
from splicev_ref import exchange, gate, gen, linkage as lk, pl_abi, posterior, reserve, toy, trust, units, walk  # noqa: E402
from splicev_ref import unit_builders as ub  # noqa: E402

MAGIC = svb.MAGIC
Bundle = svb.Bundle


def write_words(path: Path, words) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray([int(w) & 0xFFFFFFFF for w in words], dtype="<u4").tofile(path)


# ------------------------------------------------------------------ units

def unit_suite_basic(rng) -> list:
    cmds = []
    A = rng.integers(-128, 128, size=(6, 23)); B = rng.integers(-128, 128, size=(23, 13))
    cmds.append(ub.gemm(A, B, bias=rng.integers(-9000, 9000, size=13), shift=8, frame_id=1, slot=1))
    cmds.append(ub.gemm(A, B, shifts=rng.integers(4, 12, size=13), sym=True, frame_id=1, slot=2))
    cmds.append(ub.gemm(A, B, out="32", frame_id=1, slot=3))
    A16 = rng.integers(-191, 192, size=(5, 48)); B2 = rng.integers(-128, 128, size=(48, 9))
    cmds.append(ub.gemm(A16, B2, shift=7, out="16", frame_id=1, slot=4))
    X = rng.integers(-128, 128, size=(7, 9, 5)); Wt = rng.integers(-128, 128, size=(6, 3, 3, 5))
    cmds.append(ub.conv(X, Wt, 3, 1, bias=rng.integers(-500, 500, size=6), shift=9, frame_id=2))
    cmds.append(ub.conv(X, Wt[:, :1, :1, :], 1, 2, shift=6, frame_id=2, slot=1))
    aff = rng.integers(0, 1 << 15, size=(7, 9, 8)); gam = rng.integers(0, 1 << 14, size=5)
    cmds.append(ub.conv(X, Wt, 3, 1, shift=9, gamma=gam, aff=aff, sym=True, frame_id=2, slot=2))
    H, Wd, C, s = 16, 17, 2, 4
    inside = walk.inside_mask(H, Wd)
    taps = walk.export_taps(np.where(inside, rng.integers(0, 1 << 15, size=(H, Wd, 8)), 0), inside)
    betas = np.sort(rng.integers(2000, 63000, size=(H, Wd, 4)), axis=-1) + np.arange(4)
    Xs = rng.integers(-128, 128, size=(H, Wd, C))
    cmds.append(ub.stencil(taps, Xs, s, s, betas, (1, 2, 2, 3, 0), frame_id=3))
    cmds.append(ub.stencil(taps, Xs, 3, 3, None, frame_id=3, slot=1))
    cmds.append(ub.lut_table8(rng.integers(-128, 128, size=256), rng.integers(-128, 128, size=37), frame_id=4))
    cmds.append(ub.lut_sigmoid(rng.integers(-32768, 32768, size=21), frame_id=4, slot=1))
    cmds.append(ub.lut_expf(rng.integers(-300, 300, size=19), frame_id=4, slot=2))
    cmds.append(ub.requant32(rng.integers(-(1 << 20), 1 << 20, size=(3, 15)), 9, sym=True, frame_id=4, slot=3))
    cmds.append(ub.requant32(rng.integers(-(1 << 24), 1 << 24, size=(2, 9)), 6, out16=True, frame_id=4, slot=4))
    rows = [[int(x) for x in rng.integers(0, 1 << 32, size=3)] for _ in range(6)]
    cmds.append(ub.permute([5, 0, 0xFFFF, 3, 9, 2], rows, frame_id=5))
    n, F, C = 40, 7, 10
    ids = rng.integers(0, F, size=n); ids[::5] = gen.UNASSIGNED
    w = rng.integers(1, 256, size=n); Xc = rng.integers(-128, 128, size=(n, C))
    cmds.append(ub.segsum(ids, w, Xc, F, frame_id=6))
    sums = rng.integers(-200000, 200000, size=(F, C)); ws = rng.integers(0, 3000, size=F); ws[2] = 0
    cmds.append(ub.segmean(sums, ws, frame_id=6, slot=1))
    Hf = rng.integers(-128, 128, size=(F, C))
    cmds.append(ub.fill(Hf, ids, frame_id=6, slot=2))
    cmds.append(ub.detail(Hf, ids, Xc, frame_id=6, slot=3))
    cmds.append(ub.synth(rng.integers(-64, 65, size=(F, C)), ids, Xc, rng.integers(-20, 20, size=(n, C)), frame_id=6, slot=4))
    cmds.append(ub.clsband(Hf, rng.integers(-128, 128, size=(F, C)), ids, frame_id=6, slot=5))
    cmds.append(ub.modulate(rng.integers(0, 1 << 14, size=C), rng.integers(0, 1 << 15, size=n), Xc, frame_id=7))
    cmds.append(ub.energy(ids, rng.integers(-128, 128, size=(4, n, C)), F, frame_id=7, slot=1))
    big = rng.integers(0, 1 << 40, size=(F, 12)); cnt = rng.integers(0, 5000, size=(F, 12)); cnt[0, 0] = 0
    cmds.append(ub.nu(big, cnt, frame_id=7, slot=2))
    cmds.append(ub.mix(rng.integers(-300, 300, size=4), rng.integers(-128, 128, size=(n, 4, C)), 8, frame_id=7, slot=3))
    nf = 11
    K = rng.integers(-60, 60, size=(nf, 64)); Uv = rng.integers(-60, 60, size=(nf, gen.FC_UDIM))
    VO = rng.integers(-128, 128, size=(nf, 48)); Hs = rng.integers(-128, 128, size=(nf, 48))
    adm = np.triu(rng.random((nf, nf)) < 0.5, 1); adm = adm | adm.T
    mw = gate.pack_mask(adm).reshape(gen.F_CAP, -1)
    nr = rng.integers(-3000, 3000, size=nf); ar = rng.integers(-3000, 3000, size=nf)
    cmds.append(ub.forecast(K, Uv, VO, mw[3:9], Hs[3:9], nr[3:9], ar[3:9], 3, frame_id=8))
    cmds.append(ub.score(rng.integers(-32768, 32768, size=45), rng.integers(0, 1 << 15, size=45), frame_id=8, slot=1))
    cmds.append(ub.nop(frame_id=9))
    # error cases: scratch overflow, bad shape (wrong out_words), bad check
    cmds.append(ub.gemm(np.zeros((1, 200)), np.zeros((200, 256)), frame_id=10))
    bad = ub.lut_sigmoid([1, 2, 3], frame_id=10, slot=1)
    bad[gen.DESC_FIELDS["out_words"][0]] += 1
    from splicev_ref.arith import check_word
    bad[gen.DESC_WORDS - 1] = check_word(bad[:gen.DESC_WORDS - 1])
    cmds.append(bad)
    return cmds


def unit_suite_toy(seed: int) -> list:
    sc = toy.make_scene(seed)
    tr = toy.run(sc)
    L = sc["levels"]["P4"]
    cmds = []
    taps = tr["taps"]["P4"]
    s = gen.LEVELS["P4"]["steps"]
    X = L["x"][..., :4]
    # whole P4 lattice as one tile: border taps are zero, so halo 0 is exact
    cmds.append(ub.stencil(taps, X, s, 0, L["betas"], tuple(gen.BAND_EXP["P4"]) + (gen.BAND_EXP_LOW,), frame_id=100 + seed))
    nf = tr["F"]
    ids = tr["ids_anchor"]
    x_anchor = np.concatenate([L["x"].reshape(-1, 64), sc["levels"]["P3"]["x"].reshape(-1, 64)[tr["reserve"]]])
    omega = np.where(ids != gen.UNASSIGNED, tr["e_anchor"], 0)
    cmds.append(ub.segsum(ids, omega, x_anchor[:, :32], nf, frame_id=100 + seed, slot=1))
    mw = tr["mask_words"].reshape(gen.F_CAP, -1)
    cmds.append(ub.forecast(tr["K"], tr["Uv"], tr["VO"], mw[:nf], tr["H"][:, :48], tr["null_raw"], tr["alpha_raw"], 0,
                            frame_id=100 + seed, slot=2))
    cmds.append(ub.score(tr["logits"], L["gate"].ravel(), frame_id=100 + seed, slot=3))
    return cmds


# ------------------------------------------------------------------ engines

def engine_reserve(rng, out: Path) -> None:
    b = Bundle()
    cases = []
    for seed in range(3):
        sc = toy.make_scene(seed)
        L = sc["levels"]["P3"]
        cases.append((L["e"].ravel(), L["t"].ravel(), sc["geo"].w3, sc["geo"].n_reserve))
    e = np.where(rng.random(6400) < 0.4, rng.integers(77, 256, 6400), rng.integers(0, 77, 6400))
    cases.append((e, rng.integers(0, 256, 6400), 80, 256))
    b.add("n_cases", [len(cases)])
    for i, (e, t, width, count) in enumerate(cases):
        picks = reserve.select_reserve(e, t, count, width)
        b.add(f"c{i}.e", e); b.add(f"c{i}.t", t); b.add(f"c{i}.width", [width]); b.add(f"c{i}.count", [count])
        b.add(f"c{i}.picks", picks)
    b.write(out / "reserve.svb")


def _edges_array(edges):
    return np.asarray(edges, dtype=np.int64).reshape(-1, 3)


def engine_linkage(rng, out: Path) -> None:
    b = Bundle()
    cases = []
    for seed in range(3):
        tr = toy.run(toy.make_scene(seed))
        cases.append((len(tr["active"]), tr["edges"], tr["active"], tr["e_anchor"], gen.LINK_BUDGET))
    # full-size frame: 40x40 P4 + 256 reserves with clustered scores
    h4 = w4 = 40
    own = (np.arange(h4)[:, None] // 5) * 8 + (np.arange(w4)[None, :] // 5)
    sc = np.zeros((h4 * w4, 12), dtype=np.int64)
    for i in range(h4 * w4):
        y, x = divmod(i, w4)
        for k, (dy, dx) in enumerate(lk.LINK_FWD):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h4 and 0 <= xx < w4:
                sc[i, k] = int(np.clip((205 if own[y, x] == own[yy, xx] else 70) + rng.integers(-40, 40), 0, 255))
    cells = reserve.select_reserve(rng.integers(60, 256, 6400), rng.integers(0, 256, 6400))
    rsc = rng.integers(40, 240, size=(cells.size, 25))
    rrp = np.full((cells.size, 8), 0xFFFF); rrs = np.zeros((cells.size, 8))
    edges = lk.window_edges(sc, cells, rsc, rrp, rrs)
    n = h4 * w4 + cells.size
    elig = rng.integers(60, 256, n)
    active = elig >= gen.RECV_MIN_E
    cases.append((n, edges, active, elig, gen.LINK_BUDGET))
    cases.append((n, edges, active, elig, 2000))           # forced budget exit
    b.add("n_cases", [len(cases)])
    for i, (n, edges, active, elig, budget) in enumerate(cases):
        res = lk.certified(n, edges, active, budget)
        ids, F = lk.select_fragments(res, n, elig)
        b.add(f"c{i}.n", [n]); b.add(f"c{i}.edges", _edges_array(edges)); b.add(f"c{i}.active", np.asarray(active, dtype=np.int64))
        b.add(f"c{i}.elig", elig); b.add(f"c{i}.budget", [budget])
        b.add(f"c{i}.ids", ids); b.add(f"c{i}.F", [F]); b.add(f"c{i}.rounds", [res.rounds]); b.add(f"c{i}.work", [res.work])
        b.add(f"c{i}.budget_hit", [int(res.budget_hit)]); b.add(f"c{i}.overflow", [int(res.overflow)])
        b.add(f"c{i}.order", lk.anchor_order(ids))
    b.write(out / "linkage.svb")


def engine_gate(rng, out: Path) -> None:
    b = Bundle()
    cases = []
    for seed in range(3):
        tr = toy.run(toy.make_scene(seed))
        cases.append((tr["hyp_mu"], tr["hyp_prec"], tr["ids_anchor"], tr["F"]))
    n, F = 1856, 120
    ids = rng.integers(0, F, size=n); ids[rng.random(n) < 0.05] = gen.UNASSIGNED
    centers = rng.integers(0, 600, size=(F // 3, 4))
    mu = centers[ids % (F // 3)] + rng.integers(-6, 7, size=(n, 4))
    cases.append((mu, rng.integers(10, 30, size=(n, 4)), ids, F))
    b.add("n_cases", [len(cases)])
    for i, (mu, prec, ids, F) in enumerate(cases):
        mu_f, var_f = gate.fuse(mu, prec, ids, F)
        adm = gate.admissible_mask(mu_f, var_f, F)
        b.add(f"c{i}.mu", mu); b.add(f"c{i}.prec", prec); b.add(f"c{i}.ids", ids); b.add(f"c{i}.F", [F])
        b.add(f"c{i}.mu_f", mu_f)
        b.add(f"c{i}.var_m", np.asarray([[v[0] for v in row] for row in var_f]).reshape(F, 4))
        b.add(f"c{i}.var_e", np.asarray([[v[1] for v in row] for row in var_f]).reshape(F, 4))
        b.add(f"c{i}.mask", gate.pack_mask(adm).astype(np.int64))
    b.write(out / "gate.svb")


def engine_posterior_trust(rng, out: Path) -> None:
    pb, tb = Bundle(), Bundle()
    prow = []
    for _ in range(60):
        nn = int(rng.integers(0, 12))
        prow.append([int(x) for x in rng.integers(-1500, 1500, size=nn)])
    prow.append([int(x) for x in rng.integers(-800, 800, size=128)])
    for seed in range(3):
        prow += toy.run(toy.make_scene(seed))["evidence"]
    pb.add("n_rows", [len(prow)])
    for i, ev in enumerate(prow):
        p = posterior.row_marginals(ev)
        pb.add(f"r{i}.evidence", ev if ev else np.zeros(0, dtype=np.int64))
        pb.add(f"r{i}.m", [x[0] for x in p] if p else np.zeros(0, dtype=np.int64))
        pb.add(f"r{i}.e", [x[1] for x in p] if p else np.zeros(0, dtype=np.int64))
    pb.write(out / "posterior.svb")

    O = 1 << 39
    rows = []
    for _ in range(150):
        nn = int(rng.integers(0, 11))
        p = [int(x) for x in rng.integers(0, O + 1, size=nn)]
        rho = [int(x) for x in rng.integers(O // 8, 4 * O, size=nn)]
        phi = [[int(x) for x in rng.integers(-O // 2, O // 2, size=4)] for _ in range(nn)]
        rows.append((int(rng.integers(0, 2 * O)), list(range(100, 100 + nn)), p, rho, phi, gen.KQP))
    for k in (1, 2, 3):
        nn = 8
        rows.append((O, list(range(nn)), [O // 2] * nn, [O] * nn, [[O // 4, -O // 8, O // 16, 0]] * nn, k))
    huge = [[O * 3, -O * 3, O * 3, O * 2]] * 4
    rows.append((O, [1, 2, 3, 4], [O // 3] * 4, [O] * 4, huge, gen.KQP))
    tb.add("n_rows", [len(rows)])
    for i, (need, cand, p, rho, phi, kqp) in enumerate(rows):
        r = trust.solve_row(need, cand, p, rho, phi, kqp=kqp)
        tb.add(f"r{i}.need", [need]); tb.add(f"r{i}.cand", cand if cand else np.zeros(0, dtype=np.int64))
        tb.add(f"r{i}.p", p if p else np.zeros(0, dtype=np.int64)); tb.add(f"r{i}.rho", rho if rho else np.zeros(0, dtype=np.int64))
        tb.add(f"r{i}.phi", np.asarray(phi, dtype=np.int64).reshape(-1, 4)); tb.add(f"r{i}.kqp", [kqp])
        status = {"empty": 0, "converged": 1, "capped": 2, "singular": 3}[r.status]
        tb.add(f"r{i}.status", [status]); tb.add(f"r{i}.iterations", [r.iterations]); tb.add(f"r{i}.screened", [r.screened])
        tb.add(f"r{i}.sources", r.sources if r.sources else np.zeros(0, dtype=np.int64))
        tb.add(f"r{i}.lam", r.lam if r.lam else np.zeros(0, dtype=np.int64))
        tb.add(f"r{i}.source_list", r.source_list())
    tb.write(out / "trust.svb")


def pl_kernels(out: Path, seeds=(0, 1, 2)) -> None:
    """Packed word vectors for the HLS kernel tops (C-sim testbench and board conformance)."""
    b = Bundle()
    b.add("n_cases", [len(seeds)])
    for i, seed in enumerate(seeds):
        sc = toy.make_scene(seed)
        tr = toy.run(sc)
        geo = sc["geo"]
        p = f"c{i}."
        L3 = sc["levels"]["P3"]
        b.add(p + "reserve.cells", pl_abi.reserve_in(L3["e"].ravel(), L3["t"].ravel()))
        b.add(p + "reserve.args", [geo.h3 * geo.w3, geo.w3, geo.n_reserve])
        b.add(p + "reserve.picks", tr["reserve"])
        ew, aw = pl_abi.linkage_in(tr["edges"], tr["active"], tr["e_anchor"])
        b.add(p + "linkage.edges", ew)
        b.add(p + "linkage.anchors", aw)
        n = len(tr["active"])
        b.add(p + "linkage.args", [n, len(tr["edges"]), gen.LINK_BUDGET, gen.F_CAP])
        b.add(p + "linkage.ids", tr["ids_anchor"])
        b.add(p + "linkage.order", tr["order"])
        lr = tr["link"]
        b.add(p + "linkage.stats", [tr["F"], lr.rounds, lr.work & 0xFFFFFFFF, lr.work >> 32, int(lr.budget_hit), int(lr.overflow), 0,
                                    int(tr["counters"].get("fragment_cap_hit") > 0)])
        b.add(p + "gate.anchors", pl_abi.gate_in(tr["hyp_mu"], tr["hyp_prec"], tr["ids_anchor"]))
        b.add(p + "gate.args", [n, tr["F"]])
        b.add(p + "gate.mask", tr["mask_words"])
        b.add(p + "gate.pairs", [int(tr["adm"].sum()) // 2])
        ex = exchange.pack_phase1_inputs(tr["partners"], tr["evidence"], tr["need"], tr["rho"], tr["phi"])
        for k, v in ex.items():
            b.add(p + "ex1." + k, v)
        b.add(p + "ex1.args", [tr["F"]])
        b.add(p + "ex1.sources", pl_abi.pad_rows(tr["sources"], gen.F_CAP, 0).ravel()[: tr["F"] * gen.MS])
        b.add(p + "ex1.lam", pl_abi.lam_words(tr["lam_slot"]))
        b.add(p + "ex1.status", tr["trust_status"])
        ex2 = pl_abi.exchange2_in(tr["sources"], tr["lam_slot"], tr["tv"], tr["H"][:, :gen.C_TRANSPORT])
        for k, v in ex2.items():
            b.add(p + "ex2." + k, v)
        b.add(p + "ex2.dv", pl_abi.i8_rows(tr["dV"], tr["F"]).ravel())
    b.write(out / "pl_kernels.svb")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "references" / "vectors"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    suites = {"basic": unit_suite_basic(rng)}
    for seed in range(2):
        suites[f"toy{seed}"] = unit_suite_toy(seed)
    manifest = {"manifest_sha256": gen.MANIFEST_SHA256, "seed": args.seed, "suites": {}}
    for name, cmds in suites.items():
        stream = [w for c in cmds for w in c]
        rsp = units.run_stream(stream)
        write_words(out / "units" / f"{name}.cmd.bin", stream)
        write_words(out / "units" / f"{name}.rsp.bin", rsp)
        manifest["suites"][name] = {"commands": len(cmds), "cmd_words": len(stream), "rsp_words": len(rsp)}
    engine_reserve(rng, out / "engines")
    engine_linkage(rng, out / "engines")
    engine_gate(rng, out / "engines")
    engine_posterior_trust(rng, out / "engines")
    pl_kernels(out / "engines")
    (out / "vectors.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
