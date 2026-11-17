#!/usr/bin/env python3
"""G06: the PL decision engines (HLS C++) on the board vs the integer reference.

Runs sv_reserve, sv_linkage, sv_gate, sv_exchange1 and sv_exchange2 on every case of
artifacts/references/vectors/engines/pl_kernels.svb (tools/make_vectors.py) with the word
ABI of reference/splicev_ref/pl_abi.py and compares every output word.  Each case is run
--repeat times in one process (repeat/reset part of G06: identical results on reuse).

    python3 ps/apps/g06_engines.py --release artifacts/release/slice1/release.json --repeat 3
    python3 ps/apps/g06_engines.py --backend emulated
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import _common
from splicev_ref import gen, pl_kernels_ref, svb


def cmp(name, got, want):
    g = np.asarray(got, dtype=np.int64).ravel() & 0xFFFFFFFF
    w = np.asarray(want, dtype=np.int64).ravel() & 0xFFFFFFFF
    n = w.size
    if g.size < n:
        return {"output": name, "expected_words": int(n), "got_words": int(g.size), "mismatches": int(n), "first": 0}
    d = np.flatnonzero(g[:n] != w)
    return {"output": name, "expected_words": int(n), "mismatches": int(d.size), "first": int(d[0]) if d.size else None}


def run_case(b, p, eng):
    """One case through the engines (eng: Engines or None for the reference)."""
    res = []
    F = int(b[p + "ex1.args"][0])
    a = b[p + "reserve.args"]
    if eng is None:
        picks, _ = pl_kernels_ref.sv_reserve(b[p + "reserve.cells"], int(a[0]), int(a[1]), int(a[2]))
    else:
        picks, _, _ = eng.reserve(b[p + "reserve.cells"], int(a[0]), int(a[1]), int(a[2]))
    res.append(cmp("reserve.picks", picks, b[p + "reserve.picks"]))
    a = b[p + "linkage.args"]
    n = int(a[0])
    if eng is None:
        ids, order, stats = pl_kernels_ref.sv_linkage(b[p + "linkage.edges"], b[p + "linkage.anchors"], n, int(a[1]), int(a[2]), int(a[3]))
    else:
        ids, order, stats, _ = eng.linkage(b[p + "linkage.edges"], b[p + "linkage.anchors"], n, int(a[1]), int(a[2]), int(a[3]))
    res += [cmp("linkage.ids", ids, b[p + "linkage.ids"]), cmp("linkage.order", order, b[p + "linkage.order"]),
            cmp("linkage.stats", stats, b[p + "linkage.stats"])]
    a = b[p + "gate.args"]
    if eng is None:
        mask, _, gst = pl_kernels_ref.sv_gate(b[p + "gate.anchors"], int(a[0]), int(a[1]))
    else:
        mask, _, gst, _ = eng.gate(b[p + "gate.anchors"], int(a[0]), int(a[1]))
    res += [cmp("gate.mask", mask, b[p + "gate.mask"]), cmp("gate.pairs", np.asarray(gst)[:1], b[p + "gate.pairs"])]
    ex1 = {k: b[p + "ex1." + k] for k in ("partners", "evidence", "npart", "need", "rho", "phi")}
    if eng is None:
        src, lam, st = pl_kernels_ref.sv_exchange1(ex1, F)
    else:
        src, lam, st, _ = eng.exchange1(ex1, F)
    res += [cmp("ex1.sources", np.asarray(src)[:F * gen.MS], b[p + "ex1.sources"]),
            cmp("ex1.lam", np.asarray(lam)[:2 * F * gen.MS], b[p + "ex1.lam"]),
            cmp("ex1.status", np.asarray(st)[:F], b[p + "ex1.status"])]
    ex2 = {k: b[p + "ex2." + k] for k in ("sources", "lam", "tv", "h")}
    if eng is None:
        dv, _ = pl_kernels_ref.sv_exchange2(ex2, F)
    else:
        dv, _, _ = eng.exchange2(ex2, F)
    res.append(cmp("ex2.dv", np.asarray(dv)[:F * gen.C_TRANSPORT // 4], b[p + "ex2.dv"]))
    return res


def main() -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--vectors", default=str(_common.REPO / "artifacts" / "references" / "vectors"))
    args = ap.parse_args()
    s = _common.Session(args, "G06")
    b = svb.read(Path(args.vectors) / "engines" / "pl_kernels.svb")
    results = []
    try:
        eng = None
        if args.backend != "reference":
            from splicev_ps.engines import Engines
            eng = Engines(s.dev, timeout_ms=args.timeout_ms)
        for rep in range(args.repeat):
            for c in range(int(b["n_cases"][0])):
                for r in run_case(b, f"c{c}.", eng):
                    results.append({"repeat": rep, "case": c, **r})
        timing = [{"kernel": r.kernel, "t_s": r.seconds} for r in (eng.records if eng else [])]
    finally:
        s.close()
    ok = all(r["mismatches"] == 0 for r in results)
    summary = {"cases": int(b["n_cases"][0]), "repeats": args.repeat, "outputs_compared": len(results),
               "mismatching_outputs": sum(1 for r in results if r["mismatches"])}
    return s.finish("engines", {"pass": ok, "summary": summary, "results": results, "timing": timing})


if __name__ == "__main__":
    sys.exit(main())
