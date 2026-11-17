#!/usr/bin/env python3
"""G02: AIE-ML smoke kernel between the PL movers, driven from Python through XRT.

Streams frames of several lengths through mm2s_0 -> smoke kernel -> s2mm_0 (sink armed
first), checks every response word, repeats the sequence --repeat times within one
process and records the bounded wait times.  Repeat the run after a reboot for the
reboot-repeatability part of G02 (roadmap 3.3).

    python3 ps/apps/g02_smoke.py --release artifacts/release/smoke/release.json --repeat 5
    python3 ps/apps/g02_smoke.py --backend emulated
"""
from __future__ import annotations

import sys

import numpy as np

import _common
from splicev_ps.fakexrt import smoke_response

SIZES = (2, 6, 64, 1000, 4096)


def frames(seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for k, n in enumerate(SIZES):
        data = [int(x) for x in rng.integers(0, 1 << 32, size=n, dtype=np.uint64)]
        add = [1, 0x7FFFFFFF, 0xFFFFFFFF, 12345, 0][k % 5]
        out.append([0x534D4B30, n, add, 0] + data)
    return out


def main() -> int:
    args = _common.parser(__doc__).parse_args()
    s = _common.Session(args, "G02", smoke=True)
    results = []
    try:
        for rep in range(args.repeat):
            fr = frames(rep)
            want = smoke_response([w for f in fr for w in f])
            if args.backend == "reference":
                got, t = want, 0.0
            else:
                from splicev_ps.workers import Worker
                w = Worker(s.dev, 0, 0xFF, timeout_ms=args.timeout_ms)
                got, meta = w.submit(fr, n_out=len(want))
                t = meta["t_submit_s"]
            diff = [i for i in range(len(want)) if int(got[i]) != int(want[i])]
            results.append({"repeat": rep, "frames": len(fr), "words_in": sum(len(f) for f in fr), "words_out": len(want),
                            "mismatches": len(diff), "first_mismatch": diff[0] if diff else None, "t_submit_s": t})
    finally:
        s.close()
    ok = all(r["mismatches"] == 0 for r in results)
    summary = {"repeats": len(results), "frame_sizes": list(SIZES), "total_mismatches": sum(r["mismatches"] for r in results),
               "max_t_submit_s": max(r["t_submit_s"] for r in results)}
    return s.finish("smoke", {"pass": ok, "summary": summary, "results": results})


if __name__ == "__main__":
    sys.exit(main())
