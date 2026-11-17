#!/usr/bin/env python3
"""G07 (host-orchestrated): the toy vertical slice on the board, P1-style comparison.

toy features -> walk/bands (AIE STENCIL) -> reserve (PL) -> certified linkage (PL) ->
early gate G (PL) -> restriction/descriptor/heads (AIE) -> gated forecast with the same
G (AIE HEAD) -> evidence (AIE GEMM) -> trust exchange (PL) -> priority products (AIE
PERMUTE) -> dV (PL) -> synthesis/residual/class band/score (AIE).  Python sequences the
steps (host-orchestrated mode, labelled so); every arithmetic step runs on the device.
The six boundaries (ids, G, H3' sources, dV, logits, candidates) are compared with the
integer reference (toy.run): sizes, mismatch counts and first mismatch per boundary.

    python3 ps/apps/g07_slice.py --release artifacts/release/slice1/release.json --seeds 0 1 2
    python3 ps/apps/g07_slice.py --backend emulated --masks 0x03,0x44,0xB8
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

import _common
from splicev_ps.slice_runner import HOST_STEPS, BoardBackend, RefBackend, compare, run_slice
from splicev_ref import toy


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()[:16]


def main() -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--masks", help="worker body masks when no release is given, e.g. 0x03,0x44,0xB8")
    args = ap.parse_args()
    s = _common.Session(args, "G07")
    if args.masks and not s.rel:
        s.workers = _common.workers_of({}, [int(m, 16) for m in args.masks.split(",")])
        if args.backend == "emulated":
            from splicev_ps.fakexrt import FakeDevice
            s.dev = FakeDevice([int(w["mask"]) for w in s.workers])
    results = []
    try:
        if args.backend == "reference":
            be = RefBackend([int(w["mask"]) for w in s.workers])
        else:
            be = BoardBackend(s.dev, s.workers, timeout_ms=args.timeout_ms)
        for rep in range(args.repeat):
            for seed in args.seeds:
                want = toy.run(toy.make_scene(seed))
                got = run_slice(toy.make_scene(seed), be, frame_id=1 + rep * 1000 + seed)
                c = compare(got, want)
                results.append({"repeat": rep, "seed": seed, "fragments": int(got["F"]), "candidates": int(len(got["candidates"])),
                                "boundaries": {k: {**v, "sha16": digest(got["boundaries"][k])} for k, v in c.items() if isinstance(v, dict)},
                                "pass": c["all_zero_mismatch"]})
        timing = getattr(be, "timing", [])
    finally:
        s.close()
    ok = all(r["pass"] for r in results)
    summary = {"seeds": args.seeds, "repeats": args.repeat, "frames": len(results), "frames_passing": sum(r["pass"] for r in results),
               "host_steps": HOST_STEPS}
    return s.finish("slice", {"pass": ok, "summary": summary, "results": results, "timing": timing})


if __name__ == "__main__":
    sys.exit(main())
