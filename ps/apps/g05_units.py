#!/usr/bin/env python3
"""G05 (host-orchestrated): dispatcher units on the AIE worker pool vs the integer reference.

Replays artifacts/references/vectors/units/<suite>.cmd.bin (tools/make_vectors.py)
through the workers of the release (units routed to a worker class that implements the
body) and compares every response word with <suite>.rsp.bin.  Error-case units
(scratch overflow, bad shape) are part of the 'basic' suite and must answer the
reference status with the stream still aligned.

    python3 ps/apps/g05_units.py --release artifacts/release/slice1/release.json --suite basic toy0 toy1
    python3 ps/apps/g05_units.py --backend emulated --masks 0x03,0x44,0xB8
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import _common
from splicev_ref import gen, units


def per_command(rsp_words, n_cmds):
    out, i = [], 0
    for _ in range(n_cmds):
        n = gen.RSP_WORDS + units.decode_fields(rsp_words[i:i + gen.RSP_WORDS], gen.RSP_FIELDS)["out_words"] + gen.TRL_WORDS
        out.append(list(rsp_words[i:i + n]))
        i += n
    if i != len(rsp_words):
        raise SystemExit("expected response file does not frame into one response per command")
    return out


def main() -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--suite", nargs="+", default=["basic", "toy0", "toy1"])
    ap.add_argument("--vectors", default=str(_common.REPO / "artifacts" / "references" / "vectors"))
    ap.add_argument("--masks", help="worker body masks when no release is given, e.g. 0x03,0x44,0xB8")
    args = ap.parse_args()
    s = _common.Session(args, "G05")
    if args.masks and not s.rel:
        s.workers = _common.workers_of({}, [int(m, 16) for m in args.masks.split(",")])
        if args.backend == "emulated":
            from splicev_ps.fakexrt import FakeDevice
            s.dev = FakeDevice([int(w["mask"]) for w in s.workers])
    results = []
    try:
        pool = None
        if args.backend != "reference":
            from splicev_ps.workers import WorkerPool
            pool = WorkerPool(s.dev, s.workers, timeout_ms=args.timeout_ms)
        for rep in range(args.repeat):
            for suite in args.suite:
                vd = Path(args.vectors) / "units"
                cmds = units.split_commands(np.fromfile(vd / f"{suite}.cmd.bin", dtype="<u4").tolist())
                want = per_command(np.fromfile(vd / f"{suite}.rsp.bin", dtype="<u4").tolist(), len(cmds))
                if pool is None:
                    got = per_command(units.run_stream([w for c in cmds for w in c]), len(cmds))
                    t = 0.0
                else:
                    got, meta = pool.submit(cmds)
                    t = meta["t_submit_s"]
                bad = []
                for i, (g, w) in enumerate(zip(got, want)):
                    if [int(x) for x in g] != [int(x) for x in w]:
                        d = next((k for k in range(min(len(g), len(w))) if int(g[k]) != int(w[k])), min(len(g), len(w)))
                        bad.append({"unit": i, "opcode": units.decode_fields(cmds[i][:16], gen.DESC_FIELDS)["opcode"],
                                    "first_word": d, "got": int(g[d]) if d < len(g) else None, "want": int(w[d]) if d < len(w) else None})
                statuses = [units.split_responses(w)[0]["hdr"]["status"] for w in want]
                results.append({"repeat": rep, "suite": suite, "units": len(cmds), "response_words": sum(map(len, want)),
                                "non_ok_units_expected": sum(1 for x in statuses if x != gen.ST_OK),
                                "mismatching_units": len(bad), "first": bad[:5], "t_submit_s": t})
    finally:
        s.close()
    ok = all(r["mismatching_units"] == 0 for r in results)
    summary = {"suites": args.suite, "repeats": args.repeat, "workers": [hex(int(w["mask"])) for w in s.workers],
               "units": sum(r["units"] for r in results), "mismatching_units": sum(r["mismatching_units"] for r in results)}
    return s.finish("units", {"pass": ok, "summary": summary, "results": results})


if __name__ == "__main__":
    sys.exit(main())
