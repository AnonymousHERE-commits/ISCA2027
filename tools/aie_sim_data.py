#!/usr/bin/env python3
"""PLIO data files for x86simulator / aiesimulator and the comparison of their outputs.

    python3 tools/aie_sim_data.py make  --suite basic --workers-json build/aie/uniform1/src/workers.json --out build/aie/uniform1/data
    python3 tools/aie_sim_data.py smoke --out build/aie/smoke/data
    python3 tools/aie_sim_data.py check --sim <work>/x86simulator_output/data --expected build/aie/uniform1/data

A 64-bit PLIO carries two 32-bit words per line.  'make' routes the units of a suite onto
the graph's worker classes (reference/splicev_ref/routing.py, the same routing the PS
uses), pads every worker with NOP units to the same count, and writes w<i>_in.txt, the
expected responses w<i>_expected.txt (integer reference with that worker's body mask)
and units.json with SV_SIM_UNITS (the per-worker unit count the graph main() runs).
'check' ignores simulator timestamp lines ('T ...') and TLAST markers and compares
every word.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference"))

from splicev_ref import gen, routing, units  # noqa: E402
from splicev_ref import unit_builders as ub  # noqa: E402

VEC = REPO / "artifacts" / "references" / "vectors"


def worker_masks(args):
    if args.workers_json:
        info = json.loads(Path(args.workers_json).read_text())
        return [int(w["mask"]) for w in info["workers"]]
    return [routing.FULL_MASK] * args.workers


def write_plio(path: Path, words) -> None:
    words = [int(w) & 0xFFFFFFFF for w in words]
    if len(words) % 2:
        raise ValueError(f"{path}: odd word count; every unit is padded to whole 64-bit beats")
    with open(path, "w") as fh:
        for a, b in zip(words[0::2], words[1::2]):
            fh.write(f"{np.int32(np.uint32(a))} {np.int32(np.uint32(b))}\n")


def read_plio(path: Path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("T") or line.startswith("TLAST"):
            continue
        out += [int(x) & 0xFFFFFFFF for x in line.split()]
    return out


def cmd_make(args) -> int:
    src = Path(args.vectors) / "units" / f"{args.suite}.cmd.bin"
    if not src.exists():
        raise SystemExit(f"{src} is missing: run python3 tools/make_vectors.py first")
    cmds = units.split_commands(np.fromfile(src, dtype="<u4").tolist())
    masks = worker_masks(args)
    per = routing.group(cmds, masks)
    n_units = max(len(p) for p in per)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = []
    for w, idx in enumerate(per):
        lst = [cmds[i] for i in idx]
        pad = n_units - len(lst)
        lst += [ub.nop(frame_id=0xFFFF, slot=w, token=k) for k in range(pad)]
        stream = [x for c in lst for x in c]
        write_plio(out / f"w{w}_in.txt", stream)
        write_plio(out / f"w{w}_expected.txt", units.run_stream(stream, masks[w]))
        counts.append({"worker": w, "mask": f"0x{masks[w]:02X}", "units": len(idx), "nop_padding": pad})
    info = {"suite": args.suite, "n_workers": len(masks), "SV_SIM_UNITS": n_units, "workers": counts,
            "manifest_sha256": gen.MANIFEST_SHA256}
    (out / "units.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"suite": args.suite, "SV_SIM_UNITS": n_units, "units": [c["units"] for c in counts]}))
    return 0


def cmd_smoke(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    stream, expected = [], []
    for n, add in ((2, 1), (6, 0x7FFFFFFF), (100, 12345)):
        data = [int(x) for x in rng.integers(0, 1 << 32, size=n, dtype=np.uint64)]
        stream += [0x534D4B30, n, add, 0] + data
        x = 0
        for v in data:
            x ^= v
        expected += [0x534D4B31, n] + [(v + add) & 0xFFFFFFFF for v in data] + [x, 0x534D4B32]
    write_plio(out / "smoke_in.txt", stream)
    write_plio(out / "smoke_expected.txt", expected)
    print('{"SMOKE_SIM_ITER": 3}')
    return 0


def cmd_units(args) -> int:
    """Print SV_SIM_UNITS of a data directory (0 when there is no units.json, e.g. smoke)."""
    f = Path(args.data) / "units.json"
    print(json.loads(f.read_text())["SV_SIM_UNITS"] if f.exists() else 0)
    return 0


def cmd_check(args) -> int:
    sim, exp = Path(args.sim), Path(args.expected)
    bad = 0
    for e in sorted(exp.glob("*_expected.txt")):
        name = e.name.replace("_expected.txt", "_out.txt")
        s = sim / name
        if not s.exists():
            print(f"MISSING {s}")
            bad = 1
            continue
        got, want = read_plio(s), read_plio(e)
        n = min(len(got), len(want))
        diff = [i for i in range(n) if got[i] != want[i]]
        ok = not diff and len(got) == len(want)
        print(f"{name}: {len(got)} words vs {len(want)} expected: {'bit-exact' if ok else 'MISMATCH'}"
              + ("" if ok else f" ({len(diff)} differ, first at {diff[0] if diff else n})"))
        bad |= not ok
    print("PASS" if not bad else "FAIL")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--suite", default="basic")
    m.add_argument("--vectors", default=str(VEC))
    m.add_argument("--workers", type=int, default=1, help="uniform full-mask workers (ignored with --workers-json)")
    m.add_argument("--workers-json", help="workers.json written by tools/gen_aie_graph.py")
    m.add_argument("--out", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--out", required=True)
    c = sub.add_parser("check"); c.add_argument("--sim", required=True); c.add_argument("--expected", required=True)
    u = sub.add_parser("units"); u.add_argument("--data", required=True)
    args = ap.parse_args()
    return {"make": cmd_make, "smoke": cmd_smoke, "check": cmd_check, "units": cmd_units}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
