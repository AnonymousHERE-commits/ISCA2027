#!/usr/bin/env python3
"""Generate the AIE worker graph sources for one worker-class set.

    python3 tools/gen_aie_graph.py --set uniform1 --out build/aie/uniform1/src
    python3 tools/gen_aie_graph.py --masks 0x03,0x44,0xB8 --out build/aie/custom/src

Writes, from aie/templates/: sv_worker_m<XX>.cc (one AIE kernel per distinct class),
sv_workers.h, sv_graph.h, sv_graph.cpp (graph instance 'g') and workers.json, the
machine-readable worker list (index, body mask, kernel, PLIO names) that the PS runtime,
the simulator data tool and tools/release.py read instead of hard-coding names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "aie" / "templates"
OPNAMES = ["CONV", "GEMM", "STENCIL", "LUT", "PERMUTE", "SEGOPS", "TAPMOD", "HEAD"]


def load_set(name: str):
    sets = json.loads((REPO / "aie" / "worker_classes.json").read_text())["sets"]
    if name not in sets:
        raise SystemExit(f"unknown worker-class set '{name}'; known: {', '.join(sorted(sets))}")
    return [int(m, 16) if isinstance(m, str) else int(m) for m in sets[name]["masks"]]


def render(name: str, **kv) -> str:
    s = (TEMPLATES / name).read_text()
    for k, v in kv.items():
        s = s.replace(f"@{k}@", str(v))
    left = re.findall(r"@[A-Z_]+@", s)
    if left:
        raise SystemExit(f"unsubstituted placeholders {sorted(set(left))} in {name}")
    return s


def generate(masks, out: Path, graph: str = "g", stack: int = 4096, set_name: str = "custom") -> dict:
    for m in masks:
        if not 0 < m <= 0xFF:
            raise SystemExit(f"invalid body mask {m:#x}")
    covered = 0
    for m in masks:
        covered |= m
    if covered != 0xFF:
        missing = [OPNAMES[i] for i in range(8) if not (covered >> i) & 1]
        raise SystemExit(f"worker set leaves opcodes without a worker: {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    kernels = {m: f"sv_worker_m{m:02x}" for m in sorted(set(masks))}
    files = {}
    for m, kname in kernels.items():
        files[f"{kname}.cc"] = render("sv_worker.cc.in", MASK_HEX=f"0x{m:02X}", KERNEL=kname)
    decls = "\n".join(f"void {k}(input_stream<int32>* __restrict in, output_stream<int32>* __restrict out);" for k in kernels.values())
    files["sv_workers.h"] = render("sv_workers.h.in", DECLS=decls + "\n")
    create = "\n".join(f'        k[{i}] = adf::kernel::create({kernels[m]});\n        adf::source(k[{i}]) = "{kernels[m]}.cc";'
                       for i, m in enumerate(masks))
    classes = ", ".join(f"0x{m:02X}" for m in masks)
    files["sv_graph.h"] = render("sv_graph.h.in", N=len(masks), CLASSES=classes, CREATE=create, STACK=stack)
    files["sv_graph.cpp"] = render("sv_graph.cpp.in", GRAPH=graph)
    for name, text in files.items():
        (out / name).write_text(text)
    workers = [{"index": i, "mask": m, "mask_hex": f"0x{m:02X}", "bodies": [OPNAMES[b] for b in range(8) if (m >> b) & 1],
                "kernel": kernels[m], "plio_in": f"w{i}_in", "plio_out": f"w{i}_out",
                "mm2s": f"mm2s_{i}", "s2mm": f"s2mm_{i}"} for i, m in enumerate(masks)]
    digest = hashlib.sha256("".join(files[k] for k in sorted(files)).encode()).hexdigest()
    info = {"set": set_name, "graph": graph, "n_workers": len(masks), "workers": workers, "sources_sha256": digest,
            "templates": sorted(p.name for p in TEMPLATES.glob("*.in"))}
    (out / "workers.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", help="worker-class set from aie/worker_classes.json")
    g.add_argument("--masks", help="comma-separated body masks, e.g. 0x03,0x44,0xB8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--graph", default="g", help="graph instance name used by XRT")
    ap.add_argument("--stack", type=int, default=4096, help="stack bytes per worker tile")
    args = ap.parse_args()
    masks = load_set(args.set) if args.set else [int(x, 16) for x in args.masks.split(",")]
    info = generate(masks, Path(args.out), args.graph, args.stack, args.set or "custom")
    print(json.dumps({"set": info["set"], "workers": [w["mask_hex"] for w in info["workers"]], "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
