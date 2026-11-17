#!/usr/bin/env python3
"""Probe the XRT Python binding and runtime installed on the board (roadmap 3.3, 11.1).

    python3 tools/probe_pyxrt.py --output evidence/preflight/pyxrt_probe.json
    python3 tools/probe_pyxrt.py --xclbin artifacts/release/smoke/splicev.xclbin --graph g --output ...

Without --xclbin the probe is read-only: it imports pyxrt, lists the classes and members
the PS runtime may use (device, xclbin, kernel, run, bo, graph, hw_context), the XRT
packages/version strings and the native graph helper.  With --xclbin it also loads that
xclbin, opens every kernel it names (memory group ids of each buffer argument) and tries
to open the graph (without running it).  'import pyxrt' succeeding is not evidence that
the graph API works; the loopback (ps/apps/g02_smoke.py) is.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ps"))


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {"cmd": " ".join(cmd), "rc": p.returncode, "out": (p.stdout + p.stderr)[-4000:]}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": " ".join(cmd), "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(REPO / "evidence" / "preflight" / "pyxrt_probe.json"))
    ap.add_argument("--xclbin")
    ap.add_argument("--graph", default="g")
    ap.add_argument("--kernels", nargs="*", default=["sv_mm2s:{mm2s_0}", "sv_s2mm:{s2mm_0}"])
    args = ap.parse_args()
    from splicev_ps import xrtcompat
    rep = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
           "machine": platform.machine(), "python": sys.version, "pyxrt": xrtcompat.probe()}
    rep["xrt_tools"] = {t: shutil.which(t) for t in ("xbutil", "xrt-smi", "xclbinutil", "xbmgmt")}
    for t in ("xrt-smi", "xbutil"):
        if rep["xrt_tools"].get(t):
            rep[f"{t}_examine"] = run([t, "examine"])
            break
    rep["native_graph_helper"] = (REPO / "ps" / "native" / "libsv_graph.so").exists()
    for f in ("/etc/os-release", "/proc/version"):
        try:
            rep[f] = Path(f).read_text()[:2000]
        except OSError:
            pass
    if args.xclbin:
        rep["xclbin"] = {"path": args.xclbin}
        try:
            dev = xrtcompat.Device(args.xclbin)
            rep["xclbin"]["load_variant"] = dev.load_variant
            rep["xclbin"]["hw_context"] = dev.ctx is not None
            ks = {}
            for k in args.kernels:
                try:
                    kk = dev.kernel(k)
                    groups = {}
                    for a in range(4):
                        try:
                            groups[a] = kk.group(a)
                        except Exception as exc:  # noqa: BLE001
                            groups[a] = f"n/a ({str(exc)[:80]})"
                    ks[k] = {"opened": True, "variant": kk.variant, "group_ids": groups}
                except Exception as exc:  # noqa: BLE001
                    ks[k] = {"opened": False, "error": str(exc)}
            rep["xclbin"]["kernels"] = ks
            try:
                rep["xclbin"]["graph_mode"] = dev.graph.open(args.graph, "auto")
            except Exception as exc:  # noqa: BLE001
                rep["xclbin"]["graph_error"] = str(exc)
            rep["xclbin"]["graph_notes"] = dev.graph.notes
        except Exception as exc:  # noqa: BLE001
            rep["xclbin"]["error"] = str(exc)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str) + "\n")
    p = rep["pyxrt"]
    print(f"pyxrt importable: {p.get('importable')}; graph binding: {p.get('has_graph')}; hw_context: {p.get('has_hw_context')}; "
          f"native helper built: {rep['native_graph_helper']} -> {out}")
    return 0 if p.get("importable") else 1


if __name__ == "__main__":
    sys.exit(main())
