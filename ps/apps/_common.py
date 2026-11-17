"""Shared command line, device opening and evidence handling of the gate applications.

Backends
  board      the real device through XRT (pyxrt); the only backend that produces board
             evidence (provenance M)
  emulated   the same PS code path (WorkerPool, Engines, buffers, argument indices) on
             ps/splicev_ps/fakexrt.py, i.e. the integer reference behind an emulated
             device; checks the host program, never the hardware
  reference  the integer reference alone (no device code)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "ps") not in sys.path:
    sys.path.insert(0, str(REPO / "ps"))

import splicev_ps  # noqa: E402,F401  (puts reference/ and generated/py on sys.path)
from splicev_ps import evidence, release  # noqa: E402
from splicev_ps.workers import load_workers_json  # noqa: E402


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", help="release.json written by tools/release.py (xclbin, graph name, worker classes, hashes)")
    ap.add_argument("--xclbin", help="xclbin to load (default: the one named in --release)")
    ap.add_argument("--backend", choices=["board", "emulated", "reference"], default="board")
    ap.add_argument("--graph", help="graph instance name (default: from the release, else 'g')")
    ap.add_argument("--graph-mode", choices=["auto", "pyxrt", "native", "autostart"], default=None,
                    help="graph control (default: autostart if the release was packaged with AIE autostart, else auto)")
    ap.add_argument("--keep-graph", action="store_true", help="do not end the graph at exit")
    ap.add_argument("--timeout-ms", type=int, default=5000, help="bound of every wait")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the whole test (repeatability evidence)")
    ap.add_argument("--evidence-dir", help="where to write the evidence JSON (board default: evidence/<gate>/)")
    return ap


def load_release(args) -> Dict:
    if not args.release:
        return {}
    rel = json.loads(Path(args.release).read_text())
    rel["_path"] = str(Path(args.release).resolve())
    return rel


def xclbin_path(args, rel: Dict) -> Optional[Path]:
    if args.xclbin:
        return Path(args.xclbin)
    if rel.get("xclbin"):
        p = Path(rel["xclbin"])
        return p if p.is_absolute() else Path(rel["_path"]).parent / p
    return None


def workers_of(rel: Dict, default_masks=None) -> List[Dict]:
    if rel.get("aie", {}).get("workers"):
        return rel["aie"]["workers"]
    if default_masks:
        return [{"index": i, "mask": m, "mm2s": f"mm2s_{i}", "s2mm": f"s2mm_{i}"} for i, m in enumerate(default_masks)]
    return load_workers_json(None)


class Session:
    """Opened backend + identity; close() ends the graph if this process started it."""

    def __init__(self, args, gate: str, smoke: bool = False):
        self.args = args
        self.gate = gate
        self.rel = load_release(args)
        self.workers = workers_of(self.rel)
        self.dev = None
        self.graph_mode = None
        xb = xclbin_path(args, self.rel)
        if args.backend == "board":
            if xb is None:
                raise SystemExit("--backend board needs --xclbin or --release")
            if self.rel.get("problems"):
                raise SystemExit("the release is incomplete (tools/release.py reported: " + "; ".join(self.rel["problems"]) + ")")
            from splicev_ps.xrtcompat import Device
            self.identity = release.identity(xb, Path(args.release) if args.release else None)
            self.dev = Device(xb)
            name = args.graph or self.rel.get("aie", {}).get("graph", "g")
            mode = args.graph_mode or ("autostart" if self.rel.get("aie", {}).get("autostart") else "auto")
            self.graph_mode = self.dev.graph.open(name, mode)
            self.dev.graph.run(0)
        else:
            self.identity = release.identity(None, None)
            if args.backend == "emulated":
                from splicev_ps.fakexrt import FakeDevice
                self.dev = FakeDevice([int(w["mask"]) for w in self.workers], smoke=smoke)
                self.graph_mode = "emulated"
        self.identity["backend"] = args.backend
        self.identity["mode"] = "host-orchestrated"

    def close(self):
        if self.args.backend == "board" and self.dev is not None and not self.args.keep_graph:
            self.dev.graph.end()
            self.dev.graph.close()

    def finish(self, name: str, body: Dict) -> int:
        body = dict(body)
        body["gate"] = self.gate
        body["identity"] = self.identity
        body["graph_mode"] = self.graph_mode
        if self.dev is not None and hasattr(self.dev, "graph"):
            body["graph_notes"] = getattr(self.dev.graph, "notes", [])
        body["provenance"] = "M" if self.args.backend == "board" else "S"
        ok = bool(body.get("pass"))
        out = None
        if self.args.evidence_dir or self.args.backend == "board":
            out = evidence.write(self.gate, name, body, Path(self.args.evidence_dir) if self.args.evidence_dir else None)
        label = "BOARD" if self.args.backend == "board" else self.args.backend.upper() + " (not board evidence)"
        print(json.dumps(body.get("summary", {}), indent=2))
        print(f"{self.gate} {name}: {'PASS' if ok else 'FAIL'} [{label}]" + (f" -> {out}" if out else ""))
        return 0 if ok else 1
