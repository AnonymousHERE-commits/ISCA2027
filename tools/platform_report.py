#!/usr/bin/env python3
"""Describe a candidate platform: hashes and platforminfo output (roadmap 3.2).

    python3 tools/platform_report.py --xpfm <te0950.xpfm> [--xsa <hw.xsa>] \
        [--board-json evidence/preflight/board.json] --output evidence/preflight/platform_report.json

Starts from platform/platform_report_template.json, fills the fields it can read, stores the
raw platforminfo text, and leaves everything else null.  Read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(p):
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xpfm", required=True)
    ap.add_argument("--xsa")
    ap.add_argument("--board-json")
    ap.add_argument("--output", default=str(REPO / "evidence" / "preflight" / "platform_report.json"))
    a = ap.parse_args()
    rep = json.loads((REPO / "platform" / "platform_report_template.json").read_text())
    rep.pop("_doc", None)
    rep["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rep["xpfm"] = {"path": str(Path(a.xpfm).resolve()), "sha256": sha256(a.xpfm)}
    if a.xsa:
        rep["xsa"] = {"path": str(Path(a.xsa).resolve()), "sha256": sha256(a.xsa)}
    if shutil.which("platforminfo") and Path(a.xpfm).exists():
        p = subprocess.run(["platforminfo", "-p", a.xpfm], capture_output=True, text=True, timeout=600)
        text = p.stdout + p.stderr
        rep["platforminfo"] = text[-20000:]
        m = re.search(r"Part:\s*(\S+)", text)
        rep["part"] = m.group(1) if m else None
        rep["clocks"] = [{"id": int(i), "name": n, "freq_hz": None, "default": None}
                         for i, n in re.findall(r"Clock Index:\s*(\d+)\s*\n\s*Frequency:.*?\n?.*?Name:\s*(\S+)", text)] or rep["clocks"]
        rep["notes"].append("parsed fields are best-effort; the raw platforminfo text is authoritative")
    else:
        rep["notes"].append("platforminfo not available: source the Vitis settings to fill clocks/memories")
    if a.board_json and Path(a.board_json).exists():
        b = json.loads(Path(a.board_json).read_text())
        rep["linux"]["kernel"] = b.get("kernel")
        rep["power_monitors"] = b.get("hwmon", [])
        rep["xrt"]["pyxrt"] = b.get("python_packages", {}).get("pyxrt")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
