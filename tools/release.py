#!/usr/bin/env python3
"""Bind one built system into a release: artifacts/release/<system>/release.json + xclbin.

    python3 tools/release.py --system slice1 --platform /path/to/te0950.xpfm [--caps small] [--autostart]

Records the identity the roadmap requires for every run (section 4): manifest and
generated-source hashes, git commit (and whether the tree was dirty), platform XPFM hash,
AIE archive, PL kernel objects, XSA, XCLBIN, boot files, link/package configurations,
test-vector hashes and tool versions; plus what the PS runtime needs instead of hard-coded
names: graph instance name, worker classes and mover/engine instance names.  When
xclbinutil is available the compute-unit names inside the xclbin are read back and every
expected instance must be present, otherwise the release is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "generated" / "py"))

import splicev_gen as gen  # noqa: E402


def sha256(path: Path):
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root: Path, pattern: str = "**/*"):
    h = hashlib.sha256()
    files = sorted(p for p in Path(root).glob(pattern) if p.is_file())
    for p in files:
        h.update(str(p.relative_to(root)).encode() + b"\0" + (sha256(p) or "").encode())
    return h.hexdigest() if files else None


def sh(cmd, cwd=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return -1, str(exc)


def xclbin_instances(xclbin: Path):
    """CU instance names from the xclbin IP_LAYOUT section (None when xclbinutil is absent)."""
    if not shutil.which("xclbinutil"):
        return None, "xclbinutil not on PATH"
    out = xclbin.parent / "ip_layout.json"
    rc, msg = sh(["xclbinutil", "--force", "--input", str(xclbin), "--dump-section", f"IP_LAYOUT:JSON:{out}"])
    if rc != 0 or not out.exists():
        return None, msg[-500:]
    data = json.loads(out.read_text())
    ips = data.get("ip_layout", {}).get("m_ip_data", [])
    return sorted(ip.get("m_name", "") for ip in ips), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True)
    ap.add_argument("--platform", required=True, help="the .xpfm the system was linked against")
    ap.add_argument("--caps", default=None, help="kernel capacity set (default: the system's)")
    ap.add_argument("--aie-target", default="hw")
    ap.add_argument("--autostart", action="store_true", help="packaged without defer_aie_run")
    ap.add_argument("--build", default=str(REPO / "build"))
    ap.add_argument("--vectors", default=str(REPO / "artifacts" / "references" / "vectors"))
    ap.add_argument("--out-root", default=str(REPO / "artifacts" / "release"))
    a = ap.parse_args()
    table = json.loads((REPO / "pl" / "kernels.json").read_text())
    sysdef = table["systems"][a.system]
    caps = a.caps or sysdef["caps"]
    B = Path(a.build)
    sysdir = B / "system" / a.system
    aie_set = sysdef["aie"]
    aie_dir = B / "aie" / aie_set / a.aie_target
    xclbin = sysdir / "splicev.xclbin"
    problems = []
    if not xclbin.exists():
        problems.append(f"missing {xclbin} (run make package SYSTEM={a.system})")
    if aie_set == "smoke":
        graph, workers = "smoke", [{"index": 0, "mask": 0, "mm2s": "mm2s_0", "s2mm": "s2mm_0", "kernel": "sv_smoke"}]
        workers_src = None
    else:
        wj = B / "aie" / aie_set / "src" / "workers.json"
        winfo = json.loads(wj.read_text()) if wj.exists() else None
        if winfo is None:
            problems.append(f"missing {wj}")
        graph, workers = (winfo or {}).get("graph", "g"), (winfo or {}).get("workers", [])
        workers_src = str(wj)
    xos = {}
    for k in sorted(set(sysdef["kernels"])):
        xo = B / "hls" / caps / k / f"{k}.xo"
        xos[k] = {"path": str(xo), "sha256": sha256(xo)}
        if not xo.exists():
            problems.append(f"missing {xo}")
    expected_cus = [w["mm2s"] for w in workers] + [w["s2mm"] for w in workers] + \
        [f"{k}_0" for k in sysdef["kernels"] if k not in ("sv_mm2s", "sv_s2mm")]
    cus, why = (xclbin_instances(xclbin) if xclbin.exists() else (None, "no xclbin"))
    if cus is not None:
        names = {n.split(":")[-1] for n in cus}
        missing = [c for c in expected_cus if c not in names]
        if missing:
            problems.append(f"xclbin lacks compute units {missing}; found {cus}")
    rc, git = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    _, dirty = sh(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    _, vpp = sh(["v++", "--version"]) if shutil.which("v++") else (0, None)
    boot = sorted(str(p) for p in (sysdir / "package").glob("**/BOOT.BIN")) if (sysdir / "package").exists() else []
    rel = {
        "system": a.system,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "xclbin": "splicev.xclbin",
        "xclbin_sha256": sha256(xclbin),
        "manifest_id": gen.MANIFEST_ID,
        "manifest_profile": gen.PROFILE,
        "manifest_sha256": gen.MANIFEST_SHA256,
        "deployable": bool(gen.DEPLOYABLE),
        "generated_sha256": tree_sha256(REPO / "generated"),
        "git_commit": git if rc == 0 else None,
        "git_dirty": bool(dirty.strip()) if rc == 0 else None,
        "platform": {"xpfm": str(Path(a.platform).resolve()), "sha256": sha256(Path(a.platform))},
        "aie": {"set": aie_set, "graph": graph, "autostart": bool(a.autostart), "workers": workers, "workers_json": workers_src,
                "libadf": str(aie_dir / "libadf.a"), "libadf_sha256": sha256(aie_dir / "libadf.a")},
        "pl": {"caps": caps, "caps_defines": table["caps"][caps], "kernels": xos},
        "system_files": {n: {"path": str(sysdir / n), "sha256": sha256(sysdir / n)}
                         for n in ("splicev.xsa", "connectivity.cfg", "package.cfg")},
        "boot_files": [{"path": p, "sha256": sha256(Path(p))} for p in boot],
        "xclbin_compute_units": cus,
        "xclbin_check": why or "all expected compute units present",
        "expected_compute_units": expected_cus,
        "vectors": {"dir": a.vectors, "sha256": tree_sha256(Path(a.vectors))},
        "tools": {"v++": vpp},
        "problems": problems,
    }
    out = Path(a.out_root) / a.system
    out.mkdir(parents=True, exist_ok=True)
    if xclbin.exists():
        shutil.copy2(xclbin, out / "splicev.xclbin")
    (out / "release.json").write_text(json.dumps(rel, indent=2) + "\n")
    print(f"{out / 'release.json'}: " + ("OK" if not problems else "INCOMPLETE: " + "; ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
