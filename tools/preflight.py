#!/usr/bin/env python3
"""Read-only host / board inventory (roadmap 15.2, gate G01).

Desktop (after sourcing the 2025.2 settings):
    source /opt/AMD/2025.2/Vitis/settings64.sh
    python3 tools/preflight.py --target host --output evidence/preflight/host.json
Board (copy this single file; it needs only the Python standard library):
    scp tools/preflight.py petalinux@10.161.0.15:/home/petalinux/splicev_preflight.py
    ssh petalinux@10.161.0.15 'python3 /home/petalinux/splicev_preflight.py --target board --output /home/petalinux/splicev_board.json'
    scp petalinux@10.161.0.15:/home/petalinux/splicev_board.json evidence/preflight/board.json

Nothing is installed, programmed or changed.  Missing tools are recorded as missing, not
guessed; the report says what was found, where, and which version strings it printed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOST_TOOLS = ["v++", "vitis", "vitis-run", "vivado", "aiecompiler", "x86simulator", "aiesimulator", "xclbinutil",
              "platforminfo", "xsct", "bootgen", "petalinux-build", "g++", "make", "git", "python3"]
BOARD_TOOLS = ["xrt-smi", "xbutil", "xclbinutil", "xmutil", "fpgautil", "g++", "make", "python3", "git"]
VERSION_ARGS = {"vivado": ["-version"], "xsct": ["-eval", "puts [version]"], "petalinux-build": ["--version"], "make": ["--version"]}


def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": (p.stdout + p.stderr).strip()[-3000:]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def read(path, limit=4000):
    try:
        return Path(path).read_text(errors="replace")[:limit]
    except OSError:
        return None


def tools(names):
    out = {}
    for t in names:
        p = shutil.which(t)
        rec = {"path": p}
        if p and t not in ("vitis",):
            rec["version"] = run([t] + VERSION_ARGS.get(t, ["--version"]), timeout=90)
        out[t] = rec
    return out


def python_packages():
    out = {}
    for mod in ("numpy", "pyxrt", "pynq", "scipy"):
        r = run([sys.executable, "-c", f"import {mod} as m; print(getattr(m, '__version__', 'no __version__'), getattr(m, '__file__', ''))"])
        out[mod] = r
    return out


def common():
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version,
        "os_release": read("/etc/os-release"),
        "kernel": read("/proc/version"),
        "cpu_count": os.cpu_count(),
        "meminfo": (read("/proc/meminfo") or "")[:600],
        "disk_free_home": run(["df", "-h", str(Path.home())]),
        "user": os.environ.get("USER"),
        "groups": run(["id"]),
        "python_packages": python_packages(),
    }


def host_report(a):
    rep = common()
    env = {k: os.environ.get(k) for k in ("XILINX_VITIS", "XILINX_VIVADO", "XILINX_HLS", "XILINX_XRT", "PLATFORM_REPO_PATHS",
                                         "ROOTFS", "SYSROOT", "LD_LIBRARY_PATH", "PATH")}
    rep["environment"] = env
    rep["expected_settings"] = {p: Path(p).exists() for p in a.settings}
    rep["tools"] = tools(HOST_TOOLS)
    plats = []
    for root in [os.environ.get("PLATFORM_REPO_PATHS", "")] + a.platform_dirs:
        for r in filter(None, root.split(":")):
            plats += glob.glob(os.path.join(r, "**", "*.xpfm"), recursive=True)
    rep["xpfm_found"] = sorted(set(plats))[:200]
    if a.xpfm:
        rep["xpfm"] = {"path": a.xpfm, "exists": Path(a.xpfm).exists()}
        if shutil.which("platforminfo") and Path(a.xpfm).exists():
            rep["xpfm"]["platforminfo"] = run(["platforminfo", "-p", a.xpfm], timeout=300)
    return rep


def board_report(a):
    rep = common()
    rep["tools"] = tools(BOARD_TOOLS)
    rep["device_tree_model"] = read("/proc/device-tree/model")
    rep["device_tree_compatible"] = (read("/proc/device-tree/compatible") or "").replace("\x00", " ")
    rep["dri"] = sorted(glob.glob("/dev/dri/*"))
    rep["dev_nodes"] = sorted(glob.glob("/dev/zocl*") + glob.glob("/dev/xclmgmt*") + glob.glob("/dev/aie*"))
    rep["modules"] = run(["sh", "-c", "lsmod | grep -i -E 'zocl|xrt|aie|ai_engine' || true"])
    rep["zocl_sysfs"] = sorted(glob.glob("/sys/devices/platform/*zocl*"))[:20]
    rep["firmware"] = sorted(glob.glob("/lib/firmware/xilinx/*"))[:100]
    rep["xrt_dirs"] = {d: Path(d).exists() for d in ("/opt/xilinx/xrt", "/usr/lib/python3/dist-packages/pyxrt", "/etc/xocl.txt")}
    rep["xrt_ini"] = read("/etc/xrt.ini") or read(os.path.expanduser("~/xrt.ini"))
    for t in ("xrt-smi", "xbutil"):
        if shutil.which(t):
            rep[f"{t}_examine"] = run([t, "examine"], timeout=120)
            break
    if shutil.which("xmutil"):
        rep["xmutil_listapps"] = run(["xmutil", "listapps"])
    hw = []
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        rec = {"dir": d, "name": (read(d + "/name") or "").strip(), "inputs": sorted(os.path.basename(x) for x in glob.glob(d + "/*_input"))}
        hw.append(rec)
    rep["hwmon"] = hw
    rep["iio_devices"] = sorted(glob.glob("/sys/bus/iio/devices/*"))[:50]
    rep["boot_partition"] = run(["sh", "-c", "mount | grep -i -E 'boot|mmcblk|sd[a-z]' || true"])
    rep["uptime"] = read("/proc/uptime")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["host", "board"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--settings", nargs="*", default=["/opt/AMD/2025.2/Vitis/settings64.sh"])
    ap.add_argument("--platform-dirs", nargs="*", default=[], help="extra directories searched for .xpfm files")
    ap.add_argument("--xpfm", help="candidate platform to describe with platforminfo")
    a = ap.parse_args()
    rep = host_report(a) if a.target == "host" else board_report(a)
    rep["target"] = a.target
    rep["tool"] = "tools/preflight.py (read-only)"
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    found = {k: bool(v.get("path")) for k, v in rep["tools"].items()}
    print(json.dumps({"target": a.target, "tools_found": found, "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
