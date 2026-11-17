#!/usr/bin/env python3
"""Render the tool configuration files of one build from the repository tables.

    build_cfg.py hls --kernel sv_gate --caps small --out build/hls/small/sv_gate/hls_config.cfg
    build_cfg.py connectivity --system slice1 --workers-json build/aie/uniform1/src/workers.json --out <cfg>
    build_cfg.py package --out <cfg> --out-dir <dir> [--autostart] [--rootfs F --kernel-image F]

All paths written into the files are absolute (no relative-path or quoting surprises
between make, Tcl and the compilers).  Kernel names, instance names and PLIO names come
from pl/kernels.json and the generated workers.json, never from hand-typed lists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def kernels_table():
    return json.loads((REPO / "pl" / "kernels.json").read_text())


def render(template: Path, **kv) -> str:
    s = template.read_text()
    for k, v in kv.items():
        s = s.replace(f"@{k}@", str(v))
    if "@" in "".join(line for line in s.splitlines() if not line.lstrip().startswith("#")):
        raise SystemExit(f"unsubstituted placeholder in {template}")
    return s


def cmd_hls(a) -> str:
    t = kernels_table()
    if a.kernel not in t["kernels"]:
        raise SystemExit(f"unknown kernel '{a.kernel}'; pl/kernels.json lists: {', '.join(t['kernels'])}")
    if a.caps not in t["caps"]:
        raise SystemExit(f"unknown capacity set '{a.caps}'; pl/kernels.json lists: {', '.join(t['caps'])}")
    k = t["kernels"][a.kernel]
    caps = t["caps"][a.caps]
    inc = " ".join(f"-I{REPO / p}" for p in t["include"])
    defs = " ".join(t["cflags"] + [f"-D{n}={v}" for n, v in caps.items()])
    tb = ""
    if k.get("tb") and not a.no_tb:
        tb = (f"tb.file={REPO / k['tb']}\n"
              f"tb.cflags={inc} -I{REPO / 'pl' / 'tb'} {defs} -D{k['tb_define']}")
    out = Path(a.out)
    return render(REPO / "pl" / "configs" / "hls.cfg.in", PART=t["part"], XO=out.parent / f"{a.kernel}.xo",
                  SYN_FILE=REPO / k["file"], TOP=a.kernel, SYN_CFLAGS=f"{inc} {defs}", TB_LINES=tb, CLOCK=t["clock"])


def cmd_connectivity(a) -> str:
    t = kernels_table()
    if a.system not in t["systems"]:
        raise SystemExit(f"unknown system '{a.system}'; pl/kernels.json lists: {', '.join(t['systems'])}")
    sysdef = t["systems"][a.system]
    lines = [f"# v++ --link connectivity for system '{a.system}' (rendered by tools/build_cfg.py)", "[connectivity]"]
    if sysdef["aie"] == "smoke":
        lines += ["nk=sv_mm2s:1:mm2s_0", "nk=sv_s2mm:1:s2mm_0",
                  "stream_connect=mm2s_0.s:ai_engine_0.smoke_in", "stream_connect=ai_engine_0.smoke_out:s2mm_0.s"]
        cus = ["mm2s_0", "s2mm_0"]
    else:
        info = json.loads(Path(a.workers_json).read_text())
        n = info["n_workers"]
        lines += ["nk=sv_mm2s:%d:%s" % (n, ".".join(w["mm2s"] for w in info["workers"])),
                  "nk=sv_s2mm:%d:%s" % (n, ".".join(w["s2mm"] for w in info["workers"]))]
        for w in info["workers"]:
            lines += [f"stream_connect={w['mm2s']}.s:ai_engine_0.{w['plio_in']}",
                      f"stream_connect=ai_engine_0.{w['plio_out']}:{w['s2mm']}.s"]
        cus = [w["mm2s"] for w in info["workers"]] + [w["s2mm"] for w in info["workers"]]
        for k in sysdef["kernels"]:
            if k not in ("sv_mm2s", "sv_s2mm"):
                lines.append(f"nk={k}:1:{k}_0")
                cus.append(f"{k}_0")
    if a.clock_id is not None:
        lines += ["", "[clock]"] + [f"id={a.clock_id}:{cu}" for cu in cus]
    if a.sp:
        lines += [f"sp={line}" for line in a.sp]
    return "\n".join(lines) + "\n"


def cmd_package(a) -> str:
    img = "no_image=1"
    if a.rootfs and a.kernel_image:
        img = f"rootfs={Path(a.rootfs).resolve()}\nkernel_image={Path(a.kernel_image).resolve()}"
    return render(REPO / "platform" / "package.cfg.in", OUT_DIR=Path(a.out_dir).resolve(),
                  DEFER_LINE="" if a.autostart else "defer_aie_run=1", IMAGE_LINES=img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hls")
    h.add_argument("--kernel", required=True)
    h.add_argument("--caps", default="full")
    h.add_argument("--no-tb", action="store_true")
    h.add_argument("--out", required=True)
    c = sub.add_parser("connectivity")
    c.add_argument("--system", required=True)
    c.add_argument("--workers-json")
    c.add_argument("--clock-id", type=int, help="platform clock id for every CU (default: platform default clock)")
    c.add_argument("--sp", nargs="*", help="memory mapping lines '<cu>.<arg>:<bank>' (default: linker choice)")
    c.add_argument("--out", required=True)
    p = sub.add_parser("package")
    p.add_argument("--out", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--autostart", action="store_true", help="start the AIE graph at PDI load (no host graph control)")
    p.add_argument("--rootfs")
    p.add_argument("--kernel-image")
    a = ap.parse_args()
    text = {"hls": cmd_hls, "connectivity": cmd_connectivity, "package": cmd_package}[a.cmd](a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
