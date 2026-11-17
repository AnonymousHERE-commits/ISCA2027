#!/usr/bin/env python3
"""Generate the C++ (AIE, HLS) and Python (PS, reference) sources from the integer manifest.

    python3 tools/generate_sources.py                      # dev profile -> generated/
    python3 tools/generate_sources.py --check              # exit 1 if generated/ is stale
    python3 tools/generate_sources.py --manifest contracts/manifest.model.json --out build/gen_model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from svgen.manifest import load_manifest  # noqa: E402
from svgen.emit import generate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO / "contracts" / "manifest.dev.json"))
    ap.add_argument("--out", default=str(REPO / "generated"))
    ap.add_argument("--check", action="store_true", help="compare with existing files instead of writing")
    args = ap.parse_args()

    m = load_manifest(args.manifest)
    files = generate(m)
    out = Path(args.out)
    stale = []
    for rel, text in files.items():
        path = out / rel
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(rel)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if args.check:
        if stale:
            print("STALE generated files (run tools/generate_sources.py):")
            for rel in stale:
                print("  ", rel)
            return 1
        print(f"generated/ is current for {m['manifest_id']} ({m['profile']})")
        return 0
    print(f"wrote {len(files)} files for {m['manifest_id']} ({m['profile']}, deployable={m['deployable']}) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
