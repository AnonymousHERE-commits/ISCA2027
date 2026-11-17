#!/usr/bin/env python3
"""Report the deployability of an integer manifest.

Lists every null (unresolved) field, every design decision (DD) and synthetic
value (SYN) that a profile fills in, and the profile's deployable flag.  The
exit code is 0 unless --require-deployable is given and the manifest is not
deployable: listing missing fields is evidence, not an installation error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from svgen.manifest import load_manifest, load_raw, manifest_sha256, null_fields  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(REPO / "contracts" / "manifest.dev.json"))
    ap.add_argument("--require-deployable", action="store_true")
    ap.add_argument("--json", help="write the report as JSON to this path")
    args = ap.parse_args()

    m = load_manifest(args.manifest)
    raw = load_raw(Path(args.manifest))
    nulls = null_fields(m)
    decisions = m.get("decisions", {})
    # synthetic values, and values calibrated on synthetic data (provenance S), block deployment
    syn = sorted(p for p, d in decisions.items() if d.get("kind") == "synthetic" or d.get("provenance") == "S")
    dds = sorted(p for p, d in decisions.items() if d.get("kind") == "design")
    undocumented = sorted(p for p in raw.get("overrides", {}) if p not in decisions)
    deployable = bool(m.get("deployable")) and not nulls and not syn and not undocumented
    report = {
        "manifest_id": m["manifest_id"], "profile": m["profile"], "sha256": manifest_sha256(m),
        "declared_deployable": bool(m.get("deployable")), "deployable": deployable,
        "null_fields": nulls, "synthetic_values": syn, "design_decisions": dds,
        "undocumented_overrides": undocumented, "unresolved_notes": [u["field"] for u in m.get("unresolved", [])],
    }
    print(f"manifest   {m['manifest_id']}  profile={m['profile']}")
    print(f"sha256     {report['sha256']}")
    print(f"null fields ({len(nulls)}):")
    for p in nulls:
        print(f"   - {p}")
    print(f"synthetic values ({len(syn)}): " + ", ".join(f"{p} [{decisions[p]['id']}]" for p in syn))
    print(f"design decisions ({len(dds)}): " + ", ".join(sorted({decisions[p]['id'] for p in dds})))
    if undocumented:
        print(f"UNDOCUMENTED overrides ({len(undocumented)}): {undocumented}")
    print(f"deployable: {'YES' if deployable else 'NO'}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    if undocumented:
        return 2
    if args.require_deployable and not deployable:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
