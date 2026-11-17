"""Load, merge and hash SPLICE-V integer manifests.

A profile file may name a base file in "extends"; its "overrides" map dotted
paths to values, and its "additions" add new top-level sections.  The merged
manifest is the single numeric source for every generated consumer.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

REPO = Path(__file__).resolve().parents[2]
CONTRACTS = REPO / "contracts"


def _set_path(doc: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = doc
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"override path '{dotted}' does not exist in the base manifest")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"override path '{dotted}' does not exist in the base manifest")
    node[keys[-1]] = value


def load_raw(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Return the merged manifest for a profile file."""
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    raw = load_raw(path)
    if "extends" not in raw:
        merged = copy.deepcopy(raw)
        merged.setdefault("decisions", {})
        return merged
    base = load_manifest(path.parent / raw["extends"])
    merged = copy.deepcopy(base)
    for dotted, value in raw.get("overrides", {}).items():
        _set_path(merged, dotted, copy.deepcopy(value))
    for key, value in raw.get("additions", {}).items():
        if key in merged:
            raise KeyError(f"addition '{key}' collides with an existing section")
        merged[key] = copy.deepcopy(value)
    for key in ("manifest_id", "profile", "deployable", "note"):
        if key in raw:
            merged[key] = raw[key]
    decisions = dict(base.get("decisions", {}))
    decisions.update(raw.get("decisions", {}))
    merged["decisions"] = decisions
    merged["extends_chain"] = base.get("extends_chain", []) + [raw["extends"]]
    return merged


def canonical_bytes(doc: Dict[str, Any]) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def manifest_sha256(doc: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(doc)).hexdigest()


def hash16(doc: Dict[str, Any]) -> int:
    return int(manifest_sha256(doc)[:4], 16)


def iter_leaves(node: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from iter_leaves(v, f"{prefix}.{k}" if prefix else k)
    else:
        yield prefix, node


def null_fields(doc: Dict[str, Any], skip: Tuple[str, ...] = ("unresolved", "decisions", "refs", "provenance")) -> List[str]:
    out = []
    for path, value in iter_leaves(doc):
        if path.split(".")[0] in skip:
            continue
        if value is None:
            out.append(path)
    return out
