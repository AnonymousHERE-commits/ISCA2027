"""Release identity: bind the xclbin, the manifest, the generated sources and the PS code.

A run refuses a release whose manifest hash differs from the generated module the host
imports (the descriptors carry MANIFEST_HASH16 as well), and every evidence file records
the release identity (roadmap §4: 'Never print a deployment PASS because a file exists')."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import splicev_gen as gen

from . import REPO


def sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def identity(xclbin: Optional[Path] = None, release_json: Optional[Path] = None) -> Dict:
    rel = {}
    if release_json and Path(release_json).exists():
        rel = json.loads(Path(release_json).read_text())
        if rel.get("manifest_sha256") != gen.MANIFEST_SHA256:
            raise SystemExit(f"release manifest {rel.get('manifest_sha256')} != host manifest {gen.MANIFEST_SHA256}: "
                             "rebuild or regenerate before running")
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "manifest_id": gen.MANIFEST_ID,
        "manifest_profile": gen.PROFILE,
        "manifest_sha256": gen.MANIFEST_SHA256,
        "deployable": bool(gen.DEPLOYABLE),
        "xclbin": str(xclbin) if xclbin else None,
        "xclbin_sha256": sha256_file(xclbin) if xclbin else None,
        "release": rel,
    }
