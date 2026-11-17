"""Evidence records: immutable JSON reports with provenance and source tags.

Every number written here carries its provenance class (SPLICE-V §4): S simulation,
R post-route report, T training/validation, M board measurement.  A report states the
gate it serves, the mode (host-orchestrated or device-scheduled) and pass/fail from
executed checks only."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

from . import REPO


def write(gate: str, name: str, body: Dict, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else REPO / "evidence" / gate
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    path = out_dir / f"{name}_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(body, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return path
