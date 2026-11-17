"""Timestamps and power sampling on the board.

Power comes from Linux hwmon channels that the board inventory (tools/preflight.py
--target board) lists.  Which channel covers which rail, and whether a reading is
whole-board or SoC-rail scope, is declared by the user in a JSON map; nothing here
assumes TE0950 rail names (roadmap 13.4).  Energy per frame = integral P dt over the
window / completed frames.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

HWMON = Path("/sys/class/hwmon")


def list_hwmon() -> List[Dict]:
    out = []
    if not HWMON.exists():
        return out
    for d in sorted(HWMON.glob("hwmon*")):
        name = (d / "name").read_text().strip() if (d / "name").exists() else "?"
        chans = sorted(p.name for p in d.iterdir() if p.name.endswith("_input"))
        labels = {}
        for c in chans:
            lab = d / c.replace("_input", "_label")
            if lab.exists():
                labels[c] = lab.read_text().strip()
        out.append({"path": str(d), "name": name, "inputs": chans, "labels": labels})
    return out


class PowerSampler:
    """Samples declared power channels (microwatts, hwmon power*_input) at a fixed rate."""

    def __init__(self, rail_map: Path, rate_hz: float = 1000.0):
        cfg = json.loads(Path(rail_map).read_text())
        self.scope = cfg.get("scope", "undeclared")
        self.channels = {k: Path(v) for k, v in cfg["channels"].items()}
        self.period = 1.0 / rate_hz
        self.samples: List[tuple] = []
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def _read(self) -> Dict[str, float]:
        return {k: int(p.read_text().strip()) * 1e-6 for k, p in self.channels.items()}

    def _loop(self):
        nxt = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter()
            self.samples.append((t, self._read()))
            nxt += self.period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> Dict:
        self._stop.set()
        if self._t:
            self._t.join()
        energy = {}
        for k in self.channels:
            e = 0.0
            for (t0, a), (t1, b) in zip(self.samples, self.samples[1:]):
                e += 0.5 * (a[k] + b[k]) * (t1 - t0)
            energy[k] = e
        dur = self.samples[-1][0] - self.samples[0][0] if len(self.samples) > 1 else 0.0
        return {"scope": self.scope, "samples": len(self.samples), "seconds": dur, "energy_J": energy,
                "achieved_rate_hz": (len(self.samples) - 1) / dur if dur else 0.0}
