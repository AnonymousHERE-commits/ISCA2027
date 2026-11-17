"""Submit dispatcher units to the AIE workers through their PL movers (host-orchestrated mode).

One submission = one command list per worker (descriptors + payloads) streamed by
sv_mm2s while sv_s2mm collects exactly the expected response words.  Sinks are armed
before sources start, waits are bounded, and every response is checked (magic, check
words, frame/slot echo) before its payload is used.

The receive length of a worker is exact: reference/splicev_ref/units.response_length
predicts every response (statuses are decided before any body runs).  Units are routed to
worker classes by reference/splicev_ref/routing.py with the body masks of the built graph
(workers.json written by tools/gen_aie_graph.py, copied into the release).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from splicev_ref import gen, routing, units

from .xrtcompat import Device, RunRecord, XrtError


def expected_response_words(cmds: Sequence[Sequence[int]], body_mask: int = routing.FULL_MASK) -> int:
    return sum(units.response_length(c, body_mask) for c in cmds)


class Worker:
    """One worker tile with its mover pair mm2s_<i> / s2mm_<i> (connectivity nk= names)."""

    def __init__(self, dev: Device, index: int = 0, mask: int = routing.FULL_MASK, mm2s: Optional[str] = None,
                 s2mm: Optional[str] = None, capacity_words: int = 1 << 20, timeout_ms: int = 5000):
        self.dev = dev
        self.index = index
        self.mask = int(mask)
        self.mm2s_name = f"sv_mm2s:{{{mm2s or f'mm2s_{index}'}}}"
        self.s2mm_name = f"sv_s2mm:{{{s2mm or f's2mm_{index}'}}}"
        self.cap = capacity_words + (capacity_words & 1)
        self.timeout_ms = timeout_ms
        self.in_bo = dev.buffer(self.mm2s_name, 0, 4 * self.cap)
        self.out_bo = dev.buffer(self.s2mm_name, 0, 4 * self.cap)
        self.records: List[RunRecord] = []

    def start(self, cmds: Sequence[Sequence[int]], n_out: Optional[int] = None):
        """Arm the sink, then start the source.  Returns a handle for finish()."""
        words = np.asarray([int(w) & 0xFFFFFFFF for c in cmds for w in c], dtype="<u4")
        n_out = expected_response_words(cmds, self.mask) if n_out is None else int(n_out)
        if words.size % 2 or n_out % 2:
            raise XrtError("command or response length is not a whole number of 64-bit beats")
        if words.size > self.cap or n_out > self.cap:
            raise XrtError(f"worker {self.index}: {words.size} / {n_out} words exceed the buffer capacity {self.cap}")
        self.in_bo.write(words)
        s2mm = self.dev.kernel(self.s2mm_name)
        mm2s = self.dev.kernel(self.mm2s_name)
        t0 = time.perf_counter()
        run_out = s2mm.start({0: self.out_bo.bo, 2: n_out // 2})     # arm the sink first (roadmap 11.2)
        run_in = mm2s.start({0: self.in_bo.bo, 2: words.size // 2})
        return {"t0": t0, "run_in": run_in, "run_out": run_out, "n_in": int(words.size), "n_out": n_out}

    def finish(self, h) -> Tuple[List[int], Dict]:
        r_in = self.dev.wait(h["run_in"], self.mm2s_name, self.timeout_ms, h["t0"])
        r_out = self.dev.wait(h["run_out"], self.s2mm_name, self.timeout_ms, h["t0"])
        self.records += [r_in, r_out]
        rsp = self.out_bo.words(h["n_out"]).tolist()
        return rsp, {"worker": self.index, "t_submit_s": r_out.t_end - h["t0"], "words_in": h["n_in"], "words_out": h["n_out"]}

    def submit(self, cmds: Sequence[Sequence[int]], n_out: Optional[int] = None) -> Tuple[List[int], Dict]:
        return self.finish(self.start(cmds, n_out))


def check_responses(rsp: Sequence[int], cmds: Sequence[Sequence[int]]) -> List[Dict]:
    """Frame-level checks: one response per command, in order, with valid check words."""
    parsed = units.split_responses(rsp)
    if len(parsed) != len(cmds):
        raise XrtError(f"{len(parsed)} responses for {len(cmds)} commands")
    for p, c in zip(parsed, cmds):
        d = units.decode_fields(c[:gen.DESC_WORDS], gen.DESC_FIELDS)
        h = p["hdr"]
        if not (h["header_ok"] and h["trailer_ok"]):
            raise XrtError(f"corrupt response frame for frame {d['frame_id']} slot {d['slot']}")
        if h["frame_id"] != d["frame_id"] or h["slot"] != d["slot"]:
            raise XrtError("response belongs to another unit (stale or reordered stream)")
    return parsed


def load_workers_json(path: Optional[Path]) -> List[Dict]:
    """Worker list of the built graph; the default is one full-mask worker."""
    if path is None:
        return [{"index": 0, "mask": routing.FULL_MASK, "mm2s": "mm2s_0", "s2mm": "s2mm_0"}]
    info = json.loads(Path(path).read_text())
    return info["workers"] if "workers" in info else info["aie"]["workers"]


class WorkerPool:
    """All workers of the graph.  submit() routes units by class, runs every worker's
    share concurrently (all sinks armed, then all sources) and returns the responses in
    the original command order."""

    def __init__(self, dev: Device, workers: Sequence[Dict], timeout_ms: int = 5000, capacity_words: int = 1 << 20):
        self.workers = [Worker(dev, int(w["index"]), int(w["mask"]), w.get("mm2s"), w.get("s2mm"), capacity_words, timeout_ms)
                        for w in workers]
        self.masks = [w.mask for w in self.workers]
        self.log: List[Dict] = []

    def submit(self, cmds: Sequence[Sequence[int]]) -> Tuple[List[List[int]], Dict]:
        per = routing.group(cmds, self.masks)
        handles = {}
        t0 = time.perf_counter()
        for w, idx in enumerate(per):
            if idx:
                handles[w] = self.workers[w].start([cmds[i] for i in idx])
        by_cmd: Dict[int, List[int]] = {}
        metas = []
        for w, h in handles.items():
            rsp, meta = self.workers[w].finish(h)
            metas.append(meta)
            frames = check_responses(rsp, [cmds[i] for i in per[w]])
            pos = 0
            for i, fr in zip(per[w], frames):
                n = gen.RSP_WORDS + fr["hdr"]["out_words"] + gen.TRL_WORDS
                by_cmd[i] = rsp[pos:pos + n]
                pos += n
        meta = {"t_submit_s": time.perf_counter() - t0, "workers": metas, "units": len(cmds)}
        self.log.append(meta)
        return [by_cmd[i] for i in range(len(cmds))], meta
