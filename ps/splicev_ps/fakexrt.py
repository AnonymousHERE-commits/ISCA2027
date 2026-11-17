"""Host emulation of the device interface used by the PS runtime (tests only).

FakeDevice offers the subset of xrtcompat.Device that workers.py, engines.py and the
apps call (kernel(name).start(args), buffer(kernel, arg, nbytes), wait(...), graph), with
the kernels emulated by the integer reference:

  sv_mm2s:{mm2s_i} -> AIE worker i (body mask) -> sv_s2mm:{s2mm_i}   (dispatcher units)
  sv_mm2s:{mm2s_0} -> smoke kernel            -> sv_s2mm:{s2mm_0}    (smoke=True)
  sv_reserve, sv_linkage, sv_gate, sv_exchange1, sv_exchange2         (pl_kernels_ref)

The argument tables below are copied from the extern "C" signatures of pl/engines/*.cpp;
tests/test_ps_board_paths.py checks them against those sources, so the emulation tests
the PS argument indices and buffer sizes against the real kernel interfaces.  A short
stream makes the sink 'hang' (TIMEOUT state), as a real counted s2mm would.  Results from
this module are emulation, never board evidence.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from splicev_ref import pl_kernels_ref, units

from .xrtcompat import RunRecord, XrtError

# name -> argument list (kind, name); kinds: 'in' / 'out' buffers, 'stream', 'scalar'
SIGNATURES: Dict[str, List[tuple]] = {
    "sv_mm2s": [("in", "mem"), ("stream", "s"), ("scalar", "n_beats")],
    "sv_s2mm": [("out", "mem"), ("stream", "s"), ("scalar", "n_beats")],
    "sv_reserve": [("in", "cells"), ("out", "picks"), ("out", "stats"), ("scalar", "n"), ("scalar", "width"), ("scalar", "count")],
    "sv_linkage": [("in", "edges"), ("in", "anchors"), ("out", "ids"), ("out", "order"), ("out", "stats"), ("scalar", "n"),
                   ("scalar", "n_edges"), ("scalar", "budget"), ("scalar", "f_cap")],
    "sv_gate": [("in", "anchors"), ("out", "mask"), ("out", "fused"), ("out", "stats"), ("scalar", "n"), ("scalar", "n_frag")],
    "sv_exchange1": [("in", "partners"), ("in", "evidence"), ("in", "npart"), ("in", "need"), ("in", "rho"), ("in", "phi"),
                     ("out", "sources"), ("out", "lam"), ("out", "status"), ("scalar", "n_frag")],
    "sv_exchange2": [("in", "sources"), ("in", "lam"), ("in", "tv"), ("in", "h"), ("out", "dv"), ("out", "stats"),
                     ("scalar", "n_frag")],
}


class FakeBuffer:
    def __init__(self, nbytes: int):
        self.nbytes = int(nbytes)
        self.mem = np.zeros(self.nbytes, dtype=np.uint8)
        self.bo = self

    def write(self, data, offset: int = 0) -> None:
        raw = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
        if raw.size + offset > self.nbytes:
            raise XrtError("buffer overflow on write")
        self.mem[offset:offset + raw.size] = raw

    def read(self, nbytes: Optional[int] = None, offset: int = 0) -> np.ndarray:
        n = self.nbytes if nbytes is None else int(nbytes)
        return self.mem[offset:offset + n].copy()

    def words(self, n_words: int) -> np.ndarray:
        return self.read(4 * n_words).view("<u4")

    def put_words(self, words) -> None:
        w = np.asarray([int(x) & 0xFFFFFFFF for x in words], dtype="<u4")
        if 4 * w.size > self.nbytes:
            raise XrtError("kernel wrote past the end of its buffer (host allocated too little)")
        self.mem[:4 * w.size] = w.view(np.uint8)


class FakeRun:
    def __init__(self, fn):
        self.fn = fn

    def wait(self, timeout_ms: int = 0):
        return self.fn()


class FakeKernel:
    def __init__(self, dev: "FakeDevice", name: str):
        self.dev = dev
        self.name = name
        self.base = name.split(":")[0]
        m = re.search(r"\{(.+)\}", name)
        self.instance = m.group(1) if m else None
        if self.base not in SIGNATURES:
            raise XrtError(f"no kernel named '{name}' in the emulated xclbin")
        self.sig = SIGNATURES[self.base]

    def group(self, arg: int) -> int:
        if self.sig[arg][0] not in ("in", "out"):
            raise XrtError(f"{self.name} argument {arg} is not a memory argument")
        return 0

    def start(self, args: Dict[int, Any]):
        for i, (kind, pname) in enumerate(self.sig):
            if kind == "stream":
                if i in args:
                    raise XrtError(f"{self.name}: the host must not set stream argument {pname}")
            elif i not in args:
                raise XrtError(f"{self.name}: argument {i} ({pname}) not set")
        extra = set(args) - set(range(len(self.sig)))
        if extra:
            raise XrtError(f"{self.name}: unknown argument indices {sorted(extra)}")
        return getattr(self, "_run_" + self.base)(args)

    # ------------------------------------------------------------ movers + AIE
    def _worker(self) -> int:
        return int(self.instance.split("_")[-1]) if self.instance else 0

    def _run_sv_mm2s(self, a):
        w = self._worker()
        words = a[0].words(2 * int(a[2])).tolist()
        self.dev.streams.setdefault(w, []).extend(self.dev.transform(w, words))
        return FakeRun(lambda: "ert_cmd_state.ERT_CMD_STATE_COMPLETED")

    def _run_sv_s2mm(self, a):
        w = self._worker()
        n = 2 * int(a[2])
        buf = a[0]

        def done():
            q = self.dev.streams.setdefault(w, [])
            if len(q) < n:
                return "ert_cmd_state.ERT_CMD_STATE_TIMEOUT"
            buf.put_words(q[:n])
            del q[:n]
            return "ert_cmd_state.ERT_CMD_STATE_COMPLETED"
        return FakeRun(done)

    # ------------------------------------------------------------ PL engines
    @staticmethod
    def _done(outputs):
        def fn():
            for buf, words in outputs:
                buf.put_words(words)
            return "ert_cmd_state.ERT_CMD_STATE_COMPLETED"
        return FakeRun(fn)

    def _run_sv_reserve(self, a):
        n = int(a[3])
        picks, stats = pl_kernels_ref.sv_reserve(a[0].words(n), n, int(a[4]), int(a[5]))
        return self._done([(a[1], picks), (a[2], stats)])

    def _run_sv_linkage(self, a):
        n, ne = int(a[5]), int(a[6])
        ids, order, stats = pl_kernels_ref.sv_linkage(a[0].words(3 * ne), a[1].words(n), n, ne, int(np.int32(a[7])), int(a[8]))
        return self._done([(a[2], ids), (a[3], order), (a[4], stats)])

    def _run_sv_gate(self, a):
        n = int(a[4])
        mask, fused, stats = pl_kernels_ref.sv_gate(a[0].words(5 * n), n, int(a[5]))
        return self._done([(a[1], mask), (a[2], fused), (a[3], stats)])

    def _run_sv_exchange1(self, a):
        from splicev_ref import gen
        F, N = gen.F_CAP, gen.N_CAP
        packed = {"partners": a[0].words(F * N), "evidence": a[1].words(F * N), "npart": a[2].words(F), "need": a[3].words(F),
                  "rho": a[4].words(F), "phi": a[5].words(4 * F)}
        src, lam, st = pl_kernels_ref.sv_exchange1(packed, int(a[9]))
        return self._done([(a[6], src), (a[7], lam), (a[8], st)])

    def _run_sv_exchange2(self, a):
        from splicev_ref import gen
        F, MS, C = gen.F_CAP, gen.MS, gen.C_TRANSPORT
        packed = {"sources": a[0].words(F * MS), "lam": a[1].words(2 * F * MS), "tv": a[2].words(F * MS * C // 4),
                  "h": a[3].words(F * C // 4)}
        dv, stats = pl_kernels_ref.sv_exchange2(packed, int(a[6]))
        return self._done([(a[4], dv), (a[5], stats)])


class _FakeGraph:
    def __init__(self):
        self.mode = "emulated"

    def open(self, name: str, mode: str = "auto") -> str:
        return self.mode

    def run(self, iterations: int = -1) -> None:
        pass

    def end(self) -> None:
        pass


class FakeDevice:
    """Emulated device.  masks: body mask per worker; smoke: emulate the G02 smoke graph."""

    emulated = True

    def __init__(self, masks: Sequence[int] = (0xFF,), smoke: bool = False, path: str = "<emulated>"):
        self.masks = list(masks)
        self.smoke = smoke
        self.path = path
        self.streams: Dict[int, List[int]] = {}
        self.graph = _FakeGraph()
        self.load_variant = "emulated"
        self.ctx = None

    def kernel(self, name: str) -> FakeKernel:
        return FakeKernel(self, name)

    def buffer(self, kernel: str, arg: int, nbytes: int) -> FakeBuffer:
        self.kernel(kernel).group(arg)
        return FakeBuffer(nbytes)

    def wait(self, run, kernel: str, timeout_ms: int, t0: float) -> RunRecord:
        state = run.wait(timeout_ms)
        t1 = time.perf_counter()
        name = str(state).split(".")[-1]
        if "COMPLETED" not in name:
            raise XrtError(f"{kernel}: run ended in state {name} (emulated)")
        return RunRecord(kernel, t0, t1, name)

    def transform(self, worker: int, words: List[int]) -> List[int]:
        if self.smoke:
            return smoke_response(words)
        if worker >= len(self.masks):
            raise XrtError(f"no AIE worker {worker} behind mm2s_{worker}")
        return units.run_stream(words, self.masks[worker])


def smoke_response(words: Sequence[int]) -> List[int]:
    """The G02 smoke kernel (aie/smoke/sv_smoke.cc) on a stream of frames."""
    out: List[int] = []
    i = 0
    w = [int(x) & 0xFFFFFFFF for x in words]
    while i + 4 <= len(w):
        magic, n, add = w[i], w[i + 1], w[i + 2]
        i += 4
        if magic != 0x534D4B30:
            n = 0
        data = w[i:i + n]
        i += n
        x = 0
        for v in data:
            x ^= v
        out += [0x534D4B31, n] + [(v + add) & 0xFFFFFFFF for v in data] + [x, 0x534D4B32]
    return out
