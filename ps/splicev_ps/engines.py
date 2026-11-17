"""Run the PL decision engines (sv_reserve, sv_linkage, sv_gate, sv_exchange1/2) with the
word ABI of reference/splicev_ref/pl_abi.py and return decoded outputs."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from splicev_ref import gen

from .xrtcompat import Device

TIMEOUT_MS = 20000


class Engines:
    def __init__(self, dev: Device, timeout_ms: int = TIMEOUT_MS):
        self.dev = dev
        self.timeout_ms = timeout_ms
        self.records = []

    def _call(self, name: str, inputs: Dict[int, np.ndarray], outputs: Dict[int, int], scalars: Dict[int, int], n_args: int):
        k = self.dev.kernel(name)
        bos = {}
        for arg, arr in inputs.items():
            arr = np.ascontiguousarray((np.asarray(arr, dtype=np.int64) & 0xFFFFFFFF).astype("<u4").ravel())
            bo = self.dev.buffer(name, arg, max(4, arr.size * 4))
            bo.write(arr)
            bos[arg] = bo
        for arg, n_words in outputs.items():
            bos[arg] = self.dev.buffer(name, arg, max(4, 4 * n_words))
        args = {}
        for i in range(n_args):
            args[i] = bos[i].bo if i in bos else int(scalars[i])
        t0 = time.perf_counter()
        run = k.start(args)
        rec = self.dev.wait(run, name, self.timeout_ms, t0)
        self.records.append(rec)
        return {arg: bos[arg].words(n) for arg, n in outputs.items()}, rec

    def reserve(self, cells: np.ndarray, n: int, width: int, count: int):
        out, rec = self._call("sv_reserve", {0: cells}, {1: gen.N_RESERVE, 2: 4}, {3: n, 4: width, 5: count}, 6)
        return out[1][: int(out[2][0])], out[2], rec

    def linkage(self, edges: np.ndarray, anchors: np.ndarray, n: int, n_edges: int, budget: int, f_cap: int = gen.F_CAP):
        out, rec = self._call("sv_linkage", {0: edges, 1: anchors}, {2: n, 3: n, 4: 8}, {5: n, 6: n_edges, 7: budget, 8: f_cap}, 9)
        return out[2], out[3], out[4], rec

    def gate(self, anchors: np.ndarray, n: int, n_frag: int):
        out, rec = self._call("sv_gate", {0: anchors}, {1: gen.F_CAP * gen.F_CAP // 32, 2: 12 * gen.F_CAP, 3: 2}, {4: n, 5: n_frag}, 6)
        return out[1], out[2], out[3], rec

    def exchange1(self, packed: Dict[str, np.ndarray], n_frag: int):
        ins = {0: packed["partners"], 1: packed["evidence"], 2: packed["npart"], 3: packed["need"], 4: packed["rho"], 5: packed["phi"]}
        outs = {6: gen.F_CAP * gen.MS, 7: 2 * gen.F_CAP * gen.MS, 8: gen.F_CAP}
        out, rec = self._call("sv_exchange1", ins, outs, {9: n_frag}, 10)
        return out[6], out[7], out[8], rec

    def exchange2(self, packed: Dict[str, np.ndarray], n_frag: int):
        ins = {0: packed["sources"], 1: packed["lam"], 2: packed["tv"], 3: packed["h"]}
        outs = {4: gen.F_CAP * gen.C_TRANSPORT // 4, 5: 2}
        out, rec = self._call("sv_exchange2", ins, outs, {6: n_frag}, 7)
        return out[4], out[5], rec
