"""Thin, probing wrapper over the installed pyxrt (XRT Python bindings).

XRT's Python API has changed across releases (kernel construction with or without an
access mode or hardware context; bo read/write signatures; graph bindings).  This
module tries the documented forms in order, records which one worked, and fails with
a clear message when none does.  It never invents a call: tools/probe_pyxrt.py lists
what the installed module exposes, and the result is stored as evidence (roadmap 3.3).
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pyxrt  # type: ignore
except ImportError:  # pragma: no cover - only on the board
    pyxrt = None


class XrtError(RuntimeError):
    pass


def require_pyxrt():
    if pyxrt is None:
        raise XrtError("pyxrt is not importable: source the XRT setup (e.g. /opt/xilinx/xrt/setup.sh) on the board "
                       "or install the XRT Python bindings matching the platform's XRT")
    return pyxrt


def probe() -> Dict[str, Any]:
    """What the installed pyxrt exposes (stored by tools/probe_pyxrt.py)."""
    info: Dict[str, Any] = {"importable": pyxrt is not None}
    if pyxrt is None:
        return info
    info["module_file"] = getattr(pyxrt, "__file__", None)
    info["names"] = sorted(n for n in dir(pyxrt) if not n.startswith("_"))
    for cls in ("device", "xclbin", "kernel", "run", "bo", "graph", "hw_context", "uuid"):
        obj = getattr(pyxrt, cls, None)
        info[f"has_{cls}"] = obj is not None
        if obj is not None:
            info[f"{cls}_members"] = sorted(n for n in dir(obj) if not n.startswith("_"))
            info[f"{cls}_doc"] = (getattr(obj, "__init__", None).__doc__ or "")[:2000] if hasattr(obj, "__init__") else ""
    return info


def _state_name(state) -> str:
    return str(state).split(".")[-1]


@dataclass
class RunRecord:
    kernel: str
    t_start: float
    t_end: float
    state: str

    @property
    def seconds(self) -> float:
        return self.t_end - self.t_start


class Buffer:
    """One XRT buffer object with explicit synchronisation."""

    def __init__(self, dev: "Device", nbytes: int, group: int):
        x = require_pyxrt()
        self.nbytes = int(nbytes)
        flags = getattr(getattr(x.bo, "flags", x.bo), "normal", None)
        if flags is None:
            flags = getattr(x.bo, "normal")
        self.bo = x.bo(dev.dev, self.nbytes, flags, group)

    def write(self, data: np.ndarray, offset: int = 0) -> None:
        raw = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
        if raw.size + offset > self.nbytes:
            raise XrtError("buffer overflow on write")
        self.bo.write(raw, offset)
        self.bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, raw.size, offset)

    def read(self, nbytes: Optional[int] = None, offset: int = 0) -> np.ndarray:
        n = self.nbytes if nbytes is None else int(nbytes)
        self.bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, n, offset)
        raw = self.bo.read(n, offset)
        return np.frombuffer(bytes(raw), dtype=np.uint8).copy()

    def words(self, n_words: int) -> np.ndarray:
        return self.read(4 * n_words).view("<u4")


class Kernel:
    def __init__(self, dev: "Device", name: str):
        x = require_pyxrt()
        self.name = name
        self.dev = dev
        errors: List[str] = []
        self.k = None
        variants = []
        if dev.ctx is not None:
            variants.append(("hw_context", lambda: x.kernel(dev.ctx, name)))
        mode = getattr(getattr(x.kernel, "cu_access_mode", x.kernel), "shared", None)
        if mode is not None:
            variants.append(("device+uuid+shared", lambda: x.kernel(dev.dev, dev.uuid, name, mode)))
        variants.append(("device+uuid", lambda: x.kernel(dev.dev, dev.uuid, name)))
        for label, make in variants:
            try:
                self.k = make()
                self.variant = label
                break
            except Exception as exc:  # noqa: BLE001 - record and try the next documented form
                errors.append(f"{label}: {exc}")
        if self.k is None:
            raise XrtError(f"cannot open kernel '{name}': " + " | ".join(errors))

    def group(self, arg: int) -> int:
        return self.k.group_id(arg)

    def start(self, args: Dict[int, Any]):
        """Start a run setting only the given argument indices (AXI-stream arguments are
        never set by the host).  Prefers pyxrt.run + set_arg; falls back to kernel(*args)."""
        x = require_pyxrt()
        if hasattr(x, "run"):
            try:
                r = x.run(self.k)
                for i in sorted(args):
                    r.set_arg(i, args[i])
                r.start()
                return r
            except TypeError:
                pass
        n = max(args) + 1 if args else 0
        return self.k(*[args.get(i) for i in range(n)])


class Device:
    """Loaded device + xclbin.  Kernel names follow v++ link nk= instance names."""

    def __init__(self, xclbin: str | Path, index: int = 0):
        x = require_pyxrt()
        self.path = Path(xclbin)
        if not self.path.exists():
            raise XrtError(f"xclbin not found: {self.path}")
        self.dev = x.device(index)
        self.xclbin = x.xclbin(str(self.path))
        self.ctx = None
        try:
            self.uuid = self.dev.load_xclbin(self.xclbin)
            self.load_variant = "load_xclbin(xclbin)"
        except Exception:
            self.uuid = self.dev.load_xclbin(str(self.path))
            self.load_variant = "load_xclbin(path)"
        if hasattr(x, "hw_context"):
            try:
                self.ctx = x.hw_context(self.dev, self.uuid)
            except Exception:
                self.ctx = None
        self.kernels: Dict[str, Kernel] = {}
        self.graph = GraphControl(self)

    def kernel(self, name: str) -> Kernel:
        if name not in self.kernels:
            self.kernels[name] = Kernel(self, name)
        return self.kernels[name]

    def buffer(self, kernel: str, arg: int, nbytes: int) -> Buffer:
        return Buffer(self, nbytes, self.kernel(kernel).group(arg))

    def wait(self, run, kernel: str, timeout_ms: int, t0: float) -> RunRecord:
        try:
            state = run.wait(timeout_ms)
        except TypeError:
            state = run.wait()
        t1 = time.perf_counter()
        name = _state_name(state)
        if "COMPLETED" not in name:
            raise XrtError(f"{kernel}: run ended in state {name} after {1e3 * (t1 - t0):.1f} ms (timeout {timeout_ms} ms)")
        return RunRecord(kernel, t0, t1, name)


class GraphControl:
    """AIE graph lifecycle.  Order of preference: pyxrt.graph, the native helper
    library (ps/native/libsv_graph.so, XRT C++ API), or 'autostart' when the image
    was packaged without defer_aie_run (the graph runs from PDI load).

    XRT's graph run(iterations) treats 0 as 'run until ended' (the ADF simulator API
    uses -1 for that; do not mix the two).  end(cycles) stops the graph that many AIE
    cycles after its last run, so a persistent graph is ended with a small cycle count."""

    def __init__(self, dev: Device):
        self.dev = dev
        self.mode = "autostart"
        self.handle = None
        self.lib = None
        self.started = False
        self.notes: List[str] = []

    def open(self, name: str, mode: str = "auto") -> str:
        if mode == "autostart":
            self.mode = "autostart"
            return self.mode
        x = require_pyxrt()
        if mode in ("auto", "pyxrt") and hasattr(x, "graph"):
            try:
                self.handle = x.graph(self.dev.ctx, name) if self.dev.ctx is not None else x.graph(self.dev.dev, self.dev.uuid, name)
                self.mode = "pyxrt"
                return self.mode
            except Exception as exc:  # noqa: BLE001
                try:
                    self.handle = x.graph(self.dev.dev, self.dev.uuid, name)
                    self.mode = "pyxrt"
                    return self.mode
                except Exception as exc2:  # noqa: BLE001
                    self.notes.append(f"pyxrt.graph: {exc} / {exc2}")
                    if mode == "pyxrt":
                        raise XrtError(f"pyxrt.graph failed: {exc2}")
        if mode in ("auto", "native"):
            lib = Path(__file__).resolve().parents[1] / "native" / "libsv_graph.so"
            if lib.exists():
                self.lib = ctypes.CDLL(str(lib))
                self.lib.sv_graph_open.restype = ctypes.c_void_p
                self.lib.sv_graph_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                self.lib.sv_graph_run.argtypes = [ctypes.c_void_p, ctypes.c_int]
                self.lib.sv_graph_end.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
                self.lib.sv_graph_close.argtypes = [ctypes.c_void_p]
                self.handle = self.lib.sv_graph_open(str(self.dev.path).encode(), name.encode())
                if not self.handle:
                    raise XrtError("native graph helper could not open the graph (see stderr)")
                self.mode = "native"
                return self.mode
            self.notes.append(f"{lib} not built (make -C ps/native on the board)")
            if mode == "native":
                raise XrtError(f"{lib} not built (make -C ps/native on the board)")
        raise XrtError("no graph control available: the installed pyxrt has no usable graph binding and "
                       "ps/native/libsv_graph.so is not built.  Build the helper (make -C ps/native) or use an "
                       "image packaged with AIE_AUTOSTART=1 and --graph-mode autostart.  Notes: " + " | ".join(self.notes))

    def run(self, iterations: int = 0) -> None:
        """Start the graph (0: run until ended).  A graph that is already running is
        reported, not restarted."""
        try:
            if self.mode == "pyxrt":
                self.handle.run(iterations)
            elif self.mode == "native":
                if self.lib.sv_graph_run(ctypes.c_void_p(self.handle), int(iterations)) != 0:
                    raise XrtError("native graph run failed (see stderr)")
            else:
                return
            self.started = True
        except Exception as exc:  # noqa: BLE001
            if "running" in str(exc).lower():
                self.notes.append(f"graph already running: {exc}")
                return
            raise

    def end(self, cycles: int = 1000) -> None:
        if not self.started:
            return
        if self.mode == "pyxrt":
            try:
                self.handle.end(cycles)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"graph end: {exc}")
        elif self.mode == "native" and self.handle:
            self.lib.sv_graph_end(ctypes.c_void_p(self.handle), ctypes.c_ulonglong(cycles))
        self.started = False

    def close(self) -> None:
        if self.mode == "native" and self.handle:
            self.lib.sv_graph_close(ctypes.c_void_p(self.handle))
            self.handle = None
