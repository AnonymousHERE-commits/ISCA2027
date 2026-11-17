"""Tensor bundles (.svb) shared by tools/make_vectors.py, the C++ testbenches (pl/tb/svb.h)
and the PS conformance apps.

Format: u32 magic 0x31425653 'SVB1', u32 count, then per tensor: u32 name length, name
bytes padded to 4, u32 ndim, u32 dims[ndim], int64 data (little endian).
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MAGIC = 0x31425653


class Bundle:
    def __init__(self):
        self.items: List[Tuple[str, np.ndarray]] = []

    def add(self, name: str, arr) -> None:
        self.items.append((name, np.asarray(arr, dtype=np.int64)))

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<II", MAGIC, len(self.items)))
            for name, a in self.items:
                nb = name.encode("ascii")
                fh.write(struct.pack("<I", len(nb)) + nb + b"\0" * ((-len(nb)) % 4))
                fh.write(struct.pack("<I", a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape))
                fh.write(a.astype("<i8").tobytes())


def read(path: Path) -> Dict[str, np.ndarray]:
    raw = Path(path).read_bytes()
    magic, count = struct.unpack_from("<II", raw, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: not an SVB1 bundle")
    off = 8
    out: Dict[str, np.ndarray] = {}
    for _ in range(count):
        (n,) = struct.unpack_from("<I", raw, off)
        off += 4
        name = raw[off:off + n].decode("ascii")
        off += n + ((-n) % 4)
        (nd,) = struct.unpack_from("<I", raw, off)
        off += 4
        dims = struct.unpack_from(f"<{nd}I", raw, off)
        off += 4 * nd
        size = int(np.prod(dims)) if nd else 1
        out[name] = np.frombuffer(raw, dtype="<i8", count=size, offset=off).reshape(dims).astype(np.int64)
        off += 8 * size
    return out
