"""A small array-API shim so the SPLICE neck math runs on numpy (tests, synthetic-scene
training) and on torch (training on GPUs) from one source.

Only the operations the neck needs are provided.  Shapes follow numpy conventions;
`axis` is used for both libraries.  Cell tensors are (N, C, H, W).
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np


class NumpyBackend:
    name = "numpy"

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    # construction
    def asarray(self, x, like=None):
        return np.asarray(x, dtype=self.dtype)

    def zeros(self, shape, like=None):
        return np.zeros(shape, dtype=self.dtype)

    def ones(self, shape, like=None):
        return np.ones(shape, dtype=self.dtype)

    def full(self, shape, value, like=None):
        return np.full(shape, value, dtype=self.dtype)

    def arange(self, n, like=None):
        return np.arange(n)

    def eye(self, n, like=None):
        return np.eye(n, dtype=self.dtype)

    def index(self, x, like=None):
        return np.asarray(x, dtype=np.int64)

    # elementwise
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    abs = staticmethod(np.abs)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    where = staticmethod(np.where)

    def sigmoid(self, x):
        return 0.5 * (1.0 + np.tanh(0.5 * x))

    def softplus(self, x):
        return np.logaddexp(0.0, x)

    def relu(self, x):
        return np.maximum(x, 0.0)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def round_half_even(self, x):
        return np.round(x)

    # reductions
    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    def amax(self, x, axis=None, keepdims=False):
        return np.max(x, axis=axis, keepdims=keepdims)

    def cumsum(self, x, axis):
        return np.cumsum(x, axis=axis)

    # structure
    einsum = staticmethod(np.einsum)

    def stack(self, xs, axis=0):
        return np.stack(xs, axis=axis)

    def concat(self, xs, axis=0):
        return np.concatenate(xs, axis=axis)

    def reshape(self, x, shape):
        return np.reshape(x, shape)

    def take(self, x, index, axis=0):
        return np.take(x, index, axis=axis)

    def scatter_add(self, n, index, src):
        out = np.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype)
        np.add.at(out, index, src)
        return out

    def scatter_amax(self, n, index, src):
        """out[k] = max(0, max_{i: index[i] = k} src[i]) for nonnegative src."""
        out = np.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype)
        np.maximum.at(out, index, src)
        return out

    def shift2d(self, x, dy, dx):
        """y[..., i, j] = x[..., i + dy, j + dx], zero outside the lattice."""
        y = np.zeros_like(x)
        h, w = x.shape[-2:]
        ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
        xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
        y[..., yd, xd] = x[..., ys, xs]
        return y

    def detach(self, x):
        return x

    def to_numpy(self, x):
        return np.asarray(x)

    def from_numpy(self, a, like=None, dtype=None):
        return np.asarray(a, dtype=dtype or self.dtype)

    def is_array(self, x):
        return isinstance(x, np.ndarray)


class TorchBackend:
    name = "torch"

    def __init__(self, device=None, dtype=None):
        import torch
        self.torch = torch
        self.device = device
        self.dtype = dtype or torch.float32

    def _dev(self, like):
        return like.device if like is not None and hasattr(like, "device") else self.device

    def _dt(self, like):
        return like.dtype if like is not None and hasattr(like, "dtype") and like.is_floating_point() else self.dtype

    def asarray(self, x, like=None):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x
        return t.as_tensor(np.asarray(x), dtype=self._dt(like), device=self._dev(like))

    def zeros(self, shape, like=None):
        return self.torch.zeros(shape, dtype=self._dt(like), device=self._dev(like))

    def ones(self, shape, like=None):
        return self.torch.ones(shape, dtype=self._dt(like), device=self._dev(like))

    def full(self, shape, value, like=None):
        return self.torch.full(shape, float(value), dtype=self._dt(like), device=self._dev(like))

    def arange(self, n, like=None):
        return self.torch.arange(n, device=self._dev(like))

    def eye(self, n, like=None):
        return self.torch.eye(n, dtype=self._dt(like), device=self._dev(like))

    def index(self, x, like=None):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.long()
        return t.as_tensor(np.asarray(x, dtype=np.int64), device=self._dev(like))

    def exp(self, x):
        return self.torch.exp(x)

    def log(self, x):
        return self.torch.log(x)

    def sqrt(self, x):
        return self.torch.sqrt(x)

    def abs(self, x):
        return self.torch.abs(x)

    def maximum(self, a, b):
        t = self.torch
        return t.maximum(a, b if isinstance(b, t.Tensor) else t.as_tensor(b, dtype=a.dtype, device=a.device))

    def minimum(self, a, b):
        t = self.torch
        return t.minimum(a, b if isinstance(b, t.Tensor) else t.as_tensor(b, dtype=a.dtype, device=a.device))

    def where(self, c, a, b):
        t = self.torch
        ref = a if isinstance(a, t.Tensor) else b
        a = a if isinstance(a, t.Tensor) else t.as_tensor(a, dtype=ref.dtype, device=ref.device)
        b = b if isinstance(b, t.Tensor) else t.as_tensor(b, dtype=ref.dtype, device=ref.device)
        return t.where(c, a, b)

    def sigmoid(self, x):
        return self.torch.sigmoid(x)

    def softplus(self, x):
        return self.torch.nn.functional.softplus(x)

    def relu(self, x):
        return self.torch.relu(x)

    def clip(self, x, lo, hi):
        return self.torch.clamp(x, lo, hi)

    def round_half_even(self, x):
        return self.torch.round(x)          # torch.round rounds half to even

    def sum(self, x, axis=None, keepdims=False):
        return x.sum() if axis is None else x.sum(dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return x.mean() if axis is None else x.mean(dim=axis, keepdim=keepdims)

    def amax(self, x, axis=None, keepdims=False):
        return x.max() if axis is None else x.amax(dim=axis, keepdim=keepdims)

    def cumsum(self, x, axis):
        return self.torch.cumsum(x, dim=axis)

    def einsum(self, eq, *ops):
        return self.torch.einsum(eq, *ops)

    def stack(self, xs, axis=0):
        return self.torch.stack(list(xs), dim=axis)

    def concat(self, xs, axis=0):
        return self.torch.cat(list(xs), dim=axis)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def take(self, x, index, axis=0):
        return self.torch.index_select(x, axis, self.index(index, like=x))

    def scatter_add(self, n, index, src):
        out = self.torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
        return out.index_add(0, self.index(index, like=src), src)

    def scatter_amax(self, n, index, src):
        """out[k] = max(0, max_{i: index[i] = k} src[i]) for nonnegative src (no gradient use)."""
        out = self.torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
        return out.index_reduce(0, index, src, "amax", include_self=True)

    def shift2d(self, x, dy, dx):
        """y[..., i, j] = x[..., i + dy, j + dx], zero outside the lattice."""
        F = self.torch.nn.functional
        h, w = x.shape[-2:]
        p = F.pad(x, (max(0, -dx), max(0, dx), max(0, -dy), max(0, dy)))
        sy, sx = max(dy, 0), max(dx, 0)
        return p[..., sy:sy + h, sx:sx + w]

    def detach(self, x):
        return x.detach()

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def from_numpy(self, a, like=None, dtype=None):
        return self.torch.as_tensor(np.asarray(a), dtype=dtype or self._dt(like), device=self._dev(like))

    def is_array(self, x):
        return isinstance(x, self.torch.Tensor)


def backend_of(x) -> Any:
    """Pick the backend from an array."""
    if isinstance(x, np.ndarray):
        return NumpyBackend(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64)
    return TorchBackend(device=x.device, dtype=x.dtype if x.is_floating_point() else None)
