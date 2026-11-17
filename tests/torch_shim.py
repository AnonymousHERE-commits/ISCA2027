"""Forward-only numpy stand-in for the subset of PyTorch that detector/splice_det/neck.py,
backend.TorchBackend, losses.py and qat.NeckQuant use.

It lets the test suite run the SPLICE neck's forward pass (with the reserve, linkage and
exchange steps of the CPU pool), its quantization sites and its auxiliary losses on machines
without torch.  There is no autograd: training needs real PyTorch, and the tests use real
torch whenever it is installed (`load()` returns it).  Semantics follow torch where the neck
depends on them: float32 defaults, int64 indices, `dim`/`keepdim` reductions, round half to
even, zero padding, nearest interpolation, batch statistics in BatchNorm training mode.
"""
from __future__ import annotations

import contextlib
import math
import sys
import types
from typing import Iterator, Tuple

import numpy as np

_RNG = [np.random.default_rng(0)]


class _Device:
    type = "cpu"

    def __repr__(self) -> str:
        return "device(type='cpu')"

    def __eq__(self, other) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


_CPU = _Device()


def _t(x) -> "Tensor":
    return np.asarray(x).view(Tensor)


def _dtype(d, default=None):
    if d is None:
        return default
    return np.dtype(d)


class Tensor(np.ndarray):
    """numpy array with the torch methods the neck calls."""

    __array_priority__ = 1.0                         # results of mixed Tensor/Parameter ops are Tensors

    def __new__(cls, data):
        return np.asarray(data).view(cls)

    def __array_finalize__(self, obj):
        pass

    # reductions (torch names, 0-d results stay tensors)
    def _red(self, fn, dim, keepdim, axis=None, keepdims=None, **kw):
        ax = dim if dim is not None else axis
        kd = bool(keepdim) or bool(keepdims)
        return _t(fn(self.view(np.ndarray), axis=ax, keepdims=kd, **{k: v for k, v in kw.items() if k == "dtype" and v is not None}))

    def sum(self, dim=None, keepdim=False, axis=None, dtype=None, out=None, keepdims=None, **kw):
        return self._red(np.sum, dim, keepdim, axis, keepdims, dtype=dtype)

    def mean(self, dim=None, keepdim=False, axis=None, dtype=None, out=None, keepdims=None, **kw):
        return self._red(np.mean, dim, keepdim, axis, keepdims, dtype=dtype)

    def amax(self, dim=None, keepdim=False):
        return self._red(np.max, dim, keepdim)

    def amin(self, dim=None, keepdim=False):
        return self._red(np.min, dim, keepdim)

    def max(self, dim=None, keepdim=False, axis=None, out=None, keepdims=None, **kw):
        if dim is None and axis is None:
            return _t(np.max(self.view(np.ndarray)))
        ax = dim if dim is not None else axis
        a = self.view(np.ndarray)
        return _t(np.max(a, axis=ax, keepdims=keepdim)), _t(np.argmax(a, axis=ax))

    def min(self, dim=None, keepdim=False, axis=None, out=None, keepdims=None, **kw):
        if dim is None and axis is None:
            return _t(np.min(self.view(np.ndarray)))
        ax = dim if dim is not None else axis
        a = self.view(np.ndarray)
        return _t(np.min(a, axis=ax, keepdims=keepdim)), _t(np.argmin(a, axis=ax))

    # shape
    def flatten(self, start_dim=0, end_dim=-1):
        nd = self.ndim
        if nd == 0:
            return self.reshape(1)
        s, e = start_dim % nd, end_dim % nd
        return self.reshape(self.shape[:s] + (-1,) + self.shape[e + 1:])

    def unsqueeze(self, d):
        return _t(np.expand_dims(self.view(np.ndarray), d))

    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        return self.transpose(dims)

    def contiguous(self):
        return self

    def numel(self) -> int:
        return int(self.size)

    def dim(self) -> int:
        return self.ndim

    # elementwise
    def clamp(self, min=None, max=None):
        a = self.view(np.ndarray)
        if min is not None:
            a = np.maximum(a, np.asarray(min, dtype=a.dtype))
        if max is not None:
            a = np.minimum(a, np.asarray(max, dtype=a.dtype))
        return _t(a)

    def abs(self):
        return _t(np.abs(self.view(np.ndarray)))

    def exp(self):
        return _t(np.exp(self.view(np.ndarray)))

    def log(self):
        return _t(np.log(self.view(np.ndarray)))

    def sqrt(self):
        return _t(np.sqrt(self.view(np.ndarray)))

    def sigmoid(self):
        return sigmoid(self)

    # dtype / device
    def float(self):
        return self.astype(np.float32)

    def double(self):
        return self.astype(np.float64)

    def long(self):
        return self.astype(np.int64)

    def int(self):
        return self.astype(np.int32)

    def bool(self):
        return self.astype(np.bool_)

    def is_floating_point(self) -> bool:
        return bool(np.issubdtype(self.dtype, np.floating))

    @property
    def device(self):
        return _CPU

    def cpu(self):
        return self

    def cuda(self):
        return self

    def to(self, *args, dtype=None, device=None, non_blocking=False, **kw):
        for a in args:
            if isinstance(a, (np.dtype, type)) and not isinstance(a, _Device):
                try:
                    dtype = np.dtype(a)
                except TypeError:
                    pass
        return self.astype(dtype) if dtype is not None else self

    def detach(self):
        return self

    def numpy(self):
        return self.view(np.ndarray)

    def requires_grad_(self, flag=True):
        return self

    # construction / in-place
    def new_zeros(self, shape, dtype=None, device=None):
        return _t(np.zeros(shape, dtype=dtype or self.dtype))

    def new_ones(self, shape, dtype=None, device=None):
        return _t(np.ones(shape, dtype=dtype or self.dtype))

    def fill_(self, v):
        self[...] = v
        return self

    def zero_(self):
        self[...] = 0
        return self

    def copy_(self, src):
        self[...] = np.asarray(src)
        return self

    def index_reduce(self, dim, index, src, reduce, include_self=True):
        if dim != 0 or reduce != "amax":
            raise NotImplementedError("index_reduce: dim 0, amax only")
        out = np.array(self.view(np.ndarray))
        np.maximum.at(out, np.asarray(index, dtype=np.int64), np.asarray(src, dtype=out.dtype))
        return _t(out)

    def tanh(self):
        return _t(np.tanh(self.view(np.ndarray)))

    def expand_as(self, other):
        return _t(np.broadcast_to(self.view(np.ndarray), other.shape))

    def index_add(self, dim, index, src):
        if dim != 0:
            raise NotImplementedError("index_add: dim 0 only")
        out = np.array(self.view(np.ndarray))
        np.add.at(out, np.asarray(index, dtype=np.int64), np.asarray(src, dtype=out.dtype))
        return _t(out)


class Parameter(Tensor):
    __array_priority__ = 0.5

    def __new__(cls, data=None, requires_grad=True):
        a = np.array(np.asarray(data), dtype=np.asarray(data).dtype)
        return a.view(cls)


# ----------------------------------------------------------------------------- functions

float32, float64, float16, bfloat16 = np.dtype("float32"), np.dtype("float64"), np.dtype("float16"), np.dtype("float32")
int64, int32, uint8, bool_ = np.dtype("int64"), np.dtype("int32"), np.dtype("uint8"), np.dtype("bool")
long = int64


def manual_seed(seed: int):
    _RNG[0] = np.random.default_rng(seed)


def is_tensor(x) -> bool:
    return isinstance(x, Tensor)


def as_tensor(data, dtype=None, device=None):
    if isinstance(data, Tensor):
        return data.astype(dtype) if dtype is not None else data
    a = np.asarray(data)
    if dtype is not None:
        return _t(a.astype(dtype))
    if not isinstance(data, np.ndarray) and a.dtype == np.float64:
        a = a.astype(np.float32)                     # python floats -> torch default dtype
    return _t(a)


def tensor(data, dtype=None, device=None):
    return as_tensor(np.array(data), dtype=dtype) if isinstance(data, np.ndarray) else as_tensor(data, dtype)


def _shape(args):
    return tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)


def zeros(*shape, dtype=None, device=None):
    return _t(np.zeros(_shape(shape), dtype=_dtype(dtype, float32)))


def ones(*shape, dtype=None, device=None):
    return _t(np.ones(_shape(shape), dtype=_dtype(dtype, float32)))


def full(shape, value, dtype=None, device=None):
    d = _dtype(dtype, float32 if isinstance(value, float) else int64 if isinstance(value, (int, np.integer)) else float32)
    return _t(np.full(tuple(shape), value, dtype=d))


def zeros_like(x, dtype=None):
    return _t(np.zeros(x.shape, dtype=dtype or x.dtype))


def ones_like(x, dtype=None):
    return _t(np.ones(x.shape, dtype=dtype or x.dtype))


def full_like(x, value, dtype=None):
    return _t(np.full(x.shape, value, dtype=dtype or x.dtype))


def arange(*args, dtype=None, device=None):
    a = np.arange(*args)
    return _t(a.astype(dtype) if dtype is not None else a.astype(np.int64) if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32))


def eye(n, dtype=None, device=None):
    return _t(np.eye(n, dtype=_dtype(dtype, float32)))


def randn(*shape, dtype=None, device=None, generator=None):
    return _t(_RNG[0].standard_normal(_shape(shape)).astype(_dtype(dtype, float32)))


def rand(*shape, dtype=None, device=None):
    return _t(_RNG[0].random(_shape(shape)).astype(_dtype(dtype, float32)))


def randint(low, high=None, size=(), device=None):
    if high is None:
        low, high = 0, low
    return _t(_RNG[0].integers(low, high, size=size).astype(np.int64))


def where(cond, a, b):
    return _t(np.where(np.asarray(cond), a, b))


def stack(xs, dim=0):
    return _t(np.stack([np.asarray(x) for x in xs], axis=dim))


def cat(xs, dim=0):
    return _t(np.concatenate([np.asarray(x) for x in xs], axis=dim))


def sigmoid(x):
    a = np.asarray(x)
    with np.errstate(over="ignore"):
        return _t((1.0 / (1.0 + np.exp(-a))).astype(a.dtype))


def exp(x):
    return _t(np.exp(np.asarray(x)))


def log(x):
    return _t(np.log(np.asarray(x)))


def sqrt(x):
    return _t(np.sqrt(np.asarray(x)))


def abs(x):  # noqa: A001 - torch name
    return _t(np.abs(np.asarray(x)))


def tanh(x):
    return _t(np.tanh(np.asarray(x)))


def relu(x):
    return _t(np.maximum(np.asarray(x), 0))


def round(x):  # noqa: A001 - torch name (half to even, as torch.round)
    return _t(np.round(np.asarray(x)))


def minimum(a, b):
    return _t(np.minimum(np.asarray(a), np.asarray(b)))


def maximum(a, b):
    return _t(np.maximum(np.asarray(a), np.asarray(b)))


def clamp(x, min=None, max=None):  # noqa: A002
    return _t(x).clamp(min, max)


def cumsum(x, dim):
    return _t(np.cumsum(np.asarray(x), axis=dim))


def einsum(eq, *ops):
    return _t(np.einsum(eq, *[np.asarray(o) for o in ops]))


def index_select(x, dim, index):
    return _t(np.take(np.asarray(x), np.asarray(index, dtype=np.int64), axis=dim))


def div(a, b, rounding_mode=None):
    if rounding_mode == "floor":
        return _t(np.floor_divide(np.asarray(a), b))
    if rounding_mode == "trunc":
        return _t(np.trunc(np.asarray(a) / b).astype(np.asarray(a).dtype))
    return _t(np.asarray(a) / b)


def quantile(x, q):
    return _t(np.quantile(np.asarray(x), q))


def ceil(x):
    return _t(np.ceil(np.asarray(x)))


def floor(x):
    return _t(np.floor(np.asarray(x)))


def _linalg_solve(A, b):
    return _t(np.linalg.solve(np.asarray(A), np.asarray(b)))


def no_grad():
    return contextlib.nullcontext()


def autocast(*a, **k):
    return contextlib.nullcontext()


# ----------------------------------------------------------------------------- nn

class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        d = self.__dict__
        if isinstance(value, Parameter):
            d["_parameters"][name] = value
        elif isinstance(value, Module):
            d["_modules"][name] = value
        elif name in d.get("_buffers", {}):
            d["_buffers"][name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        d = self.__dict__
        for k in ("_parameters", "_buffers", "_modules"):
            if k in d and name in d[k]:
                return d[k][name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")

    def register_buffer(self, name, value, persistent=True):
        self.__dict__["_buffers"][name] = value

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix, self
        for n, m in self._modules.items():
            yield from m.named_modules(f"{prefix}.{n}" if prefix else n)

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def children(self):
        return iter(self._modules.values())

    def named_parameters(self, prefix: str = ""):
        for mp, m in self.named_modules(prefix):
            for n, p in m._parameters.items():
                yield (f"{mp}.{n}" if mp else n), p

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_buffers(self, prefix: str = ""):
        for mp, m in self.named_modules(prefix):
            for n, b in m._buffers.items():
                yield (f"{mp}.{n}" if mp else n), b

    def train(self, mode: bool = True):
        for m in self.modules():
            object.__setattr__(m, "training", mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *a, **k):
        return self

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Sequential(Module):
    def __init__(self, *mods):
        super().__init__()
        for i, m in enumerate(mods):
            self._modules[str(i)] = m

    def __getitem__(self, i):
        return list(self._modules.values())[i]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class ModuleList(Module):
    def __init__(self, mods=()):
        super().__init__()
        for i, m in enumerate(mods):
            self._modules[str(i)] = m

    def __getitem__(self, i):
        return list(self._modules.values())[i]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())


def _uniform(shape, bound):
    return Parameter(_RNG[0].uniform(-bound, bound, shape).astype(np.float32))


class Conv2d(Module):
    def __init__(self, cin, cout, k, s=1, p=0, groups=1, bias=True):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("groups != 1")
        k = k if isinstance(k, int) else k[0]
        self.kernel_size, self.stride, self.padding, self.groups = (k, k), (s, s), (p, p), groups
        bound = 1.0 / math.sqrt(cin * k * k)
        self.weight = _uniform((cout, cin, k, k), bound)
        self.bias = _uniform((cout,), bound) if bias else None

    def forward(self, x):
        a = np.asarray(x)
        k, s, p = self.kernel_size[0], self.stride[0], self.padding[0]
        if p:
            a = np.pad(a, ((0, 0), (0, 0), (p, p), (p, p)))
        n, c, h, w = a.shape
        ho, wo = (h - k) // s + 1, (w - k) // s + 1
        out = 0.0
        W = np.asarray(self.weight)
        for ky in range(k):
            for kx in range(k):
                sl = a[:, :, ky:ky + s * (ho - 1) + 1:s, kx:kx + s * (wo - 1) + 1:s]
                out = out + np.einsum("oc,nchw->nohw", W[:, :, ky, kx], sl)
        if self.bias is not None:
            out = out + np.asarray(self.bias)[None, :, None, None]
        return _t(np.asarray(out, dtype=np.float32))


class BatchNorm2d(Module):
    def __init__(self, c, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(c, dtype=np.float32))
        self.bias = Parameter(np.zeros(c, dtype=np.float32))
        self.register_buffer("running_mean", _t(np.zeros(c, dtype=np.float32)))
        self.register_buffer("running_var", _t(np.ones(c, dtype=np.float32)))

    def forward(self, x):
        a = np.asarray(x)
        if self.training:
            mu, var = a.mean(axis=(0, 2, 3)), a.var(axis=(0, 2, 3))
        else:
            mu, var = np.asarray(self.running_mean), np.asarray(self.running_var)
        y = (a - mu[None, :, None, None]) / np.sqrt(var[None, :, None, None] + self.eps)
        return _t((y * np.asarray(self.weight)[None, :, None, None] + np.asarray(self.bias)[None, :, None, None]).astype(np.float32))


class SiLU(Module):
    def forward(self, x):
        return _t(x) * sigmoid(x)


class Linear(Module):
    def __init__(self, cin, cout, bias=True):
        super().__init__()
        bound = 1.0 / math.sqrt(cin)
        self.in_features, self.out_features = cin, cout
        self.weight = _uniform((cout, cin), bound)
        self.bias = _uniform((cout,), bound) if bias else None

    def forward(self, x):
        y = np.asarray(x) @ np.asarray(self.weight).T
        if self.bias is not None:
            y = y + np.asarray(self.bias)
        return _t(y.astype(np.float32))


def _zeros_(t):
    t[...] = 0
    return t


# ----------------------------------------------------------------------------- functional

def _softplus(x, beta=1.0, threshold=20.0):
    a = np.asarray(x)
    return _t(np.logaddexp(0.0, a).astype(a.dtype))


def _pad(x, pad, mode="constant", value=0.0):
    a = np.asarray(x)
    widths = [(0, 0)] * a.ndim
    for i in range(len(pad) // 2):
        widths[a.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    return _t(np.pad(a, widths, constant_values=value))


def _interpolate(x, size=None, scale_factor=None, mode="nearest"):
    if mode != "nearest":
        raise NotImplementedError(mode)
    a = np.asarray(x)
    h, w = a.shape[-2:]
    ho, wo = (size if isinstance(size, (tuple, list)) else (size, size)) if size is not None else \
        (int(h * scale_factor), int(w * scale_factor))
    iy = np.floor(np.arange(ho) * (h / ho)).astype(np.int64)
    ix = np.floor(np.arange(wo) * (w / wo)).astype(np.int64)
    return _t(a[..., iy[:, None], ix[None, :]])


def _bce_logits(logit, target, weight=None, reduction="mean"):
    x, t = np.asarray(logit), np.asarray(target)
    loss = np.maximum(x, 0) - x * t + np.log1p(np.exp(-np.abs(x)))
    if reduction == "none":
        return _t(loss)
    return _t(loss.mean() if reduction == "mean" else loss.sum())


def _silu(x):
    return _t(x) * sigmoid(x)


# ----------------------------------------------------------------------------- module assembly

def build() -> types.ModuleType:
    """A `torch` module object made of this file's definitions."""
    torch = types.ModuleType("torch")
    me = sys.modules[__name__]
    for name in ("Tensor", "float32", "float64", "float16", "bfloat16", "int64", "int32", "uint8", "long",
                 "manual_seed", "is_tensor", "as_tensor", "tensor", "zeros", "ones", "full", "zeros_like", "ones_like",
                 "full_like", "arange", "eye", "randn", "rand", "randint", "where", "stack", "cat", "sigmoid", "exp",
                 "log", "sqrt", "abs", "tanh", "relu", "round", "minimum", "maximum", "clamp", "cumsum", "einsum",
                 "index_select", "div", "quantile", "ceil", "floor", "no_grad", "autocast"):
        setattr(torch, name, getattr(me, name))
    torch.bool = bool_
    torch.float = float32
    torch.linalg = types.SimpleNamespace(solve=_linalg_solve)
    torch.device = lambda *a, **k: _CPU
    torch.__version__ = "0.0-numpy-shim"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda s: None)
    nn = types.ModuleType("torch.nn")
    for name in ("Module", "Sequential", "ModuleList", "Conv2d", "BatchNorm2d", "SiLU", "Linear", "Parameter"):
        setattr(nn, name, getattr(me, name))
    nn.init = types.SimpleNamespace(zeros_=_zeros_)
    fn = types.ModuleType("torch.nn.functional")
    fn.softplus, fn.pad, fn.interpolate, fn.binary_cross_entropy_with_logits, fn.silu = \
        _softplus, _pad, _interpolate, _bce_logits, _silu
    fn.sigmoid = sigmoid
    utils = types.ModuleType("torch.nn.utils")
    parametrize = types.ModuleType("torch.nn.utils.parametrize")
    parametrize.is_parametrized = lambda m, name=None: False
    utils.parametrize = parametrize
    utils.clip_grad_norm_ = lambda params, max_norm: 0.0
    nn.functional, nn.utils = fn, utils
    torch.nn = nn
    torch._splice_shim = True
    return torch


def load():
    """Real torch when it is installed; otherwise this shim, registered in sys.modules."""
    try:
        import torch  # noqa: F401
        if not getattr(torch, "_splice_shim", False):
            return torch
    except ImportError:
        pass
    torch = build()
    sys.modules.update({"torch": torch, "torch.nn": torch.nn, "torch.nn.functional": torch.nn.functional,
                        "torch.nn.utils": torch.nn.utils, "torch.nn.utils.parametrize": torch.nn.utils.parametrize})
    return torch
