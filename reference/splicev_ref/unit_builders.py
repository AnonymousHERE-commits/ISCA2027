"""Build dispatcher commands (descriptor + payload words) from numpy arrays.

Used by the tests, by tools/make_vectors.py and by the PS runtime, so the
host, the reference and the vectors share one encoder.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from . import gen
from .units import Desc, command, pack_i64, pack_rows, pack_u8, pack_u16, pack_u32

F = gen


def _d(opcode: int, **kw) -> Desc:
    return Desc(opcode=opcode, **kw)


def gemm(A, B, bias=None, shift: int = 0, shifts=None, out: str = "8", sym: bool = False, **kw) -> List[int]:
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    a16_req = kw.pop("a16", False)
    a16 = a16_req or bool(np.any(A < -128) or np.any(A > 127))
    flags = (F.GEMM_F_A16 if a16 else 0) | {"8": 0, "16": F.GEMM_F_OUT16, "32": F.GEMM_F_OUT32}[out]
    flags |= F.GEMM_F_BIAS if bias is not None else 0
    flags |= F.GEMM_F_PERCH if shifts is not None else 0
    flags |= F.GEMM_F_SYM if sym else 0
    payload = pack_rows(B, "8")
    if bias is not None:
        payload += pack_u32(bias)
    if shifts is not None:
        payload += pack_u8(shifts)
    payload += pack_rows(A, "16" if a16 else "8")
    return command(_d(F.OP_GEMM, dim_m=m, dim_n=n, dim_k=k, param0=flags, param1=shift, **kw), payload)


def conv(X, Wt, k: int, stride: int = 1, bias=None, shift: int = 0, shifts=None, sym: bool = False,
         gamma=None, aff=None, **kw) -> List[int]:
    """X (H, W, Cin) int8, Wt (Cout, k, k, Cin) int8; aff (H, W, 8) u16 and gamma (Cin,) u16 enable modulation."""
    X = np.asarray(X, dtype=np.int64)
    Wt = np.asarray(Wt, dtype=np.int64)
    h, w, cin = X.shape
    cout = Wt.shape[0]
    mod = gamma is not None
    flags = (F.CONV_F_MOD if mod else 0) | (F.CONV_F_BIAS if bias is not None else 0) \
        | (F.CONV_F_PERCH if shifts is not None else 0) | (F.CONV_F_SYM if sym else 0)
    payload = pack_rows(Wt.reshape(cout, -1), "8")
    if bias is not None:
        payload += pack_u32(bias)
    if shifts is not None:
        payload += pack_u8(shifts)
    if mod:
        payload += pack_u16(gamma)
    for y in range(h):
        payload += pack_u8(X[y].ravel())
        if mod:
            payload += pack_u16(np.asarray(aff, dtype=np.int64)[y].ravel())
    return command(_d(F.OP_CONV, dim_m=h | (w << 16), dim_n=cout, dim_k=cin, param0=flags, param1=shift,
                      param2=k | (stride << 8), **kw), payload)


def stencil(taps, X, steps: int, halo: int, betas=None, exps: Sequence[int] = (0, 0, 0, 0, 0), in16: bool = False, **kw) -> List[int]:
    """taps (H, W, 9) int16, X (H, W, C) int8 (or int16 Q0.15 with in16), betas (H, W, 4) u16."""
    taps = np.asarray(taps, dtype=np.int64)
    X = np.asarray(X, dtype=np.int64)
    h, w, c = X.shape
    payload = []
    for t9 in taps.reshape(-1, 9):
        payload += pack_u16(t9)
    if betas is not None:
        for b4 in np.asarray(betas, dtype=np.int64).reshape(-1, 4):
            payload += pack_u16(b4)
    for ch in range(c):
        plane = X[..., ch].ravel()
        payload += pack_u16(plane) if in16 else pack_u8(plane)
    p1 = sum((int(e) & 0xF) << (4 * j) for j, e in enumerate(exps))
    flags = (F.STENCIL_F_BANDS if betas is not None else 0) | (F.STENCIL_F_IN16 if in16 else 0)
    return command(_d(F.OP_STENCIL, dim_m=h | (w << 16), dim_n=c, dim_k=steps, param0=flags, param1=p1, param2=halo, **kw), payload)


def lut_table8(table, x, **kw):
    return command(_d(F.OP_LUT, dim_m=len(x), param0=F.LUT_TABLE8, **kw), pack_u8(table) + pack_u8(x))


def lut_sigmoid(x, **kw):
    return command(_d(F.OP_LUT, dim_m=len(x), param0=F.LUT_SIGMOID, **kw), pack_u16(x))


def lut_expf(x, **kw):
    return command(_d(F.OP_LUT, dim_m=len(x), param0=F.LUT_EXPF, **kw), pack_u16(x))


def requant32(parts, shift: int, sym: bool = False, out16: bool = False, **kw):
    parts = np.asarray(parts, dtype=np.int64)
    payload = []
    for p in parts:
        payload += pack_u32(p)
    return command(_d(F.OP_LUT, dim_m=parts.shape[1], dim_k=parts.shape[0], param0=F.LUT_REQUANT32, param1=shift,
                      param2=(1 if sym else 0) | (2 if out16 else 0), **kw), payload)


def permute(idx, rows_words, **kw):
    rows_words = [list(r) for r in rows_words]
    rw = len(rows_words[0])
    payload = pack_u16(idx)
    for r in rows_words:
        payload += r
    return command(_d(F.OP_PERMUTE, dim_m=len(idx), dim_n=rw, dim_k=len(rows_words), **kw), payload)


def segsum(ids, weights, X, n_frag: int, **kw):
    X = np.asarray(X, dtype=np.int64)
    payload = pack_u16(ids) + pack_u8(weights) + pack_rows(X, "8")
    return command(_d(F.OP_SEGOPS, dim_m=X.shape[0], dim_n=X.shape[1], dim_k=n_frag, param0=F.SEGOPS_SEGSUM, **kw), payload)


def segmean(sums, wsum, **kw):
    sums = np.asarray(sums, dtype=np.int64)
    payload = pack_u32(wsum) + pack_rows(sums, "32")
    return command(_d(F.OP_SEGOPS, dim_m=1, dim_n=sums.shape[1], dim_k=sums.shape[0], param0=F.SEGOPS_SEGMEAN, **kw), payload)


def fill(H, ids, **kw):
    H = np.asarray(H, dtype=np.int64)
    payload = pack_rows(H, "8") + pack_u16(ids)
    return command(_d(F.OP_SEGOPS, dim_m=len(ids), dim_n=H.shape[1], dim_k=H.shape[0], param0=F.SEGOPS_FILL, **kw), payload)


def detail(H, ids, X, **kw):
    H = np.asarray(H, dtype=np.int64)
    payload = pack_rows(H, "8") + pack_u16(ids) + pack_rows(np.asarray(X, dtype=np.int64), "8")
    return command(_d(F.OP_SEGOPS, dim_m=len(ids), dim_n=H.shape[1], dim_k=H.shape[0], param0=F.SEGOPS_DETAIL, **kw), payload)


def synth(dH, ids, X, R=None, **kw):
    dH = np.asarray(dH, dtype=np.int64)
    X = np.asarray(X, dtype=np.int64)
    payload = pack_rows(dH, "8") + pack_u16(ids)
    for i in range(X.shape[0]):
        payload += pack_u8(X[i])
        if R is not None:
            payload += pack_u8(np.asarray(R, dtype=np.int64)[i])
    return command(_d(F.OP_SEGOPS, dim_m=X.shape[0], dim_n=dH.shape[1], dim_k=dH.shape[0], param0=F.SEGOPS_SYNTH,
                      param1=1 if R is not None else 0, **kw), payload)


def clsband(PiN, RZ, ids, **kw):
    PiN = np.asarray(PiN, dtype=np.int64)
    payload = pack_rows(PiN, "8") + pack_rows(np.asarray(RZ, dtype=np.int64), "8") + pack_u16(ids)
    return command(_d(F.OP_SEGOPS, dim_m=len(ids), dim_n=PiN.shape[1], dim_k=PiN.shape[0], param0=F.SEGOPS_CLSBAND, **kw), payload)


def modulate(gamma, aff, X, **kw):
    X = np.asarray(X, dtype=np.int64)
    payload = pack_u16(gamma) + pack_u16(aff) + pack_rows(X, "8")
    return command(_d(F.OP_TAPMOD, dim_m=X.shape[0], dim_n=X.shape[1], param0=F.TAPMOD_MODULATE, **kw), payload)


def energy(ids, planes, n_frag: int, **kw):
    """planes (J, n, C) int8."""
    planes = np.asarray(planes, dtype=np.int64)
    j, n, c = planes.shape
    payload = pack_u16(ids)
    for jj in range(j):
        payload += pack_rows(planes[jj], "8")
    return command(_d(F.OP_TAPMOD, dim_m=n, dim_n=c, dim_k=n_frag, param0=F.TAPMOD_ENERGY, param1=j, **kw), payload)


def nu(sums, counts, shift: int = gen.NU_ENERGY_SHIFT, **kw):
    """sums (F, J) int64, counts (F, J) int32."""
    sums = np.asarray(sums, dtype=object)
    counts = np.asarray(counts, dtype=np.int64)
    nf, j = counts.shape
    payload = []
    for f in range(nf):
        payload += pack_i64(sums[f]) + pack_u32(counts[f])
    return command(_d(F.OP_TAPMOD, dim_n=j, dim_k=nf, param0=F.TAPMOD_NU, param1=shift, **kw), payload)


def mix(gains, planes, shift: int, **kw):
    """planes (n, J, C) int8 interleaved per cell."""
    planes = np.asarray(planes, dtype=np.int64)
    n, j, c = planes.shape
    payload = pack_u16(gains)
    for i in range(n):
        payload += pack_rows(planes[i], "8")
    return command(_d(F.OP_TAPMOD, dim_m=n, dim_n=c, dim_k=j, param0=F.TAPMOD_MIX, param1=shift, **kw), payload)


def forecast(K, Uv, VO, mask_rows, Hs_rows, null_raw_rows, alpha_raw_rows, row0: int, **kw):
    """Resident columns K/Uv/VO for all F fragments; per-row data for receivers row0..row0+R-1.

    mask_rows (R, F_cap/32) uint32 words of the shared mask G.
    """
    K = np.asarray(K, dtype=np.int64)
    Uv = np.asarray(Uv, dtype=np.int64)
    VO = np.asarray(VO, dtype=np.int64)
    nf = K.shape[0]
    rows = len(Hs_rows)
    payload = pack_rows(K, "8") + pack_rows(Uv, "8") + pack_rows(VO, "8")
    for r in range(rows):
        payload += [int(x) & 0xFFFFFFFF for x in mask_rows[r]]
        payload += pack_u8(Hs_rows[r])
        payload += pack_u32([null_raw_rows[r], alpha_raw_rows[r]])
    return command(_d(F.OP_HEAD, dim_m=rows, dim_n=Uv.shape[1], dim_k=nf, param0=F.HEAD_FORECAST, param1=row0,
                      param2=VO.shape[1], **kw), payload)


def score(logits, gates, tau: int = gen.TAU_DET_Q15, **kw):
    return command(_d(F.OP_HEAD, dim_m=len(logits), param0=F.HEAD_SCORE, param1=tau, **kw),
                   pack_u16(logits) + pack_u16(gates))


def nop(**kw):
    return Desc(opcode=F.OP_NOP, **kw).words()
