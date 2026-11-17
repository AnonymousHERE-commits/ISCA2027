"""Deterministic routing of dispatcher units onto worker classes (shared-worker model).

A worker is described by its body mask (SV_BODY_MASK of its AIE kernel: bit i enables
opcode i).  route() assigns every command to a worker whose class implements its opcode,
balancing an estimated load (input + output words); NOP and malformed descriptors go to
the least-loaded worker.  The same function routes the PS submissions, the simulator data
(tools/aie_sim_data.py) and the expected responses, so all three agree.
"""
from __future__ import annotations

from typing import List, Sequence

from . import gen
from .arith import check_word
from .units import decode_fields

FULL_MASK = 0xFF


def opcode_of(cmd: Sequence[int]) -> int:
    return decode_fields(cmd[:gen.DESC_WORDS], gen.DESC_FIELDS)["opcode"]


def route(cmds: Sequence[Sequence[int]], masks: Sequence[int]) -> List[int]:
    """Worker index for each command."""
    if not masks:
        raise ValueError("at least one worker is required")
    load = [0] * len(masks)
    out = []
    for c in cmds:
        hdr = [int(x) & 0xFFFFFFFF for x in c[:gen.DESC_WORDS]]
        d = decode_fields(hdr, gen.DESC_FIELDS)
        op = d["opcode"]
        framed = d["magic"] == gen.CMD_MAGIC and d["check"] == check_word(hdr[:gen.DESC_WORDS - 1]) \
            and d["version"] == gen.PROTO_VERSION
        if framed and op <= 7:
            cand = [w for w, m in enumerate(masks) if (int(m) >> op) & 1]
            if not cand:
                raise ValueError(f"no worker class implements opcode {op}; masks {[hex(int(m)) for m in masks]}")
        else:
            cand = list(range(len(masks)))
        w = min(cand, key=lambda i: (load[i], i))
        load[w] += len(c) + (int(d["out_words"]) if framed else 0) + gen.RSP_WORDS + gen.TRL_WORDS
        out.append(w)
    return out


def group(cmds: Sequence[Sequence[int]], masks: Sequence[int]) -> List[List[int]]:
    """Command indices per worker, in submission order."""
    r = route(cmds, masks)
    per: List[List[int]] = [[] for _ in masks]
    for i, w in enumerate(r):
        per[w].append(i)
    return per
