"""Guarded average linkage on the near window (SPLICE-V §2 Coarsen, §3, Lemma 2).

Scores are u8 codes q (value q/255).  A cluster pair's statistics are
(sum q, count, #q >= HI, #q >= VHI) over its existing window pairs:
  above(theta_T) :  2 * sum > 255 * count                  (mean > 0.5)
  guard          :  #hi >= 2  or  #vhi >= 1                 (two pairs > 0.5 or one > 0.8)
  eligible       :  above and guard;   guard-failed : above and not guard
Order <: mean descending (cross-multiplied), then pair key ascending, where
the pair key is (min, max) of the clusters' keys and a key is the largest
anchor index of the cluster (key_rule="max").  Keys are unique, so < is total.

sequential()  : merge the <-first eligible pair until none remains.
certified()   : Lemma 2 rounds -- each cluster names its <-first eligible and
                <-first guard-failed neighbour; a mutual pair is committed if it
                is the round's <-first eligible pair or no guard-failed pair
                incident to either cluster precedes it.  With the counted work
                budget (DD-11) the branch is a pure function of the frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import gen
from .arith import Counters

HI = gen.LINK_HI
VHI = gen.LINK_VHI
MIN_HI = gen.LINK_MIN_HI
DEN = gen.LINK_SCORE_DEN

# forward half-window offsets (dy, dx) of the P packet, 12 per P4 cell
LINK_FWD = tuple((dy, dx) for dy in range(0, 3) for dx in range(-2, 3) if dy > 0 or dx > 0)
assert len(LINK_FWD) == 12

Stats = List[int]  # [sum, count, n_hi, n_vhi]


def edge_stats(q: int) -> Stats:
    return [q, 1, 1 if q >= HI else 0, 1 if q >= VHI else 0]


def above(st: Stats) -> bool:
    return 2 * st[0] > DEN * st[1]


def guard(st: Stats) -> bool:
    return st[2] >= MIN_HI or st[3] >= 1


def eligible(st: Stats) -> bool:
    return above(st) and guard(st)


def guard_failed(st: Stats) -> bool:
    return above(st) and not guard(st)


def precedes(sa: Stats, ka: Tuple[int, int], sb: Stats, kb: Tuple[int, int]) -> bool:
    """(sa, ka) < (sb, kb): higher mean first, then smaller pair key."""
    lhs = sa[0] * sb[1]
    rhs = sb[0] * sa[1]
    if lhs != rhs:
        return lhs > rhs
    return ka < kb


@dataclass
class LinkResult:
    labels: Dict[int, int]                 # anchor -> cluster representative (active anchors)
    keys: Dict[int, int]                   # representative -> key
    rounds: int = 0
    work: int = 0
    budget_hit: bool = False
    overflow: bool = False
    counters: Counters = field(default_factory=Counters)


class _Graph:
    def __init__(self, n: int, edges: Iterable[Tuple[int, int, int]], active: Sequence[bool], key_rule: str):
        self.key_rule = key_rule
        self.nodes = [i for i in range(n) if active[i]]
        self.rep = {i: i for i in self.nodes}
        self.members = {i: [i] for i in self.nodes}
        self.key = {i: i for i in self.nodes}
        self.nbr: Dict[int, Dict[int, Stats]] = {i: {} for i in self.nodes}
        for a, b, q in edges:
            if a == b or not (active[a] and active[b]):
                continue
            if b in self.nbr[a]:
                raise ValueError(f"duplicate window pair ({a}, {b})")
            st = edge_stats(int(q))
            self.nbr[a][b] = st
            self.nbr[b][a] = list(st)

    def pair_key(self, a: int, b: int) -> Tuple[int, int]:
        ka, kb = self.key[a], self.key[b]
        return (ka, kb) if ka < kb else (kb, ka)

    def merge(self, a: int, b: int) -> int:
        """Merge clusters a and b; the survivor is the one whose key is kept."""
        if self.key_rule == "max":
            keep, drop = (a, b) if self.key[a] > self.key[b] else (b, a)
        else:
            keep, drop = (a, b) if self.key[a] < self.key[b] else (b, a)
        for c, st in self.nbr[drop].items():
            if c == keep:
                continue
            cur = self.nbr[keep].get(c)
            new = list(st) if cur is None else [x + y for x, y in zip(cur, st)]
            self.nbr[keep][c] = new
            self.nbr[c][keep] = list(new)
            del self.nbr[c][drop]
        self.nbr[keep].pop(drop, None)
        del self.nbr[drop]
        self.members[keep].extend(self.members.pop(drop))
        for m in self.members[keep]:
            self.rep[m] = keep
        del self.key[drop]
        return keep

    def best_eligible(self) -> Optional[Tuple[int, int]]:
        best = None
        for a, row in self.nbr.items():
            for b, st in row.items():
                if a < b and eligible(st):
                    k = self.pair_key(a, b)
                    if best is None or precedes(st, k, best[2], best[3]):
                        best = (a, b, st, k)
        return None if best is None else (best[0], best[1])


def sequential(n: int, edges, active, key_rule: str = "max") -> LinkResult:
    g = _Graph(n, edges, active, key_rule)
    while True:
        pair = g.best_eligible()
        if pair is None:
            break
        g.merge(*pair)
    return LinkResult(labels=dict(g.rep), keys=dict(g.key))


def _first(g: _Graph, a: int, pred) -> Optional[Tuple[int, Stats, Tuple[int, int]]]:
    best = None
    for b, st in g.nbr[a].items():
        if pred(st):
            k = g.pair_key(a, b)
            if best is None or precedes(st, k, best[1], best[2]):
                best = (b, st, k)
    return best


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def certified(n: int, edges, active, budget: Optional[int] = gen.LINK_BUDGET, key_rule: str = "max",
              plain_mnn: bool = False, nbmax: Optional[int] = gen.LINK_NBMAX) -> LinkResult:
    g = _Graph(n, edges, active, key_rule)
    res = LinkResult(labels={}, keys={})
    if nbmax is not None and any(len(r) > nbmax for r in g.nbr.values()):
        res.overflow = True
        res.counters.add("link_nb_overflow")
    while not res.overflow:
        firsts_e = {a: _first(g, a, eligible) for a in g.nbr}
        if all(v is None for v in firsts_e.values()):
            break
        if budget is not None and res.work >= budget:
            res.budget_hit = True
            res.counters.add("link_budget_exit")
            break
        res.rounds += 1
        firsts_g = {a: _first(g, a, guard_failed) for a in g.nbr}
        glob = None
        for a, fe in firsts_e.items():
            if fe is not None and (glob is None or precedes(fe[1], fe[2], glob[1], glob[2])):
                glob = (a, fe[1], fe[2])
        committed = []
        for a, fe in firsts_e.items():
            if fe is None:
                continue
            b = fe[0]
            if a >= b or firsts_e.get(b) is None or firsts_e[b][0] != a:
                continue
            st, k = fe[1], fe[2]
            ok = plain_mnn or (glob is not None and k == glob[2])
            if not ok:
                ok = True
                for c in (a, b):
                    fg = firsts_g.get(c)
                    if fg is not None and precedes(fg[1], fg[2], st, k):
                        ok = False
                        break
            if ok:
                committed.append((a, b))
        entries = sum(len(r) for r in g.nbr.values())
        merge_entries = sum(len(g.nbr[a]) + len(g.nbr[b]) for a, b in committed)
        res.work += (_ceil_div(entries, gen.LINK_SCAN_LANES) + gen.LINK_PHASE_FILL
                     + _ceil_div(len(g.nbr), gen.LINK_COMMIT_LANES) + gen.LINK_PHASE_FILL
                     + _ceil_div(merge_entries, gen.LINK_MERGE_UNITS) + gen.LINK_PHASE_FILL)
        for a, b in committed:
            g.merge(a, b)
        if nbmax is not None and any(len(r) > nbmax for r in g.nbr.values()):
            res.overflow = True
            res.counters.add("link_nb_overflow")
            break
    res.labels = dict(g.rep)
    res.keys = dict(g.key)
    res.counters.add("link_rounds", res.rounds)
    return res


def partition_of(res: LinkResult) -> frozenset:
    groups: Dict[int, List[int]] = {}
    for a, r in res.labels.items():
        groups.setdefault(r, []).append(a)
    return frozenset(frozenset(v) for v in groups.values())


def select_fragments(res: LinkResult, n: int, elig: Sequence[int], f_cap: int = gen.F_CAP,
                     cnt: Counters | None = None) -> Tuple[np.ndarray, int]:
    """Anchor fragment ids (u16, UNASSIGNED for dropped/inactive) and F.

    Clusters are ranked by eligibility mass (ties to the smaller key); the
    top F_cap become fragments, numbered in ascending key order.
    """
    groups: Dict[int, List[int]] = {}
    for a, r in res.labels.items():
        groups.setdefault(r, []).append(a)
    ranked = sorted(groups.items(), key=lambda kv: (-sum(int(elig[a]) for a in kv[1]), res.keys[kv[0]]))
    kept = ranked[:f_cap]
    if cnt is not None and len(ranked) > f_cap:
        cnt.add("fragment_cap_hit")
    kept.sort(key=lambda kv: res.keys[kv[0]])
    ids = np.full(n, gen.UNASSIGNED, dtype=np.int64)
    for fid, (_r, members) in enumerate(kept):
        ids[members] = fid
    return ids, len(kept)


def anchor_order(ids: np.ndarray) -> np.ndarray:
    """Compact form: anchors sorted by (fragment id, anchor index), unassigned last."""
    ids = np.asarray(ids, dtype=np.int64)
    return np.lexsort((np.arange(ids.size), ids))


def window_edges(scores_p4: np.ndarray, reserve_cells_p3: np.ndarray, reserve_scores: np.ndarray,
                 rr_partner: np.ndarray, rr_score: np.ndarray,
                 h4: int = gen.P4_H, w4: int = gen.P4_W, w3: int = gen.P3_W,
                 cnt: Counters | None = None) -> List[Tuple[int, int, int]]:
    """Edge list (a < b, q) of the near window from the P and H1 packets.

    Anchors 0 .. h4*w4-1 are the P4 cells in raster order; anchor h4*w4 + r is
    reserve r, placed at its P4 parent (P4-parent window convention).
    """
    n4 = h4 * w4
    edges = []
    sc = np.asarray(scores_p4).reshape(n4, 12)
    for i in range(n4):
        y, x = divmod(i, w4)
        for k, (dy, dx) in enumerate(LINK_FWD):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h4 and 0 <= xx < w4:
                edges.append((i, yy * w4 + xx, int(sc[i, k])))
    cells = np.asarray(reserve_cells_p3, dtype=np.int64)
    rs = np.asarray(reserve_scores).reshape(-1, 25)
    px, py = (cells % w3) >> 1, (cells // w3) >> 1
    for r in range(cells.size):
        for k in range(25):
            dy, dx = k // 5 - 2, k % 5 - 2
            yy, xx = py[r] + dy, px[r] + dx
            if 0 <= yy < h4 and 0 <= xx < w4:
                edges.append((int(yy * w4 + xx), n4 + r, int(rs[r, k])))
    rp = np.asarray(rr_partner).reshape(-1, 8)
    rq = np.asarray(rr_score).reshape(-1, 8)
    for r in range(cells.size):
        for s in range(8):
            p = int(rp[r, s])
            if p == 0xFFFF or p <= r:
                continue
            if p >= cells.size or max(abs(int(px[p] - px[r])), abs(int(py[p] - py[r]))) > gen.LINK_WINDOW:
                if cnt is not None:
                    cnt.add("rr_out_of_window")
                continue
            edges.append((n4 + r, n4 + p, int(rq[r, s])))
    return edges
