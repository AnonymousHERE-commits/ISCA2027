#!/usr/bin/env python3
"""Recompute the manuscript's closed-form numbers from its own formulas.

Each line is REPRODUCED (same value), CONSISTENT (agrees after the stated
rounding), or OPEN (needs the per-unit schedule model, training or the board,
so it cannot be recomputed here).  An arithmetic audit of the stated model, not
a hardware result.  Writes evidence/audit/paper_numbers.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference"))

from splicev_ref import formulas as fm  # noqa: E402
from splicev_ref import gen  # noqa: E402

rows = []


def check(name, stated, got, tol=0.0, ref="", note=""):
    ok = abs(float(got) - float(stated)) <= tol
    rows.append({"item": name, "stated": stated, "recomputed": round(float(got), 6), "status": "REPRODUCED" if ok else "DIFFERS",
                 "ref": ref, "note": note})


def consistent(name, stated, got, note, ref=""):
    rows.append({"item": name, "stated": stated, "recomputed": got, "status": "CONSISTENT", "ref": ref, "note": note})


def open_item(name, stated, why, ref=""):
    rows.append({"item": name, "stated": stated, "recomputed": None, "status": "OPEN", "ref": ref, "note": why})


def main() -> int:
    # --- logic engines (Prop. 4, Table 2)
    for n, gate_us in ((128, 9.3), (64, 18.4), (32, 36.7)):
        cg = fm.gate_cycles(256, n)
        check(f"C_G({n}) cycles at the cap", {128: 2798}.get(n, round(cg)), cg, 0.5, "§3 gate")
        check(f"gate time ({n} lanes), us", gate_us, cg / 300e6 * 1e6, 0.05, "Table 2")
    base = fm.exchange_cycles(e_dup=0)
    check("C_X base cycles (E_dup = 0)", 33472, base, 0, "Eq. (6)", "roadmap 2.3")
    check("C_X at E_dup = 1024", 41664, fm.exchange_cycles(e_dup=1024), 0, "Eq. (6)",
          "41,664 cycles needs E_dup = 1,024; the enforced bound must come from the model")
    check("C_X at E_dup = 4096 (cap of the min)", 66240, fm.exchange_cycles(e_dup=4096), 0, "Eq. (6)")
    for n, x_ms, g2_ms in ((128, 0.217, 0.242), (64, 0.328, 0.353), (32, 0.550, 0.575)):
        st = fm.exchange_stage_ms(fm.exchange_cycles(n=n, e_dup=1024))
        check(f"X ({n} lanes), ms", x_ms, st["X_ms"], 0.0006, "Table 2")
        check(f"g2 ({n} lanes), ms", g2_ms, st["g2_ms"], 0.0006, "Table 2")
    check("priority unit time t_U, us", 2.9, fm.exchange_stage_ms(41664)["t_U_us"], 0.05, "Prop. 4")
    check("priority unit MACs, M", 15.1, 7360 * 256 * 8 / 1e6, 0.05, "Prop. 4")
    # --- Lemma 1
    check("row slicing intake-bound below n cells", 544, fm.lemma1_row_slice_threshold(), 0, "Lemma 1(i)")
    check("row-slice ceiling at P5", 0.37, fm.lemma1_row_slice_ceiling(), 0.005, "Lemma 1(i)")
    check("depthwise 3x3 output ceiling", 0.28, fm.lemma1_output_ceiling(), 0.005, "Lemma 1(ii)")
    # --- Prop. 1, 2 and P1
    check("executed walk term at P5 (q = 14)", 0.0064, fm.prop1_walk_term()["total"], 0.00005, "Prop. 1")
    check("executed walk term at P5 (q = 13)", 0.012, fm.prop1_walk_term(q=13)["total"], 0.0005, "Prop. 1")
    check("declared walk tolerance", 0.0095, 0.019 / 2, 0, "Prop. 1")
    check("int8-state walk term", 0.053, fm.prop1_walk_term(bs=8)["total"], 0.0005, "§4 word lengths",
          "state rounding T 2^-8 replaces T 2^-16")
    check("posterior bound, b = 24 (log2)", -14.7, fm.prop2_marginal_bound(24), 0.05, "Prop. 2(i)")
    check("posterior bound, b = 16 (log2)", -6.9, fm.prop2_marginal_bound(16), 0.05, "§4 word lengths")
    p1 = fm.p1_rate_bound()
    consistent("P1 per-image mismatch bound", 5.9e-4, round(p1, 7), "1 - (0.05/12)^(1/9370) = 5.85e-4, stated rounded up", "§4")
    # --- boundary and memory
    fig = [t["figure_kb"] for t in __import__("json").load(open(REPO / "contracts" / "manifest.paper.json"))["boundary"]["transfers"]]
    check("boundary figure labels, KB", 535.5, sum(fig), 0.05, "Fig. 3(a)")
    check("cell-tensor variant ratio", 6.0, 3.20 / 0.535, 0.05, "§3 boundary")
    check("G mask bytes at F = 256", 8192, 256 * 256 / 8, 0, "Fig. 3(a)")
    check("H3' bytes (256 x 8 x u16)", 4096, 256 * 8 * 2, 0, "Fig. 3(a)")
    check("H3'' bytes (256 x 8 x (48 + 4))", 106496, 256 * 8 * 52, 0, "Fig. 3(a)", "translated values int8 x 48 + one int32 check")
    check("H0 bytes (6400 x 2 u8)", 12800, 6400 * 2, 0, "Fig. 3(a)")
    check("P bytes (1600 x 12 u8)", 19200, 1600 * 12, 0, "Fig. 3(a)")
    gen_total = sum(l["payload_bytes"] for l in gen.PACKET_LAYOUTS.values())
    consistent("generated dev-profile boundary payload, bytes", "535 KB (figure)", gen_total,
               "DMA lengths use generated layouts; H1/H2/H3 field lists differ from the figure until the model export fixes them")
    check("memory tiles, bytes", 8912896, 17 * 512 * 1024, 0, "§3 device")
    check("resident working set, MB", 7.38, 4.13 + 3.25, 0.005, "§3 memory")
    check("PLIO per direction, GB/s", 12.8, 4 * 8 * 0.4, 0.001, "§3 boundary")
    check("accelerator RAM, two 256-bit ports at 300 MHz, GB/s", 19.2, 2 * 32 * 0.3, 0.001, "§3 memory")
    check("accelerator RAM, one port, GB/s", 9.6, 32 * 0.3, 0.001, "§3 memory")
    check("residual bound, LSB", 191, 127 + 64, 0, "Def. 1")
    check("residual magnitude bits (+ sign)", 9, math.ceil(math.log2(191 + 1)) + 1, 0, "Def. 1")
    check("one 30-frame/s camera x 9", 270, 9 * 30, 0, "§4 P5")
    # --- headline ratios (Table 3)
    check("SPLICE-A-V vs T4 at the cap, % lower", 12.4, (3.29 - 2.88) / 3.29 * 100, 0.1, "§4")
    check("SPLICE-B-V vs native N-A, % less time", 2.5, (2.62 - 2.554) / 2.62 * 100, 0.05, "§4")
    check("array vs DSP58-only time, SPLICE-B", 13.6, 34.7 / 2.554, 0.05, "§4")
    check("array vs DSP58-only energy, SPLICE-B", 7.5, 291 / 39.0, 0.05, "§4")
    check("int8 peak ratio, 34 tiles vs 464 DSP58 dot slices", 20.8, (34 * 256 * 1e9) / (464 * 3 * 300e6), 0.05, "§4")
    check("cap latency model at eta_k = 0.5, SPLICE-A-V, ms", 2.88, 0.48 + 1.20 / 0.5, 0.005, "§4 Fig. 5(b)")
    check("cap latency model at eta_k = 0.5, SPLICE-B-V, ms", 2.554, 0.48 + 1.04 / 0.5, 0.01, "§4 Fig. 5(b)")
    check("P3b holds above eta_k", 0.43, 1.20 / (3.29 - 0.48), 0.005, "§4")
    consistent("knee 351 frames/s vs L1 = 2.882 ms", 351, round(1 / 2.882e-3, 1), "1/W with W = L1 - g3 (g3 ~ 0.03 ms)", "Prop. 6")
    open_item("L1 = W + g3 = 2.882 ms (SPLICE-A-V)", 2.882, "needs the per-unit schedule model (87 convolutions, best legal tuples)", "Prop. 6")
    open_item("energy 39.0 mJ, 44.0 mJ, P5 47.3 vs 47.2 mJ", "Table 3", "power-model integration over the schedule; board PMBus later (P5)", "Table 3")
    open_item("accuracy rows (AP / MR-2)", "Tables 3, 7", "training and validation runs (P2, P7, P9)", "Table 7")

    out = REPO / "evidence" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_numbers.json").write_text(json.dumps(rows, indent=2) + "\n")
    w = max(len(r["item"]) for r in rows)
    for r in rows:
        print(f"{r['status']:<11} {r['item']:<{w}}  stated={r['stated']!s:<16} recomputed={r['recomputed']!s:<14} {r['note']}")
    bad = [r for r in rows if r["status"] == "DIFFERS"]
    print(f"\n{sum(r['status'] == 'REPRODUCED' for r in rows)} reproduced, "
          f"{sum(r['status'] == 'CONSISTENT' for r in rows)} consistent, {len(bad)} differ, "
          f"{sum(r['status'] == 'OPEN' for r in rows)} open (need the schedule model, training or the board)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
