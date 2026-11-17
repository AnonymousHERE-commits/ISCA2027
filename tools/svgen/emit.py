"""Emit the four consumers' sources from one merged manifest.

Outputs (all deterministic, byte-identical for identical manifests):
  generated/cpp/splicev_gen.h       constants, tables and ABI for AIE C++ and Vitis HLS C++
  generated/py/splicev_gen.py       the same for the PS host and the integer reference
  generated/abi/boundary_layout.json byte-exact boundary packet layouts
  generated/abi/ABI.md              human-readable descriptor / packet tables
  generated/MANIFEST.sha256         hash of the canonical merged manifest
"""
from __future__ import annotations

import ast
import json
import operator
from collections import OrderedDict
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from . import tables as T
from .manifest import manifest_sha256, hash16

DTYPE_BYTES = {"u8": 1, "i8": 1, "u16": 2, "i16": 2, "u32": 4, "i32": 4, "u64": 8, "i64": 8}
_OPS = {ast.Add: operator.add, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv, ast.Div: operator.floordiv, ast.Sub: operator.sub}


def eval_count(expr: str, dims: Dict[str, int]) -> int:
    """Evaluate a count expression with + - * / over integers and dims names."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return int(dims[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv)) and a % b:
                raise ValueError(f"inexact division in count '{expr}'")
            return _OPS[type(node.op)](a, b)
        raise ValueError(f"unsupported count expression '{expr}'")
    return ev(ast.parse(str(expr), mode="eval"))


def packet_layouts(m: Dict[str, Any]) -> Dict[str, Any]:
    pk = m["packets"]
    dims = pk["dims"]
    align = pk["alignment_bytes"]
    header_bytes = 4 * pk["header_words"]
    out = OrderedDict()
    for label, spec in pk["layouts"].items():
        off = 0
        fields = []
        for name, dt, expr in spec["fields"]:
            n = eval_count(expr, dims)
            size = n * DTYPE_BYTES[dt]
            off = (off + align - 1) // align * align
            fields.append({"name": name, "dtype": dt, "count": n, "offset": off, "bytes": size})
            off += size
        payload = (off + align - 1) // align * align
        out[label] = {"label_id": spec["label"], "dir": spec["dir"], "header_bytes": header_bytes,
                      "payload_bytes": payload, "total_bytes": header_bytes + payload, "fields": fields}
    return out


def _q39(x) -> int:
    return T.to_q(x, 39)


def _ceil255(theta) -> int:
    """Smallest u8 code q with q/255 >= theta."""
    from decimal import ROUND_CEILING
    return int((Decimal(str(theta)) * 255).to_integral_value(rounding=ROUND_CEILING))


def scalar_constants(m: Dict[str, Any]) -> "OrderedDict[str, Tuple[int, str]]":
    g = m["geometry"]
    lv = g["levels"]
    f = m["formats"]
    w = m["walk"]
    b = m["bands"]
    lk = m["linkage"]
    sf = lk["score_format"]
    wm = lk["work_model"]
    tr = m["trust"]
    fc = m["forecast"]
    lf = fc["logit_format"]
    lm = fc["logit_multipliers"]
    rd = m["readout"]
    st = rd["sigmoid_table"]
    abi = m["dispatcher_abi"]
    c = OrderedDict()

    def put(name, value, ctype="int"):
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        c[name] = (value, ctype)

    put("DEPLOYABLE", bool(m["deployable"]))
    put("MANIFEST_HASH16", hash16(m))
    for name in ("P3", "P4", "P5"):
        put(f"{name}_H", lv[name]["h"]); put(f"{name}_W", lv[name]["w"])
        put(f"{name}_STRIDE", lv[name]["stride"]); put(f"{name}_STEPS", lv[name]["walk_steps"])
    put("N_P3", lv["P3"]["h"] * lv["P3"]["w"]); put("N_P4", lv["P4"]["h"] * lv["P4"]["w"]); put("N_P5", lv["P5"]["h"] * lv["P5"]["w"])
    put("C_LATENT", g["latent_channels"]); put("C_TRANSPORT", g["transport_channels"]); put("C_PROTECTED", g["protected_channels"])
    put("N_ANCHOR_P4", g["anchors_p4"]); put("N_RESERVE", g["anchors_p3_reserve"]); put("N_ANCHOR", g["anchors_total"])
    put("F_CAP", g["fragment_cap"]); put("N_CAP", g["units_per_row_cap"]); put("KMAX", g["kmax"])
    put("MS", g["sources_ms"]); put("KQP", g["qp_iterations_kqp"]); put("KEY_D", g["key_dim_d"])
    put("ACT_MIN", f["act"]["min"]); put("ACT_MAX", f["act"]["max"])
    put("TR_MIN", f["transport"]["min"]); put("TR_MAX", f["transport"]["max"])
    put("RES_BOUND", f["residual"]["bound_lsb"])
    put("TAP_Q", w["tap_frac_q"]); put("STATE_FRAC", w["state_frac"]); put("COEF_R", w["coef_bits_r"])
    put("AFF_FRAC", w["affinity_frac"]); put("BETA_FRAC", b["beta_frac"])
    for name in ("P3", "P4", "P5"):
        for j, e in enumerate(b["exponents"][name]):
            put(f"BAND_EXP_{name}_{j}", e)
    put("BAND_EXP_LOW", b["lowpass_exponent"])
    put("NU_ENERGY_SHIFT", m["descriptor_nu"]["energy_shift"])
    put("NU_DIM", m["descriptor_nu"]["levels"] * m["descriptor_nu"]["bands_per_level"])
    put("UNASSIGNED", m["membership"]["unassigned_id"])
    put("RESERVE_MIN_E", m["reserve"]["candidate_min_eligibility"])
    put("LINK_WINDOW", lk["window_chebyshev"]); put("LINK_SCORE_DEN", sf["den"])
    put("LINK_HI", sf["hi_code"]); put("LINK_VHI", sf["vhi_code"]); put("LINK_MIN_HI", lk["guard_min_pairs_above_theta"])
    put("LINK_BUDGET", lk["budget_cycles"]); put("LINK_SCAN_LANES", wm["scan_lanes"]); put("LINK_COMMIT_LANES", wm["commit_lanes"])
    put("LINK_MERGE_UNITS", wm["merge_units"]); put("LINK_PHASE_FILL", wm["phase_fill"]); put("LINK_NBMAX", lk["neighbour_list_cap"])
    put("RECV_MIN_E", _ceil255(lk["receiver_min_eligibility"])); put("DONOR_MIN_E", _ceil255(lk["donor_min_eligibility"]))
    for i, bound in enumerate(m["gate"]["strata"]["bounds"]):
        put(f"GATE_STRATUM_BOUND_{i}", bound)
    put("GATE_N_STRATA", len(m["gate"]["thresholds"])); put("GATE_PREC_FRAC", m["gate"]["prec_frac"])
    put("BF_MBITS", m["bf24"]["mantissa_bits"]); put("BF_GUARD", m["bf24"]["add_guard_bits"])
    put("LOG2E_Q30", T.log2e_q30(), "long long"); put("LN2_Q30", T.ln2_q30(), "long long")
    put("EVID_FRAC", 8)
    put("TRUST_FRAC", tr["frac_bits"])
    put("A_R_Q39", _q39(tr["a_r"]), "long long"); put("B_R_Q39", _q39(tr["b_r"]), "long long"); put("C_R_Q39", _q39(tr["c_r"]), "long long")
    put("D_MIN_Q39", 1 << (39 + tr["D_clamp_log2"][0]), "long long"); put("D_MAX_Q39", 1 << (39 + tr["D_clamp_log2"][1]), "long long")
    put("LAMBDA_MAX_Q39", tr["iterate_clamp"][1] << 39, "long long")
    put("KKT_TOL_Q39", 1 << (39 + tr["kkt_tolerance_log2"]), "long long")
    put("PHI_FRO2_MAX_LOG2", tr["phi_fro2_max_log2"])
    put("Q39_MAX", (1 << 47) - 1, "long long"); put("Q39_MIN", -(1 << 47), "long long")
    put("FC_B", fc["clip_B_lsb"]); put("FC_SHRINK_NUM", (1 << (-fc["epsilon_log2"])) - 1); put("FC_SHRINK_SHIFT", -fc["epsilon_log2"])
    put("FC_LOGIT_FRAC", lf["frac"]); put("FC_LOGIT_MIN", lf["clamp"][0]); put("FC_LOGIT_MAX", lf["clamp"][1])
    put("FC_KK_MULT", lm["kk_mult"]); put("FC_UU_MULT", lm["uu_mult"]); put("FC_SHIFT", lm["shift"])
    put("FC_NULL_MULT", lm["null_mult"]); put("FC_NULL_SHIFT", lm["null_shift"])
    put("FC_ALPHA_MULT", lm["alpha_mult"]); put("FC_ALPHA_SHIFT", lm["alpha_shift"])
    put("FC_UDIM", fc["u_dim"]); put("FC_SCALE_SHIFT", 30); put("FC_OUT_SHIFT", 45)
    put("SIG_IN_MIN", st["clamp"][0]); put("SIG_IN_MAX", st["clamp"][1]); put("SIG_IDX_SHIFT", st["index_shift"])
    put("SCORE_FRAC", rd["score_frac"]); put("TAU_DET_Q15", T.to_q(rd["tau_det"], 15))
    put("SKIP_BLOCK", rd["skip_block"]); put("RZ_SHIFT", rd["rz_shift"])
    put("CMD_MAGIC", abi["cmd_magic"], "unsigned"); put("RSP_MAGIC", abi["rsp_magic"], "unsigned")
    put("PROTO_VERSION", abi["protocol_version"]); put("DESC_WORDS", abi["descriptor_words"])
    put("RSP_WORDS", abi["response_header_words"]); put("TRL_WORDS", abi["trailer_words"]); put("SCRATCH_BYTES", abi["scratch_bytes"])
    put("CHECK_SEED", 0x9E3779B9, "unsigned")
    for k, v in abi["opcodes"].items():
        put(f"OP_{k}", v)
    for k, v in abi["status"].items():
        put(f"ST_{k}", v)
    for body, flags in abi["flags"].items():
        for k, v in flags.items():
            put(f"{body}_F_{k}", v)
    for body, subs in abi["subops"].items():
        for k, v in subs.items():
            put(f"{body}_{k}", v)
    put("PKT_MAGIC", m["packets"]["magic"], "unsigned"); put("PKT_HEADER_WORDS", m["packets"]["header_words"])
    return c


def bf_constants(m: Dict[str, Any]) -> "OrderedDict[str, List[Tuple[int, int]]]":
    out = OrderedDict()
    scale = Decimal(2) ** m["gate"]["prec_frac"]
    out["GATE_TAU_BF"] = [T.bf24_from_decimal(Decimal(str(t)) * scale) for t in m["gate"]["thresholds"]]
    pr = m["posterior"]["prior_table"]
    out["PRIOR_BF"] = [T.bf24_from_value(p) for row in pr for p in row]
    return out


def _cpp_literal(value: int, ctype: str) -> str:
    if ctype == "long long":
        return f"{value}LL" if value != -(1 << 63) else "(-9223372036854775807LL - 1)"
    if ctype == "unsigned":
        return f"{value}u"
    return str(value)


def _wrap(items: List[str], per_line: int = 12, indent: str = "    ") -> str:
    lines = []
    for i in range(0, len(items), per_line):
        lines.append(indent + ", ".join(items[i:i + per_line]) + ",")
    return "\n".join(lines)


def emit_cpp(m: Dict[str, Any], sha: str) -> str:
    consts = scalar_constants(m)
    tabs = T.all_tables(m)
    bfc = bf_constants(m)
    abi = m["dispatcher_abi"]
    layouts = packet_layouts(m)
    s = []
    s.append("// GENERATED FILE - DO NOT EDIT. Source: SPLICE-V integer manifest.")
    s.append(f"// manifest_id: {m['manifest_id']}  profile: {m['profile']}  deployable: {str(bool(m['deployable'])).lower()}")
    s.append(f"// manifest sha256: {sha}")
    s.append("// Regenerate with: python3 tools/generate_sources.py")
    s.append("#ifndef SPLICEV_GEN_H")
    s.append("#define SPLICEV_GEN_H")
    s.append("")
    s.append("namespace svgen {")
    s.append(f'static const char MANIFEST_ID[] = "{m["manifest_id"]}";')
    s.append(f'static const char PROFILE[] = "{m["profile"]}";')
    s.append(f'static const char MANIFEST_SHA256[] = "{sha}";')
    for name, (value, ctype) in consts.items():
        s.append(f"static const {ctype} {name} = {_cpp_literal(value, ctype)};")
    s.append("")
    for name, pairs in bfc.items():
        s.append(f"static const int {name}_N = {len(pairs)};")
        s.append(f"static const unsigned {name}_M[{len(pairs)}] = {{ {', '.join(str(p[0]) + 'u' for p in pairs)} }};")
        s.append(f"static const int {name}_E[{len(pairs)}] = {{ {', '.join(str(p[1]) for p in pairs)} }};")
    s.append("")
    s.append("// Descriptor and response fields: word index, least significant bit, width.")
    for kind, fields in (("DESC", abi["descriptor"]), ("RSP", abi["response"]), ("TRL", abi["trailer"])):
        for name, word, lsb, width in fields:
            up = name.upper()
            s.append(f"static const int {kind}_{up}_W = {word}; static const int {kind}_{up}_LSB = {lsb}; static const int {kind}_{up}_WID = {width};")
    s.append("")
    s.append("// Boundary packets (byte offsets within the payload; header is PKT_HEADER_WORDS x 32 bit).")
    for label, lay in layouts.items():
        up = label.upper()
        s.append(f"static const int PKT_{up}_LABEL = {lay['label_id']}; static const int PKT_{up}_PAYLOAD_BYTES = {lay['payload_bytes']};")
        for fld in lay["fields"]:
            fu = fld["name"].upper()
            s.append(f"static const int PKT_{up}_{fu}_OFF = {fld['offset']}; static const int PKT_{up}_{fu}_COUNT = {fld['count']};")
    s.append("")
    pl_only = ("RECIP_SEED", "EXP2_HI")   # BF24 tables of the PL posterior/trust engines
    for name, values in tabs.items():
        ctype = "unsigned" if name in ("FC_EXP", "RECIP_SEED", "EXP2_HI") else "short"
        lit = [f"{v}u" if ctype == "unsigned" else str(v) for v in values]
        if name in pl_only:
            s.append("#ifndef SVGEN_AIE  // not needed on the AIE tiles: keeps their data memory free")
        s.append(f"static const {ctype} {name}[{len(values)}] = {{")
        s.append(_wrap(lit))
        s.append("};")
        if name in pl_only:
            s.append("#endif")
    s.append("}  // namespace svgen")
    s.append("")
    s.append("#endif  // SPLICEV_GEN_H")
    return "\n".join(s) + "\n"


def emit_py(m: Dict[str, Any], sha: str) -> str:
    consts = scalar_constants(m)
    tabs = T.all_tables(m)
    bfc = bf_constants(m)
    abi = m["dispatcher_abi"]
    layouts = packet_layouts(m)
    s = []
    s.append('"""GENERATED FILE - DO NOT EDIT. Source: SPLICE-V integer manifest.')
    s.append(f"manifest_id: {m['manifest_id']}  profile: {m['profile']}  deployable: {bool(m['deployable'])}")
    s.append("Regenerate with: python3 tools/generate_sources.py")
    s.append('"""')
    s.append(f'MANIFEST_ID = "{m["manifest_id"]}"')
    s.append(f'PROFILE = "{m["profile"]}"')
    s.append(f'MANIFEST_SHA256 = "{sha}"')
    for name, (value, _ctype) in consts.items():
        s.append(f"{name} = {value}")
    for name, pairs in bfc.items():
        s.append(f"{name} = {tuple(tuple(p) for p in pairs)!r}")
    s.append("BAND_EXP = {")
    for lvl in ("P3", "P4", "P5"):
        s.append(f"    {lvl!r}: {tuple(m['bands']['exponents'][lvl])!r},")
    s.append("}")
    s.append("LEVELS = {")
    for lvl in ("P3", "P4", "P5"):
        L = m["geometry"]["levels"][lvl]
        s.append(f"    {lvl!r}: {{'h': {L['h']}, 'w': {L['w']}, 'stride': {L['stride']}, 'steps': {L['walk_steps']}}},")
    s.append("}")
    s.append("DESC_FIELDS = {")
    for name, word, lsb, width in abi["descriptor"]:
        s.append(f"    {name!r}: ({word}, {lsb}, {width}),")
    s.append("}")
    s.append("RSP_FIELDS = {")
    for name, word, lsb, width in abi["response"]:
        s.append(f"    {name!r}: ({word}, {lsb}, {width}),")
    s.append("}")
    s.append("TRL_FIELDS = {")
    for name, word, lsb, width in abi["trailer"]:
        s.append(f"    {name!r}: ({word}, {lsb}, {width}),")
    s.append("}")
    s.append(f"OPCODES = {dict(abi['opcodes'])!r}")
    s.append(f"STATUS = {dict(abi['status'])!r}")
    s.append(f"PACKET_LAYOUTS = {json.loads(json.dumps(layouts))!r}")
    for name, values in tabs.items():
        s.append(f"{name} = (")
        s.append(_wrap([str(v) for v in values]))
        s.append(")")
    return "\n".join(s) + "\n"


def emit_abi_md(m: Dict[str, Any], sha: str) -> str:
    abi = m["dispatcher_abi"]
    layouts = packet_layouts(m)
    s = [f"# Generated ABI ({m['manifest_id']}, profile {m['profile']})", "",
         f"Manifest SHA-256: `{sha}`. Generated file; edit the manifest and run `python3 tools/generate_sources.py`.", "",
         "## Unit descriptor (16 x 32-bit words, little-endian)", "",
         "| field | word | lsb | width |", "|---|---|---|---|"]
    for name, word, lsb, width in abi["descriptor"]:
        s.append(f"| {name} | {word} | {lsb} | {width} |")
    s += ["", "## Unit response header (8 x 32-bit words)", "", "| field | word | lsb | width |", "|---|---|---|---|"]
    for name, word, lsb, width in abi["response"]:
        s.append(f"| {name} | {word} | {lsb} | {width} |")
    s += ["", "## Unit response trailer (4 x 32-bit words, after the payload)", "", "| field | word | lsb | width |", "|---|---|---|---|"]
    for name, word, lsb, width in abi["trailer"]:
        s.append(f"| {name} | {word} | {lsb} | {width} |")
    s += ["", f"Framing: {abi['framing']}.", "", f"Check word: {abi['check_rule']}.", "", "## Opcodes", ""]
    for k, v in abi["opcodes"].items():
        s.append(f"- `{k}` = {v}")
    s += ["", "## Boundary packets", "",
          "Each packet carries an 8-word header (magic, label, version, frame_id, generation, payload_bytes, count, check).", "",
          "| label | dir | payload bytes | figure KB | fields |", "|---|---|---|---|---|"]
    fig = {t["label"]: t["figure_kb"] for t in m["boundary"]["transfers"]}
    total = 0
    for label, lay in layouts.items():
        total += lay["payload_bytes"]
        flds = ", ".join(f"{f['name']}:{f['dtype']}x{f['count']}" for f in lay["fields"])
        s.append(f"| {label} | {lay['dir']} | {lay['payload_bytes']} | {fig.get(label, '-')} | {flds} |")
    s += ["", f"Generated payload total: {total} bytes ({total / 1000:.1f} decimal KB); figure labels total {sum(fig.values()):.1f} KB.",
          "Figure labels are rounded display values; DMA lengths, FIFO sizes and counters use the generated sizes (roadmap 2.3)."]
    return "\n".join(s) + "\n"


def _get_path(doc: Dict[str, Any], dotted: str):
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return "(not present)"
        node = node[part]
    return node


def emit_decisions_md(m: Dict[str, Any], sha: str) -> str:
    """Every value the paper does not fix, with its id, kind and reason (roadmap 2.1)."""
    dec = m.get("decisions", {})
    s = [f"# Design decisions and synthetic values ({m['manifest_id']}, profile {m['profile']})", "",
         f"Manifest SHA-256: `{sha}`. Generated from the manifest's `decisions` table; edit the manifest, not this file.", "",
         "- **DD-xx** (design): an implementation choice where SPLICE-V is silent; it must be confirmed against the model "
         "source before a deployable manifest is frozen.",
         "- **SYN-xx** (synthetic): a calibration value used only for bring-up on synthetic data; it must be replaced by "
         "the trained arm-(e) export (gate G04).", "",
         f"Deployable: **{'yes' if m.get('deployable') else 'no'}**.", "",
         "| id | kind | manifest field | value | reason |", "|---|---|---|---|---|"]
    def key(item):
        ident = item[1].get("id", "")
        kind, _, num = ident.partition("-")
        return (kind, int(num) if num.isdigit() else 0, item[0])
    for path, d in sorted(dec.items(), key=key):
        v = json.dumps(_get_path(m, path), separators=(",", ":"))
        if len(v) > 80:
            v = v[:77] + "..."
        why = str(d.get("why", "")).replace("|", "/")
        s.append(f"| {d.get('id', '')} | {d.get('kind', '')} | `{path}` | `{v}` | {why} |")
    items = m.get("unresolved", []) if isinstance(m.get("unresolved"), list) else []
    if items:
        s += ["", "## Listed as unresolved by the paper seed (manifest.paper.json)", ""]
        for it in items:
            s.append(f"- `{it.get('field')}`: {it.get('why')}" if isinstance(it, dict) else f"- {it}")
    return "\n".join(s) + "\n"


def generate(m: Dict[str, Any]) -> Dict[str, str]:
    sha = manifest_sha256(m)
    layouts = packet_layouts(m)
    return {
        "DECISIONS.md": emit_decisions_md(m, sha),
        "cpp/splicev_gen.h": emit_cpp(m, sha),
        "py/splicev_gen.py": emit_py(m, sha),
        "abi/boundary_layout.json": json.dumps({"manifest_sha256": sha, "layouts": layouts}, indent=2) + "\n",
        "abi/ABI.md": emit_abi_md(m, sha),
        "MANIFEST.sha256": f"{sha}  {m['manifest_id']}\n",
    }
