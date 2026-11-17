# Design decisions and synthetic values (splicev-model-synthetic-trained, profile model)

Manifest SHA-256: `6fd5b465bff41d0d38609472e22e0cd205a4a28b5b69094f5ea66392a1550c3d`. Generated from the manifest's `decisions` table; edit the manifest, not this file.

- **DD-xx** (design): an implementation choice where SPLICE-V is silent; it must be confirmed against the model source before a deployable manifest is frozen.
- **SYN-xx** (synthetic): a calibration value used only for bring-up on synthetic data; it must be replaced by the trained arm-(e) export (gate G04).

Deployable: **no**.

| id | kind | manifest field | value | reason |
|---|---|---|---|---|
| CAL-01 | calibrated | `bands.exponents` | `{"P3":[4,4,4,3],"P4":[4,4,3,3],"P5":[4,4,3,2]}` | floor(log2(127 / q99.99 /Z/ / lsb)) on the calibration split |
| CAL-01 | calibrated | `bands.lowpass_exponent` | `0` | low-pass band range on the calibration split |
| CAL-02 | model | `gate.strata` | `{"by":"fragment_count_F","bounds":[32,96,192]}` | strata of the calibration |
| CAL-02 | calibrated | `gate.thresholds` | `[13.314546671973261,13.314546671973261,13.314546671973261,13.314546671973261]` | Mondrian conformal quantile of same-instance d^2, delta = 0.01 |
| CAL-03 | calibrated | `posterior.prior_table` | `[[0.3967226380740441,0.4675298401780295,0.13574752174792637]]` | frequencies of min(K, Kmax) true partners among admissible partners |
| CAL-04 | model | `trust.a_r` | `0.25` | trust constant of the trained model |
| CAL-04 | model | `trust.b_r` | `1.0` | trust constant of the trained model |
| CAL-04 | model | `trust.c_r` | `0.5` | trust constant of the trained model |
| CAL-05 | model | `forecast.logit_multipliers` | `{"kk_mult":1,"uu_mult":1,"shift":9,"null_mult":1,"null_shift":9,"alpha_mult":...` | derived from the export scales |
| CAL-05 | model | `forecast.u_dim` | `16` | width of U in the trained model |
| CAL-06 | model | `readout.rz_shift` | `7` | exponent of the exported residual maps Z_l |
| DD-01 | design | `formats.transport.max` | `127` | symmetric transport range |
| DD-01 | design | `formats.transport.min` | `-127` | symmetric transport range |
| DD-01 | design | `formats.transport.saturation` | `"symmetric"` | makes /dH/ <= 127 so the stated /r/ <= 127 + B = 191 LSB holds (roadmap 2.2) |
| DD-02 | design | `modulation.w_frac` | `15` | same affinity codes as the walk export |
| DD-02 | design | `walk.affinity_frac` | `15` | w in Q0.15; lateral tap = div_rne(w << 13, 2^15 + sum w) is exact integer export of P = I/2 + S^-1(I+W)/2 |
| DD-03 | design | `bands.beta_frac` | `16` | beta in Q0.16; powers by repeated RNE products give monotone cumulative sums |
| DD-04 | design | `bands.exponent_semantics` | `"static per (level, band); band j int8 code = sat8(rshift_rne(acc_j, 23 - e_j...` | static exponents keep the band format fixed per layer |
| DD-05 | design | `descriptor_nu.definition` | `"nu[l,j] = min(127, rshift_rne(div_rne(sum_{i in f at level l} sum_{c<48} z_j...` | per-fragment band energies, paper section 2 |
| DD-05 | design | `descriptor_nu.energy_shift` | `7` | nu as int8 log-free energy code; replace by the model's descriptor definition |
| DD-06 | design | `membership.mean_rounding` | `"half_even"` | exact integer division rounded half-even |
| DD-06 | design | `membership.unassigned_id` | `65535` | u16 id per cell, 0xFFFF reserved |
| DD-07 | design | `membership.level_extension` | `{"P4":"the anchor's fragment id","P3":"own fragment id if the cell is a reser...` | must match the trained model's export |
| DD-08 | design | `membership.class_band_weights` | `"cell eligibility e (u8) at the level; uniform weights if the fragment's elig...` | R needs nonzero weights at every level; anchors exist only at P3/P4 |
| DD-08 | design | `membership.unassigned_readout` | `"class band row = 0 (Pi row is zero), box band = N, fill rZ = 0"` | follows Pi = M R with zero rows for unassigned cells (roadmap 2.2, Lemma 3 note) |
| DD-09 | design | `reserve.candidate_min_eligibility` | `77` | receivers e >= 0.3 (Fig. 1) -> code 77 |
| DD-09 | design | `reserve.distance` | `"squared Euclidean distance on the P3 grid (integers)"` | integer distance, no rounding |
| DD-09 | design | `reserve.priority_rule` | `"first pick argmax t; then argmax dmin2 * (256 + t)"` | thin-aware farthest-point sampling with deterministic ties |
| DD-09 | design | `reserve.tie_order` | `"smallest P3 cell index"` | deterministic total order |
| DD-10 | design | `linkage.score_format` | `{"bits":8,"den":255,"theta_rule":"2*sum > 255*count","hi_code":128,"vhi_code"...` | u8 scores q/255; theta_T and the guard compared by integer cross-products |
| DD-11 | design | `linkage.neighbour_list_cap` | `128` | bounded cluster lists; overflow is a counted protocol error, never a silent truncation |
| DD-11 | design | `linkage.work_model` | `{"scan_lanes":64,"commit_lanes":64,"merge_units":16,"phase_fill":8,"check":"a...` | logical work charge shared by reference and PL; independent of FIFOs and clocks (roadmap 8.6) |
| DD-12 | design | `gate.mask_layout` | `"full square F_cap x F_cap bits, row-major, LSB-first within 32-bit words, ro...` | fixed 8192-byte square mask (roadmap 8.3) |
| DD-12 | design | `gate.prec_frac` | `8` | precision codes p = 2^8 / sigma^2 in mu-code units; tau is stored as tau * 2^8 |
| DD-13 | design | `bf24.add_guard_bits` | `2` | addition is exact RNE: exponent differences below 24 + guard (= 26) are aligned exactly in 64 bits and normalized once; larger differences return the larger operand, which is the RNE result |
| DD-13 | design | `bf24.recip_seed_table` | `"T[i] = floor(2^43 / (2 i + 4097)), 0 <= i < 2048, Q2.30; two Newton steps y ...` | 12-bit seed + two Newton steps as drawn in Fig. 3(d) |
| DD-14 | design | `posterior.duplicate_grouping` | `"identity (each admissible source is its own unit)"` | identity grouping until the model's duplicate rule is imported; E_dup = 0 |
| DD-14 | design | `posterior.evidence_format` | `"int16 Q7.8"` | pair evidence s in Q7.8; w = e^s via a two-level 2^f table with relative error below 2^-20 |
| DD-15 | design | `trust.phi_presolve_source` | `"per-source moment rows phi_g (4 x Q8.39) delivered with H3 head outputs; the...` | interface decision; resolves the dependency audit of roadmap 2.2 only once the model confirms the producer |
| DD-16 | design | `forecast.abstention_event` | `"C_f empty, or the null weight w_0 >= sum of admissible weights (counted, nev...` | operational definition of the counted abstention (roadmap 2.2) |
| DD-16 | design | `forecast.exp_table` | `{"entries":256,"out_frac":8,"offset":8,"formula":"round_half_even(2^8 * exp((...` | uint32 weights; numerator and denominator sums fit int64 |
| DD-16 | design | `forecast.logit_format` | `{"bits":16,"frac":4,"clamp":[-128,127]}` | Q?.4 logits clamped to [-8, 8): the table exponential needs no row maximum, so column blocks add exactly |
| DD-17 | design | `readout.sigmoid_table` | `{"entries":256,"in_frac":8,"clamp":[-2048,2047],"index_shift":4,"formula":"mi...` | sigma < 1 strictly, Q0.15 |
| DD-17 | design | `readout.skip_block` | `8` | skip granularity for Proposition 3 |
| DD-18 | design | `dispatcher.binary_layout` | `"dispatcher_abi"` | 16-word little-endian descriptor (roadmap 9.3) |
| DD-19 | design | `boundary.packet_layouts` | `"packets"` | generated byte-exact layouts; figure KB are labels only (roadmap 2.3) |

## Listed as unresolved by the paper seed (manifest.paper.json)

- `bands.exponents / bands.lowpass_exponent / bands.exponent_semantics`: per-layer and per-band exponents are not given; static vs dynamic selection not prescribed (roadmap 7.3)
- `activation and weight scales, requantization sites`: come from the trained arm-(e) export (G04)
- `forecast.exp_table / readout.sigmoid_table / bf24.recip_seed_table / exp input range`: LUT contents and input ranges not given (roadmap 2.2)
- `formats.transport.saturation`: 191-LSB residual bound needs |dH| <= 127; symmetric clipping must be confirmed (roadmap 2.2)
- `partition.catalog`: the sixteen legal tuples, placements, buffer capacities and DMA patterns are not enumerated (roadmap 7.4)
- `dispatcher.binary_layout / boundary.packet_layouts`: descriptor concepts only; packet packing and candidate limits unspecified (roadmap 9.2-9.3)
- `gate.strata / gate.thresholds / posterior.prior_table / posterior.duplicate_grouping / posterior.duplicate_edge_bound`: Mondrian strata, calibrated thresholds, partner-count priors and the duplicate rule come from calibration (roadmap 2.2)
- `linkage.work_model / linkage.neighbour_list_cap / linkage.score_format`: logical cost per work item and budget check order unspecified (roadmap 2.2)
- `trust.a_r / trust.b_r / trust.c_r / trust.phi_presolve_source`: model constants; Eq. (2) needs Phi_f before the trust solve but its producer is scheduled after it (roadmap 2.2 dependency audit)
- `forecast.probe_distribution / forecast.abstention_event`: training recipe and the counted abstention event are not given (roadmap 2.2)
- `membership.unassigned_id / level_extension / class_band_weights / unassigned_readout`: Lemma 3 adds an unassigned class; the deployed handling of unassigned readout cells must be specified (roadmap 2.2)
- `reserve.priority_rule / reserve.distance / reserve.tie_order / reserve.candidate_min_eligibility`: farthest-point sampling details are not given
- `E_dup enforced bound`: Eq. (6) reproduces 41,664 cycles only at E_dup = 1,024; the enforced bound must come from the model (roadmap 2.3)
- `platform: XPFM, boot image, kernel, device tree, XRT, Python bindings, clocks, power-monitor mapping`: board-specific; produced by G01 inventory and the platform track
