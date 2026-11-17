# Acceptance matrix (roadmap sections 12, 13, 16)

A gate passes only when its evidence files exist and were produced by executed checks
(`evidence/<gate>/...json`, provenance tag inside).  A routed image, a generated header or a
passing host test is a partial milestone, never a pass of a hardware gate.

## Provenance classes (SPLICE-V section 4)

| tag | meaning | produced here by |
|---|---|---|
| S | simulation / emulation | `--backend reference|emulated` apps, x86sim/aiesim, HLS csim/cosim |
| R | post-route report | Vivado/Vitis reports copied into `evidence/G11/` |
| T | training / validation | the arm-(e) model work (G04, G12) |
| M | board measurement | `--backend board` apps (`ps/apps/*.py`) |

## Tests mapped to the manuscript

| test | criterion (roadmap 13) | where it runs | status |
|---|---|---|---|
| P1 internal conformance | six boundaries (ids, G, H3' sources, dV, logits, candidates) identical: sizes, hashes, mismatch counts, first mismatch; exact forecast/residual traces; cap, tie-rich, duplicate, budget-exit, QP-cap and clamp cases | `ps/splicev_ps/slice_runner.py` `compare()`, `ps/apps/g07_slice.py`; reference tests in `tests/` | toy slice exact on reference and emulated device; board run pending G02 |
| P2 accuracy, descriptor arm | DFP32 vs M non-inferiority: AP > -0.1, MR-2 < +0.2 (one-sided, paired bootstrap 10,000, three seeds, family correction for 12 tests) | model work (G04/G12) | blocked on the trained model |
| P3 timing | same detector/threshold on T4 and TE0950; deployment and evaluation thresholds | G12 | blocked |
| P4 latency model | predicted vs actual frame latency, p99 absolute error < 0.1 ms | G12 (`reference/splicev_ref/formulas.py` gives the predictions) | blocked |
| P5 energy | at 270 frames/s: E_frame = integral(P dt) / completed frames; enhanced <= 1.05 x native | `ps/splicev_ps/measure.py` with an audited rail map | blocked (rail inventory in G01) |
| P6 knee | observed throughput/EDP knee vs model | G12 | blocked |
| P7 accuracy, forecast arm | Dpub vs DFP32: AP > -0.3, MR-2 < +0.5 | G04/G12 | blocked |
| P8 movement ceilings | per-layer movement ceiling vs attained efficiency, exceptions listed | G12 | blocked |
| P9 occlusion | heavy-occlusion recall loss <= 0.5 percentage point; gate effect separate from exact skipping | G12 | blocked |

## Required negative and robustness cases (roadmap 13.1, 16)

| case | covered by |
|---|---|
| empty admissibility, F cap, N cap | `tests/test_gate_forecast_readout.py`, `tests/test_posterior_trust.py`, linkage cap handling in `sv_linkage_k.cpp` |
| tie-rich linkage, exhaustive small graphs (Lemma 2) | `tests/test_linkage.py` (7^6 and 7^4 exhaustive, random tie-rich) |
| forced linkage budget exits | `tests/test_linkage.py` (budget determinism), `linkage.stats` in the PL vectors |
| QP iteration cap, clamp hits | `tests/test_posterior_trust.py` vs `reference/splicev_ref/exact/oracles.py` |
| malformed units (bad magic/check/version/shape, scratch overflow, missing body) | `tests/test_units.py`, `basic` unit suite (G05) |
| stale or reordered responses | `ps/splicev_ps/workers.py` `check_responses` (frame id / slot echo, check words) |
| bounded waits, short streams | `tests/test_ps_board_paths.py` (emulated sink timeout) |
| repeat/reset | `--repeat N` of every app; reboot repeat for G02 |
