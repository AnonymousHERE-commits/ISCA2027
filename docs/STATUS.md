# Status

What has been checked, where, and what has not.  Gate-by-gate view: `planning/gates.json`.

## Verified in this repository (host only, no AMD tools, no board)

| item | check | result |
|---|---|---|
| integer reference vs exact oracles | `make test` (146 tests in all) | Lemma 2 linkage exhaustive on small graphs (7^6, 7^4 assignments) and tie-rich random cases; posterior within 2^-14.7 and trust QP within 2^-13.2 of exact rational solutions; screening sound; lumpability; RNE/saturation fixtures |
| generated sources | `make check-generated` | one manifest hash for C++ and Python consumers |
| paper numbers | `make audit` | 49 reproduced, 3 consistent, 0 differ, 3 open (schedule latency, energy, accuracy: need the schedule model, board power and training) |
| PL cores (g++, `__int128`) | `make host-pl` | reserve 4/4, linkage 5/5, gate 4/4, posterior 92/92, trust 154/154 cases bit-exact |
| PL kernel tops (word ABI) | `make host-pl` | sv_reserve, sv_linkage, sv_gate, sv_exchange1/2: 3/3 toy cases each, full and small capacities |
| AIE dispatcher, scalar path (g++) | `make host-aie` | 38 units (basic, toy0, toy1), every response word bit-exact, error statuses included |
| generated AIE graphs | `make host-graphs` | uniform1, split3, split6 compile against a mock ADF layer; every worker reproduces the reference through the simulator data path; smoke kernel exact |
| toy vertical slice orchestration | `tests/test_slice_runner.py` | units + PL engines reproduce the reference at all six boundaries (12 seeds checked; 3 in the test), for every worker-class set |
| PS board code | `make emulate`, `tests/test_ps_board_paths.py` | WorkerPool/Engines/apps on an emulated device whose kernel argument lists are checked against the HLS sources; bounded-wait failure path |
| detector float ops | `tests/test_detector_core.py` | bands within 3 LSB of the integer walk, restriction, forecast within 3 LSB, gate agreement > 97%, float exchange marginals / dV within 2 LSB of the integer exchange |
| PyTorch SPLICE neck | `tests/test_detector_torch.py` | forward pass end to end in the seven readout arms and the descriptor variants of SPLICE-V on a numpy stand-in for torch (fragments, differentiable exchange equal to the reference, residual definitions, exposure discount, readout invariants, losses, QAT grid, eval mode, YOLO26 output formats); real torch is used when installed |
| evaluators | `tests/test_detector_metrics.py` | CrowdHuman AP / MR^-2 equal to a literal transcription of the official loop (ignore regions, heavy-occlusion stratum, weights); COCO matching on known answers; two-level bootstrap |
| dataset views | `tests/test_detector_data.py` | COCO and CrowdHuman layouts discovered, disjoint train_fit / calib / val roles, labels, ignore regions, visible ratios, links without copies |
| Arrhenius suite | `tests/test_hpc_suite.py`, `tests/test_hpc_download.py` | the real launcher and runner through a mock Slurm in dry mode: job chain, gates, bounded retries, requeue after infrastructure faults, continuation of time-limited trainings (with and without `scontrol requeue`), optional and fatal failures, resume, reduction, packaging, time-limit clamp, the read-only check mode; the dataset download against a local server |
| synthetic-scene training | `tests/test_detector_train.py`, `artifacts/trained/synthetic_arm_e/` | forecast gradients vs finite differences; test-scene residual 895 -> 175 LSB^2 after training; calibrated manifest passes the contract checker; trained heads bit-exact through units and engines under the calibrated profile |

## Not run yet (needs a GPU cluster)

- **Image training of the detector and the train channel of SPLICE-V**: `detector/hpc/`
  (`check`, then the smoke scale, a pilot, and the registered protocol on COCO and CrowdHuman).
  No torch is installed where this repository was prepared, so the PyTorch code ran only on the
  numpy stand-in (forward pass); autograd, the Ultralytics data pipeline, YOLO26's criterion,
  the GPU and the real datasets are first exercised by the smoke scale on Arrhenius, whose
  prepare job runs the unit tests and the autograd self-checks on real torch.  No measured
  detector accuracy exists yet; the manuscript's values are carried only as targets.

## The current draft of SPLICE-V against this repository

| part | state |
|---|---|
| detector, training and train-channel measurements (`detector/`) | follow the current draft: seven readout arms, exposure discount in the confined forecast, descriptor variants, need in [0, 1] in Eq. (2), gradients through Eq. (1) and the optimality system of Eq. (2), 12 QAT epochs, Tables 5-7 and tests 2, 3, 9, 10, 12 in the reduction |
| integer reference, contract, generated sources, PL, AIE, PS | follow an earlier draft.  Known differences of the current draft: the forecast logit carries log(1 - xi_g), which has no integer format in the contract yet; the logic's DSP58 count is 104 of 464 (160 in the earlier draft); the analytical latency is 2.56 ms (2.55 ms) |
| `make audit` (paper numbers) | recomputes the earlier draft's numbers; not redone for the current draft |

## Not verified yet (needs AMD tools or the board)

- **Vitis HLS**: C simulation with `ap_int<128>` (`SPLICEV_USE_AP_INT`), synthesis, II and
  resources, RTL co-simulation.  The 128-bit divisions of the trust solver will be large; they are
  correct first, optimised later.
- **AIE compiler**: acceptance of the generated graph and kernels, program memory (16 KB) and
  data memory (40 KB dispatcher scratch + stack) fit, x86simulator/aiesimulator runs.  If the
  full dispatcher does not fit one tile, use the `split3`/`split6` worker classes.
- **v++ link/package**: option spellings of the installed 2025.2 tools, platform clock ids and
  memory banks, SD packaging details.
- **XRT on the board**: the `pyxrt` API variant (kernel/run/bo/graph), graph lifecycle
  (`run(0)` = until ended, `end(cycles)`), interrupts/polling, buffer groups.  `tools/probe_pyxrt.py`
  records what the installed binding offers.
- **Platform**: no qualified TE0950 XPFM exists in this repository (`platform/README.md`).

## Limitations of the current implementation

- The manifest is the **dev-synthetic** profile: deployable = no.  SYN-01..06 are synthetic
  calibration values, DD-01..19 are design decisions where the paper is silent, and three fields
  stay null (`generated/DECISIONS.md`).  The image-trained arm-(e) export (G04) replaces them;
  `artifacts/trained/synthetic_arm_e/` shows the replacement with synthetic-scene calibration
  (provenance S, still non-deployable).
- The vertical slice uses a synthetic scene and synthetic heads; it validates arithmetic,
  protocol and integration, not detection quality.
- Host-orchestrated mode: Python sequences the slice and performs the glue steps listed in
  `ps/splicev_ps/slice_runner.HOST_STEPS`.  The PL sequencer of the device-scheduled mode (G09) is
  not written.
- AIE bodies are the scalar, bit-exact path; vectorised AIE API versions (needed for the paper's
  throughput) must reproduce them exactly.
- Multiworker packet switching on the four handoff PLIOs, the partition catalogue (sixteen
  tuples) and the 34-worker graph (G08) are not started.
- `sv_linkage` stores exact-capacity neighbour lists per anchor: the toy-slice build (`small`,
  ~0.86 MB) fits the VE2302 only in UltraRAM, and the full-frame capacity (~8.3 MB) does not fit
  at all.  A pooled list store is required before full frames (WP-3).
