# SPLICE-V on Versal VE2302 (Trenz TE0950)

Implementation of **SPLICE-V**: the graph-wavelet SPLICE neck with certified fragment linkage,
the shared conformal admissibility mask, the confined early forecast, the screened trust
exchange and the late exact residual, mapped onto the AMD Versal VE2302 of the Trenz
TE0950-03 (`xcve2302-sfva784-1LP-e-S`).  The detector, its training and the measurements of
the paper's `train` channel follow the current draft of SPLICE-V; the Versal track
(contract, integer reference, PL, AIE, PS) follows an earlier draft and the roadmap
(`docs/STATUS.md` lists what the current draft changes there):

- **PL**: Vitis HLS C++ decision engines (reserve, certified linkage, conformal gate,
  posterior + trust exchange) and DDR/PLIO movers
- **AIE-ML**: C++ shared dispatcher workers in one static ADF graph (eight unit bodies)
- **PS**: Python on Linux/XRT (release loading, submission, bounded waits, evidence)

Everything is generated from one integer manifest and checked bit for bit against an integer
reference, which is itself checked against exact oracles.  The implementation follows
`docs/IMPLEMENTATION_ROADMAP.md` gate by gate.

```
SPLICE detector (seven readout arms, PyTorch: detector/) model, training suite (Slurm), calibration, QAT, evaluation, export
  -> integer contract (contracts/, generated/)          one manifest hash for every consumer
  -> integer reference (reference/splicev_ref/)         forward semantics of model M, exact oracles
  -> PL engines (pl/)  +  AIE-ML workers (aie/)          bit-exact C++ on the VE2302
  -> PS runtime and gate applications (ps/)             Python/XRT on the A72, evidence files
  -> system build, platform, release (Makefile, platform/, tools/)
```

Which detector stage runs where: `detector/README.md`.

> **Status.** All host-side checks pass (Python reference and oracles, C++ engines and
> dispatcher, generated AIE graphs, PS board code on an emulated device, the PyTorch detector's
> forward pass in every readout arm, its synthetic-scene training, the evaluators, and the
> Arrhenius training suite through a mock Slurm).  The detector's image training
> (`detector/hpc/README.md`) has not been run yet: its smoke scale on Arrhenius is the first
> execution of the PyTorch code on a GPU.  The HLS/AIE compilers, the simulators and the board
> have not been run yet; the TE0950 platform still has to be built; the manifest is the
> non-deployable *dev-synthetic* profile until the image-trained arm-(e) export replaces its
> synthetic values.  Details: `docs/STATUS.md`.

## Quick start (host only)

```bash
git clone https://github.com/AnonymousHERE-commits/ISCA2027.git splicev && cd splicev
make test          # 146 host tests
make host-cpp      # C++ PL engines, AIE dispatcher and generated graphs vs the reference
make emulate       # the board applications on the emulated device
```

Hardware flow (platform, HLS, AIE, link, package, deploy, board runs): `docs/BUILD_AND_RUN.md`.
Detector training and the train-channel measurements (Arrhenius, Slurm): `detector/hpc/README.md`.

## Layout

| path | content |
|---|---|
| `detector/` | the SPLICE detector: PyTorch model on YOLO26-S in its seven readout arms (`splice_det/`), training/calibration/QAT/evaluation/export, the Arrhenius suite (`hpc/`: launcher, runner, dataset download), stage-to-engine map, generated `export_spec.json` |
| `contracts/` | `manifest.paper.json` (paper seed, unresolved fields) and `manifest.dev.json` (bring-up profile: ABI, packets, DD/SYN decisions) |
| `generated/` | C++/Python constants, ABI, packet layouts, `DECISIONS.md` - generated, never edited |
| `reference/splicev_ref/` | integer reference of every operation, exact oracles, unit/packet builders, routing, toy slice |
| `pl/` | HLS cores, kernel tops, testbenches, kernel/system table, HLS config template |
| `aie/` | dispatcher bodies, graph/worker templates, worker classes, smoke graph, host tests |
| `ps/` | XRT runtime (`splicev_ps/`), gate applications (`apps/`), native graph helper |
| `platform/` | TE0950 platform work package and templates |
| `tools/` | generators, vectors, configuration rendering, release, preflight, probes, audits |
| `tests/` | host test suite (with a mock Slurm for the training suite) |
| `planning/` | gates G00-G13 (`gates.json`), acceptance matrix, open work packages |
| `docs/` | roadmap, architecture, build and run, status |
| `evidence/` | immutable run reports (preflight, audit, gate results) |
| `artifacts/` | trained exports (`trained/`), checkpoints, reference vectors, releases |

## Gates

| gate | content | state |
|---|---|---|
| G00, G03 | identity, manifest, generated contract, reference | done for the dev profile |
| G01, G02 | inventory, platform, smoke on the board | tooling ready; needs platform and board |
| G05-G07 | AIE units, PL engines, toy vertical slice | host-verified; ready for tools and board |
| G04 | detector model, training, calibration, export | code, synthetic-scene export and the Arrhenius suite done; image training pending (smoke scale first) |
| G08-G13 | scale-out, schedule, full detector, closure, study, release | open |

See `planning/gates.json` for the required evidence of every gate.
