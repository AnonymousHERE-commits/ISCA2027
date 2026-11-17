# Architecture of this implementation

This document describes how the repository implements SPLICE-V on the Versal VE2302 of the
Trenz TE0950, and how correctness is carried from the manifest to the device.  The Versal
track follows an earlier draft of SPLICE-V; the detector package follows the current one
(`docs/STATUS.md`).
Method details are in the paper; the implementation plan is `docs/IMPLEMENTATION_ROADMAP.md`.

## 0. The detector

The SPLICE detector (YOLO26-S with the SPLICE neck; seven readout arms, deployed arm (e)) is
the PyTorch package `detector/splice_det`: model, training on a GPU cluster (the Arrhenius
suite in `detector/hpc/`), calibration on the calibration split, quantization-aware training
at the manifest's rounding sites, evaluation of the paper's train channel, and an export that
writes int8 weights, the calibrated coefficients and a model manifest.  The float neck
operations follow the integer reference's conventions and the discrete decisions (reserve,
linkage) are made by the integer reference itself, so a trained model maps onto model M.
`detector/README.md` maps each detector stage to its torch code, reference module and device
engine; `detector/export_spec.json` (generated from the manifest) lists every value and tensor
the export must supply to turn the bring-up profile into a deployable manifest.

## 1. One contract, four consumers

```
contracts/manifest.paper.json   paper seed: every number SPLICE-V fixes; nulls where it is silent
contracts/manifest.dev.json     bring-up profile: overrides + additions (ABI, packets) + decisions
        |                         (DD-xx design choices, SYN-xx synthetic calibration) - never deployable
        v  tools/generate_sources.py (tools/svgen)
generated/cpp/splicev_gen.h     -> AIE kernels (aie/), HLS kernels (pl/), C++ testbenches
generated/py/splicev_gen.py     -> integer reference (reference/), PS runtime (ps/)
generated/abi/ABI.md, boundary_layout.json, generated/DECISIONS.md, MANIFEST.sha256
```

Every consumer is generated from the same canonical manifest hash; unit descriptors carry
`MANIFEST_HASH16` and the PS refuses a release built from another manifest
(`ps/splicev_ps/release.py`).

## 2. Integer reference and exact oracles

`reference/splicev_ref/` is the executable definition of every integer operation (Definition 1
of the paper): RNE shifts and divisions, explicit saturation, walk taps in Q1.14, band
coefficients from monotone cumulative sums, 24-bit block floating point for the posterior,
the screened trust QP in Q8.39, the gated forecast (Eq. 3), the readout (Eq. 5).  Independent
exact oracles (`reference/splicev_ref/exact/`) check it: Lemma 2 linkage exhaustively on small
graphs, the posterior within 2^-14.7 and the trust QP within 2^-13.2 of exact rational results,
screening soundness, lumpability (Lemma 3).

## 3. Mapping onto the device

| engine | role | code |
|---|---|---|
| AIE-ML (shared workers) | one static graph of dispatcher workers; each invocation executes one *unit* chosen by a 16-word descriptor: CONV, GEMM, STENCIL (walk + bands), LUT, PERMUTE, SEGOPS (restriction, fill, detail, synthesis, class band), TAPMOD (modulation, energies, nu, mix), HEAD (forecast Eq. 3, score) | `aie/kernels/sv_units.h`, `aie/templates/`, `tools/gen_aie_graph.py` |
| PL (HLS C++) | decisions with data-dependent control: reserve selection (FPS), certified MNN linkage with counted work budget, conformal gate -> shared mask G, exchange phase 1 (ESP posterior + screened trust QP -> sources H3', lambda) and phase 2 (dV), DDR<->PLIO movers | `pl/include/` (cores), `pl/engines/` (kernel tops) |
| PS (Python on the A72, XRT) | loads the release, owns buffers and frame ids, submits work with bounded waits, checks every response, records evidence | `ps/splicev_ps/`, `ps/apps/` |

### Dispatcher unit protocol

```
in : descriptor (16 words: magic 'SPLV', version|hash16, frame, slot/opcode/phase/flags, layer/band/partition,
     worker/lanes, dims m n k, in_words, out_words, param0..2, token, check) + in_words payload words
out: header (8 words: magic, frame, opcode|status|slot|worker, out_words, token, in_words, 0, check)
     + out_words payload words (only when status is OK) + trailer (4 words: saturations, aux0, aux1, check)
```

Payloads are padded to whole 64-bit PLIO beats; check words use `c = rotl32(c, 5) ^ w`.  A unit
is rejected before any body runs (bad magic/check/version, missing body in this worker class,
shape, scratch), so the host can size every counted receive exactly
(`units.predict_status`, `units.response_length`).  Field-level layout: `generated/abi/ABI.md`.

### Worker classes

An AIE-ML tile has 16 KB of program memory.  The graph generator builds one kernel per *class*
(`SV_BODY_MASK`, bit i = opcode i) from `aie/templates/`; `aie/worker_classes.json` defines the
sets (`uniform1` = one full worker, `split3` = conv+gemm / stencil+tapmod / lut+permute+segops+head,
`split6`, ...).  `reference/splicev_ref/routing.py` routes every unit to a worker that implements
its body; the same routing produces the simulator data and the PS submissions, and the tests
show the six boundaries do not depend on the class set.

## 4. The toy vertical slice (gate G07)

`reference/splicev_ref/toy.py` defines a synthetic three-level scene and its reference run;
`ps/splicev_ps/slice_runner.py` executes the same computation as device steps:

```
A1   STENCIL x3 (walk + 4 detail bands + low-pass per level)            AIE
H0   sv_reserve (FPS reserve on P3)                                      PL
A2a  sv_linkage (certified MNN, counted work budget) -> ids, order      PL
     sv_gate (fused hypotheses, Mondrian thresholds) -> G               PL
A2b  SEGSUM/SEGMEAN restriction -> H; ENERGY x3 + NU -> nu;              AIE
     GEMM heads (K, U, null/alpha, VO, need/rho/phi)
     HEAD FORECAST reading the same G -> dH_hat (with abstention)        AIE
X    GEMM K K^T evidence -> sv_exchange1 (posterior + trust) -> H3'      AIE, PL
     PERMUTE priority products -> sv_exchange2 -> dV                     AIE, PL
A3a  SEGOPS SYNTH with dH_hat                                            AIE
A3b  residual r = dH - dH_hat (int16) -> GEMM r Z -> class band only;    AIE
     DETAIL (box path, no residual), GEMM logits, HEAD SCORE -> candidates
```

The six P1 boundaries (ids, G, H3' sources, dV, logits, candidates) are compared with the
reference by `compare()`.  In this host-orchestrated bring-up mode Python sequences the steps
and performs the few glue steps listed in `slice_runner.HOST_STEPS`; the device-scheduled mode
(G09) moves sequencing to a PL sequencer.

## 5. Verification chain

```
manifest --generate--> reference (Python) ==exact oracles==> (tests/)
reference --vectors--> C++ PL cores/kernels (g++, __int128)           make host-pl
reference --vectors--> AIE dispatcher (g++, scalar path)              make host-aie
reference --sim data--> generated worker graphs (mock ADF layer)      make host-graphs
reference --> PS board code on an emulated device (fakexrt)           make emulate / tests
----------------------------- requires AMD tools / board ------------------------------
reference --vectors--> HLS csim / cosim (ap_int<128>)                 make hls-csim / hls-cosim
reference --sim data--> x86simulator / aiesimulator                   make aie-sim
reference --> board (XRT): G02 smoke, G05 units, G06 engines, G07 slice  ps/apps/
```

Everything above the line is exercised in this repository; everything below has the code,
configuration and commands, and produces its evidence when run with the tools and the board.
