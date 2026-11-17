# SPLICE-V implementation roadmap

Source: `SPLICEV_Implementation_Roadmap.pdf` (13 November 2026, 22 pages), the
implementation instruction for this repository.  The text below was extracted with
`pdftotext -layout` (page headers removed); tables keep the PDF's column layout.  The PDF is
authoritative where the extraction is ambiguous.

## Where each roadmap section is implemented

| roadmap section | repository |
|---|---|
| 2 Freeze detector, artifacts, incomplete specs | `detector/` (import checklist, `export_spec.json`), `contracts/manifest.paper.json` (paper seed, `unresolved`), `generated/DECISIONS.md`, `tools/check_contract.py` |
| 2.3 Numeric accounting (Fig. 3 bytes, Eq. 6 E_dup) | `tools/audit_paper_numbers.py` -> `evidence/audit/paper_numbers.json`, `generated/abi/boundary_layout.json` |
| 3 Platform strategy, 3.3 first hardware test | `platform/`, `aie/smoke/`, `pl/engines/sv_movers.cpp`, `ps/apps/g02_smoke.py`, `tools/preflight.py`, `tools/probe_pyxrt.py` |
| 4 Repository and release layout | this tree; `tools/release.py` (release identity), `evidence/` |
| 5 Manifest-first generated contract | `contracts/`, `tools/svgen/`, `tools/generate_sources.py`, `generated/` |
| 6 Reference model, training | `reference/splicev_ref/` (integer reference), `reference/splicev_ref/exact/` (exact oracles), `detector/README.md` (training/export notes) |
| 7 AIE-ML work packages | `aie/kernels/sv_units.h` (dispatcher, 8 bodies), `aie/templates/` + `tools/gen_aie_graph.py` (graph, worker classes) |
| 8 PL engines (HLS C++) | `pl/include/` (cores), `pl/engines/` (kernel tops), `pl/tb/`, `pl/kernels.json`, `pl/configs/` |
| 9 Schedule and compact ABI | `contracts/manifest.dev.json` (`dispatcher_abi`, `packets`), `generated/abi/ABI.md` |
| 10 Memory, clocks, feasibility | `reference/splicev_ref/formulas.py`, `tests/test_membership_packets_formulas.py` |
| 11 PS runtime and system build | `ps/splicev_ps/`, `ps/apps/`, `ps/native/`, `Makefile`, `tools/build_cfg.py` |
| 12 Stage gates G00-G13 | `planning/gates.json`, `planning/ACCEPTANCE.md`, `docs/STATUS.md` |
| 13 Test plan P1-P9 | `ps/splicev_ps/slice_runner.py` (P1 boundary comparison), `planning/ACCEPTANCE.md` |
| 15 First commands | `docs/BUILD_AND_RUN.md` |
| 16 Troubleshooting rules | `docs/BUILD_AND_RUN.md` (troubleshooting) |

## Text

```text
R E V I S I O N 1 2 · I M P L E M E N TAT I O N & V E R I F I C AT I O N

SPLICE-V
on Versal VE2302
From the upgraded detector
to a reproducible heterogeneous implementation

PL — Vitis HLS C++
AIE-ML — C++ kernels and one static ADF graph
PS — Python on Linux / XRT
Hardware — Trenz TE0950-03, VE2302
Workspace — ~/projects

   Design principle
   Preserve the fragment representation, share the admissible mask, forecast early, and deliver
   the exact residual only to its readout consumer.

   Build correctness gates before scale, throughput and energy claims.

Based on the SPLICE-V manuscript (SPLICE-V.pdf).
13 November 2026 · Roadmap and host-tested planning utilities.
This document is not a completed FPGA implementation.

Contents
1. What must change from the previous implementation plan
2. Freeze the detector, artifacts and incomplete specifications
3. Platform strategy: qualify one complete toolchain
4. Repository and release layout
5. Manifest first: a generated cross-engine contract
6. Reference model and training the upgraded detector
7. AIE-ML C++ implementation work packages
8. PL engines in Vitis HLS C++
9. Schedule and compact interface ABI
10. Memory, clocks and feasibility gates
11. Python PS runtime and system build
12. Stage-gated implementation roadmap
13. Test plan mapped to the manuscript
14. Baselines, optimization and resource choices
15. First commands and immediate next decision
16. Release acceptance and troubleshooting rules
Sources and reading order

Source of truth: the SPLICE-V manuscript (SPLICE-V.pdf).
Target: Trenz TE0950-03, xcve2302-sfva784-1LP-e-S.
Project root: ~/projects.
Isolated workspace: ~/projects/splicev.
Languages: programmable logic in Vitis HLS C++; AIE-ML kernels and ADF graph in C++;
processing-system application in Python/XRT.
Prepared: 13 November 2026.

Scope and evidence
This is an implementation specification and staged execution plan, not a completed accelerator. The
accompanying pack contains host-tested invariant utilities, a paper-derived contract seed, platform
inventory scripts and implementation work packages. It does not contain the trained arm-(e)
detector, a complete integer manifest, all accelerator sources, a qualified TE0950 XPFM, or a
runnable XCLBIN.

References [SPLICE-V] identify the manuscript by section, equation, figure or table.
[D1]–[D8] identify official AMD/Trenz material checked for the tool
flow. Recommendations and proposed file/ABI names are explicitly engineering choices; they are
not quotations from the paper.

The latest paper distinguishes documentation, published baselines, analytical design values,
simulation, post-route reports, training and board measurements. Its Table 3 labels the proposed-
system entries A, while Table 6 reports validation outcomes. Preserve those distinctions. Neither a
manuscript number nor this planning pack proves a corresponding local TE0950 measurement.
Import the actual evidence files before marking a local implementation gate as passed. [SPLICE-V §4,
Tables 3, 5–6]

1. What must change from the previous
implementation plan
The upgraded method is not the old forecast with a new title. The following changes alter code,
interfaces, training and scheduling.

 Revision-12 requirement                Consequence for implementation

 Admissible set C_f exists              Predict hypotheses per anchor in A1; fuse them after fragment IDs; execute the
 immediately after linkage              conformal gate before the forecast.

 Forecast is masked, can abstain        Replace free all-fragment attention with Eq. (3), including the null option, band-
 and is shrunk                          descriptor compatibility, clipping and shrinkage.

 One mask is shared by forecast         Add handoff G; use the same frame/version-tagged mask on AIE and PL. Do not
 and exchange                           independently reconstruct it.

 Signed bands have one exponent         Accumulate bands from walk powers in int32; quantize once per band. Do not
 each                                   subtract already-quantized int8 low-pass tensors.

 Late residual is int16                 Preserve the stated 191-LSB range and use int16×int8 residual GEMMs. An int8
                                        residual loses information.

 The neck is nonlinear                  Late residual execution is exact relative to the co-designed detector; equivalence
                                        to earlier exact transport is an accuracy question, not an identity.

 Budgets count work, never              Freeze a logical work-accounting rule shared by reference and PL; distinguish it
 elapsed time                           from watchdogs and measured clock cycles.

 Revision-12 requirement                Consequence for implementation

 Movement bound includes bank           Select partitions with Lemma 1, including DMA reads, rereads, writes and legal
 capacity and output                    neighbor/cascade connections.

 Sixteen legal tuples; 34 or 32         Supply the actual catalog and validate every selected placement. A product of
 active tiles                           dimensions below 34 is not sufficient for legality.

 128 physical lanes, 32/64/128          Keep per-row arithmetic order independent of lane assignment. Reduce enabled
 enabled                                work with clock enables, not by changing DSP modes.

 Accelerator RAM uses two 256-bit       A one-port feeder cannot supply the four 400-MHz weight streams at their
 ports                                  modeled peak.

 Six internal conformance               Compare IDs, mask, source lists, delta-V, logits and candidates; matching final
 boundaries                             boxes alone is insufficient.

The new boundary is approximately 535 KB in ten transfers, not the old 498-KB/nine-transfer
plan. The four transfer labels H3, H3', H3'' and H4 must retain their distinct meanings. Both joins
are hidden only in the declared operating regime; the paper explicitly describes exposure above
approximately eta_k = 0.75. [SPLICE-V §§2–3, Figs. 3–4, Props. 4–6]

Recommended order: implement SPLICE-B, deployed arm (e), first; retain SPLICE-A as a separate
model/configuration after the common core is verified. Keep the native YOLO26-S baseline
executable throughout. This is an engineering sequencing choice, not a change to the paper's
model definitions.

2. Freeze the detector, artifacts and incomplete
specifications

2.1 Do not reconstruct a trained detector from prose
Import the model source, exact commit, trained checkpoint, export graph, preprocessing, class
ordering, calibration splits and metric implementation. Record every hash. The revised forecast
introduces learned maps that are not supplied as weight values in the PDF: U, u0, u_alpha, O_F, the
per-level residual projections Z_l, and the anchor hypothesis head. Reusing a checkpoint trained for
the old free forecast does not reproduce arm (e). [SPLICE-V §2, Eq. (3), Table 7]

The initial export inventory should describe each layer's input/output shapes, channels, tensor
layout, integer scales, operation type and legal consumers. Compare its layer/operator inventory
with the paper's model inventory; do not substitute a newly installed YOLO package version merely
because its model name matches.

Preserve five separately identified versions: Dpub, DFP32, Db, M, and H. The comparison chain is the
paper's mechanism for separating the readout change, bounded algorithms, quantized arithmetic
and hardware conformance. A single checkpoint or one global accuracy number cannot represent all
five. [SPLICE-V §1; §4]

2.2 Items the PDF does not fully specify
The following require the actual artifact or an explicit, versioned design decision before deployment.
The pack records these as unresolved fields instead of inventing values.

 • The exact per-layer and per-band exponents; activation/weight scales; LUT contents; exponential
   input range; reciprocal seed table; integer divisions and rounding sites; saturation versus
   rejection policy.

 • The sixteen legal partition tuples and their placements, buffer capacities and DMA patterns; the
   precise eight dispatcher unit bodies; packet packing and candidate limits.
 • The exact transport output range: the paper's 191-LSB residual bound uses 127+64, while an
   unrestricted signed int8 value also includes -128. Confirm symmetric clipping or another range
   guarantee; keep the residual int16 either way.
 • The complete Mondrian density strata, calibrated thresholds, prior partner-count tables,
   duplicate grouping rule and duplicate-edge bound.
 • The logical cost of each linkage work item and the order in which a budget is checked; the
   behavior when an early-exit trust row remains feasible but does not satisfy a stricter mass
   predicate.
 • The distribution and amplitude of the fragment-coordinate probe delta used during training; the
   operational event used to count forecast abstention.
 • Board-specific XPFM, boot image, kernel, device tree, XRT, Python bindings, exposed clocks and
   power-monitor mapping.

Two dependency audits are particularly important. First, Eq. (2) requires Phi_f before solving for
trust, whereas the selected-source translation/value/check GEMMs are scheduled after the source
list is known. The actual model must specify how the pre-solve error moments are obtained. Do not
schedule their only producer after the trust solver or replace them with an untrained approximation.
Second, Lemma 3 introduces an unassigned-cell class for its stochastic argument, while deployed
membership permits unassigned cells. Keep the proof-domain extension distinct from real fragment
IDs and specify how every unassigned readout cell is handled. [SPLICE-V §2, Eq. (2); §3, Lemma 3]

2.3 Resolve numeric accounting without changing the algorithm
Figure 3's displayed payload rows sum to 535.5 decimal KB, consistent with its approximately
535-KB total. Use exact byte sizes from generated packet layouts, not rounded figure labels, for
DMA lengths, FIFO sizing and counters.

Equation (6) contains E_dup. Substitution of its listed cap parameters (F=256, N=128, n=128, Kmax=3, Ms=8,
KQP=16, cr=8, cw=10, cf=32) gives:

  C_X = 33,472 + 8 * min(E_dup, 4,096) cycles

Thus 41,664 cycles corresponds to E_dup=1,024; the same formula gives 66,240 cycles at E_dup=4,096.
This is an arithmetic audit of the stated model, not a new hardware result. Obtain the actual
enforced duplicate-edge bound before treating the quoted point as a universal cap. Do not silently
cap E_dup at 1,024 merely to recover that number. The included audit reproduces this calculation.
[SPLICE-V §3, Eq. (6), p. 7]

3. Platform strategy: qualify one complete
toolchain

3.1 Preferred production track
For replication of the manuscript's environment, target Vivado/Vitis/AIE tools 2025.2 with a
regenerated and qualified TE0950 platform and compatible Linux/XRT software. The old accelerator-
only TE0950 hardware flow used 2024.2, while the earlier board Linux image was 2024.1. Neither is
proof that the new AIE+PL Python runtime is available. [SPLICE-V §4; project history]

Trenz's published TE0950 test-board project documents 2024.2. Its extensible-platform tutorial
describes an older 2023.2-based procedure. These are board-specific starting points, not evidence
that a ready-made 2025.2 TE0950 XPFM exists. AMD's custom-thin-platform tutorial is a useful
process reference, but its VCK190/VEK280 board files must not be used as TE0950 board data. [D1,
D2, D7]

If a matched 2025.2 TE0950 platform cannot yet be built, qualify the full 2024.2 stack as a
separate development branch. Rebuild all HLS/AIE/platform artifacts there and record the
deviation. Do not combine newly compiled 2025.2 AIE kernels, an old packaged 2024.2 PL IP, and a
2024.1 root filesystem as though they were a tested release.

3.2 Hardware-platform work package
Clone the vendor design with all relative dependencies. Retain CIPS, DDR timing, reference clocks,
pin constraints, reset topology, boot support and required board-management functions. Trim
unused video/peripheral logic only in a new branch after the base design boots and its exported
interfaces are understood.

Expose and document: AIE array access, NoC/DDR connectivity, PL control and memory ports, reset
domains, interrupts, clocks, PLIO endpoints, and accelerator-RAM access. Produce an extensible
XSA/XPFM with a software domain and Linux artifacts matching the chosen integration flow. The
fixed XSA from the previous bare-metal test is not interchangeable with that platform.

Produce platform_report.json with part, board revision, tool builds, XPFM/XSA hashes, accessible
memory banks, clock IDs, AIE tile allocation, stream interfaces and Linux/XRT versions. Read the
tool-generated address map. Do not reuse 0xA4090000, 0xA4140000 or 0xB0000000 from the earlier
experiment without generating and checking the new map.

3.3 First hardware acceptance test
Build one AIE-ML C++ integer pass-through or add kernel with PL HLS C++ MM2S/S2MM movers.
Control it from Python on the A72 through XRT. Test multiple packet lengths, exact bytes, reset/
restart and repeated submissions.

Pass evidence: valid platform identity, successful Linux/XRT load, correct Python graph/kernel/
buffer API availability, zero byte mismatches, legal stream completion, bounded waits, and
repeatability after reboot. This establishes only the platform. It is not a SPLICE detector result.

The Python binding must actually expose the graph API used by the host. A successful import pyxrt
alone does not establish this. Probe the installed module and run the loopback. An absent graph
binding requires a matched XRT build or an explicitly documented binding layer, not an invented API
call. [D5]

4. Repository and release layout
Install this pack under the project root without replacing the earlier plan:

  ~/projects/
    splicev/
      paper/SPLICE-V.pdf
      docs/IMPLEMENTATION_ROADMAP.md
      contracts/manifest.paper.json
      reference/                 # host invariant helpers now; full oracle later
      tests/                     # host tests now; engine tests later
      tools/                     # install, inventory, audits, constant generation
      planning/                  # work packages, acceptance matrix
      platform/                  # TE0950 platform specifications
      aie/                       # C++ kernel / ADF implementation specification
      pl/                        # HLS C++ implementation specification
      ps/                        # Python runtime implementation specification
      generated/                 # paper constants now; deployment ABI later
      artifacts/checkpoints/     # imported, versioned trained weights
      artifacts/references/      # exact manifest inputs and expected traces
      build/                     # disposable, stage-specific outputs
      logs/
      evidence/                  # immutable run reports and provenance

The earlier project remains outside this directory. Reuse its engineering lessons, not
its numerical contract, fixed addresses or compiled binary. This is a different detector,
different graph structure and different host/runtime architecture.

Each release must bind the model hash, complete manifest hash, generated-source hash,
checkpoint hash, XPFM/XSA, AIE archive, PL kernel objects, XCLBIN/PDI, boot image, Python
package versions and test-vector hashes. A successful command exits zero only when its required
tests have executed and passed. Never print a deployment PASS because a file exists.

5. Manifest first: a generated cross-engine
contract

5.1 Paper constants to preserve
 Quantity                         Required meaning

 Input                            640×640, batch 1; P3/P4/P5 = 80² / 40² / 20²

 Latent channels                  64, split into transported 48 and protected 16

 Anchors                          1,600 P4 + 256 reserved P3 anchors

 Fragments / row units            F <= 256, N_f <= 128

 Partner / source / solve caps    Kmax=3, Ms=8, KQP=16

 Walk lengths                     4, 8, 12 at the three levels

 Walk arithmetic                  int16 states; Q1.14 taps; exact integer self tap

 Bands                            int32 accumulation, one final conversion to int8 at each band's exponent

 Posterior                        24-bit block-floating mantissas

 Trust                            Q8.39, wide block-floating directions; explicit [0,2] iterate clamp

 Forecast                         same admissible set as exchange, null option, shrink epsilon=1/16, clip=64 LSB

 Residual                         int16; stated magnitude bound 191 LSB; int16×int8 readout product

 Class gate                       Q0.15 product rounded down; sigmoid strictly below one

 Quantity                         Required meaning

 Linkage budget                   48,000 units of specified work-cycle accounting, not wall-clock elapsed time

Source: [SPLICE-V §2; Definition 1; Props. 1–4; Table 5].

5.2 Generator outputs and ownership
The complete generator should emit common integer types/constants, packet schemas, Python
serialization, LUTs, C++ headers, layer descriptors, legal partition IDs, reference vectors and
assertions. Emit four consistent consumers: AIE C++, PL HLS C++, PS Python, integer-reference
Python. Handwritten control logic may remain, but it must consume generated numeric constants
and ABI definitions. [SPLICE-V p. 5]

manifest.paper.json in this pack is only a seed. Its null fields intentionally prevent deployment
approval. generate_paper_constants.py emits clearly marked paper constants, not complete kernels or
a final numerical implementation.

In the final manifest, specify sign-bit conventions explicitly. A label such as Q8.39 is not a substitute
for total_bits, fractional_bits, signed, intermediate widths, saturation mode and rounding location.
Use Python arbitrary-precision integers for the oracle, not floating-point emulation of wide fixed
point.

5.3 Numeric tests before any engine build
Test positive and negative half-even ties, maximum/minimum inputs, multiplication widths,
narrowing, shifts of zero and negative scale adjustments, saturations, zero denominators and exp
underflow. Distinguish an explicit arithmetic narrowing from a C++ cast. Verify CPU, HLS and AIE
conversions against the same generated vectors.

The class-score product intentionally uses a different rounding rule from most operators. For
nonnegative Q0.15 inputs:

  score = (sigmoid_code * gate_code) >> 15

A block is skippable only when all its gate codes are strictly below the matching threshold code,
with the complete receptive halo handled. Gate-induced accuracy change is separate from the
exact skip of an already-gated detector. [SPLICE-V Prop. 3]

6. Reference model and training the upgraded
detector

6.1 Reference ladder
Implement a slow, readable Python reference before optimizing kernels. Maintain separate exact/
rational or high-precision mathematical oracles and the executed integer manifest reference. The
former verifies error bounds; the latter defines bit-for-bit device outputs.

For every example, export six primary boundary traces: memberships/anchor order, admissible
mask, selected source lists, delta-V, class logits and candidates. Add forecast, residual, bands, work
counts, trust iterates and saturation counters for diagnosis. Cross-engine conformance is to the
integer reference; FP32/TensorRT remains an accuracy/performance comparator, not a bit-exact
oracle. [SPLICE-V §4]

Start with synthetic fixtures: constants; impulses; two fragments; no admissible pair; tie-rich
window linkage; F=0/1; row N=0/1/128; F=256; near-threshold pair distances; a zero/null forecast;
exhausted work budgets; maximum residual; and successive frames with radically different
fragment counts.

6.2 Gated forecast: implement Eq. (3), not free attention
For admissible donor g, compute the sum of the scaled key inner product and the band-descriptor
compatibility term. Add a distinct null logit. Normalize over C_f union {0}. Form a clipped
innovation relative to the receiver's own state, not an unconstrained weighted value. Apply
the shrink factor before the final manifest-defined quantization:

  logit_fg = dot(K_f,K_g)/sqrt(64) + dot(U*nu_f,U*nu_g)
  logit_f0 = dot(u0,nu_f)
  alpha_f = (1 - 1/16) * sigmoid(dot(u_alpha,nu_f))
  forecast_f = alpha_f * SUM_g a_fg * clip_64(V_g*O_F - H_f^s)

Masked donors contribute exactly zero. The null contributes to the denominator but contributes no
transported innovation. Do not renormalize only the non-null survivors after softmax; doing so
removes abstention. Empty C_f produces zero forecast. In exact arithmetic, each channel is
bounded by alpha_f*(1-a_f0)*64; fixed-point error must be accounted for when converting that bound
to codes. The supplied rational tests cover these structural properties, not the unspecified deployed
LUT. [SPLICE-V §2, Eq. (3)]

6.3 Training and calibration sequence
Import the original detector as arm (a). Train arm (e)'s hypotheses, band-conditioned forecast and
revised readout with the exact exchange in the loop. Use the paper's detector loss plus the mean-
squared forecast loss with mu=0.1, stopping gradients into the exchange target, and the fragment-
constant sensitivity penalty with lambda_s=0.05.

The auxiliary forecast loss must update only its intended maps; do not accidentally train the exact
exchange to imitate the forecast. Define and record the random fragment probe's distribution and
units. The PDF does not supply that full recipe. Quantization-aware training must execute the
eventual rounding and branch semantics in its forward path; any chosen gradient estimator must
be separately documented. Refit the integer-dependent calibration tables on the declared
calibration split, not on the final test set. [SPLICE-V §2, Eq. (5) discussion; §4]

Export and freeze DFP32, Db and M. For each, retain clean AP/MR, occlusion strata, fragment-count
distributions, forecast residuals, crossing-share measurements, band exponents and calibration
hashes. Do not reuse an old arm-(d) checkpoint and call the new gated model reproduced.

6.4 Exactness and approximation are different tests
At the final readout, the exact residual joins only the class-band input. Holding forecast/features
fixed, changing r must leave raw box-head outputs unchanged. Candidate selection can still change
because classification scores change; that does not violate the branch-separation requirement.

Earlier in the nonlinear neck, replacing exact transport with a forecast is not algebraically invisible.
Measure the fragment-direction sensitivity and the lumpability quantities in Lemma 3. An exact
Q*M=0 test at the readout cannot replace the P7 non-inferiority experiment against the earlier-
injection detector. Likewise, exact telescoping holds before independently quantizing each signed
band; after per-band int8 conversions, test the specified reconstruction/error contract rather than
claiming an unqualified integer identity. [SPLICE-V Lemma 3; Prop. 5]

7. AIE-ML C++ implementation work packages

7.1 Compile feasibility before the complete model
Build a one-worker kernel for each primitive shape before composing the full dispatcher: int8 GEMM,
int16×int8 GEMM, int16 walk stencil, ordered fragment reduction, band accumulation and LUT/
normalization operations. Verify one transaction, then multiple transactions and deliberate
backpressure. Follow this with a two-tile neighbor-sharing/cascade test, then a four-tile partition
switch, and only then the 34-worker graph.

The target has finite program and data memory. Official AIE-ML guidance states 16 KB of program
memory and 64 KB of local data memory split into eight banks, with access to north/south/west
neighbors where present. The eight-body dispatcher must actually fit; a large C++ switch is not
proof of fit. Track code size, stack, spill traffic and per-bank placement for every added body. Edge
tiles do not have all possible neighbors. [D3, D4]

Use one compiled ADF graph for the shared workers. Runtime descriptors choose a unit type and a
legal precompiled partition; they do not create a new graph topology. If a common dispatcher does
not fit its instruction memory, reduce its specialization/unrolling or create a documented worker-
class variant and re-evaluate the paper's sharing comparison. Do not silently reserve separate
static graphs and retain the shared-array performance claim.

7.2 Suggested C++ modules
 Proposed source         Required behavior and acceptance evidence
 module

 dense_i8.cpp            YOLO/MLP and fragment products; exact int32 accumulation, generated conversion; compare
                         all output elements.

 typed_window.cpp        Assemble each output cell's window, apply ownership taps and perform the specified
                         intermediate int8 rounding before convolution.

 walk_i16.cpp            Q1.14 row, exact self tap, three three-tap bands, one accumulation and rounding per walk
                         step; constant reproduction.

 bands_i32.cpp           Combine the stored walk powers with signed coefficient differences; one per-band exponent
                         and final rounding.

 membership.cpp          Permute anchors, ordered partial sums, exact division policy, fragment fills and retained detail.

 forecast.cpp            Mask G, null row option, descriptor compatibility, clipped innovation, shrink and final
                         quantization.

 fragment_products.c     Hypothesis/fragment heads and the post-trust priority translation/value/check products.
 pp

 readout.cpp             int16 residual subtraction and int16×int8 projection; class-only insertion; gate and
                         deterministic local candidates.

 dispatcher.cpp,         Validate descriptors, legal unit selection, per-generation ownership, completion and static ADF
 graph.cpp               connectivity.

These module names are proposed organization. The paper says eight dispatcher bodies; map
these implementation modules to eight body/opcode groups only after the complete operator
inventory is known. Do not mistake the module list for an already specified ABI. [SPLICE-V §3, shared
workers and array kernels]

7.3 Walks and bands
Use the paper's native block shapes as performance candidates: 4×8×8 int8 products, 2×4×8 int16
walk blocks and 4×4×8 mixed-width stencil blocks. Confirm exact API support and achieved
performance in the pinned AIE tool release. The quoted 154/47/63 useful MAC-per-cycle rates are
design/model inputs, not acceptance thresholds automatically achieved by source templates. [SPLICE-V
p. 6; Table 5]

For a three-tap band over two adjacent output cells, place the six useful taps in a 2×4 matrix and
multiply the shared four-cell, eight-channel panel. Accumulate the three spatial bands before one
step conversion. Use a temporal halo covering the selected walk depth; check corners, boundaries,
masks and strides. Preserve the row-sum constraint in exported taps. A constant-preserving row-
stochastic operation does not generally conserve the ordinary global sum; the revised paper
explicitly avoids relying on weighted mass conservation after quantization. [SPLICE-V Prop. 1]

For wavelet bands, compute signed coefficient combinations on the walk powers directly. Fixed
calibrated exponents are an engineering option, but the PDF does not prescribe how e_j is selected.
Freeze either static or dynamic exponent semantics in M and its QAT path before implementation.
Measure band clipping and rare low-amplitude bands; independent int8 low-pass differencing is the
ablation, not the deployed operation.

7.4 Movement-aware partition selection
For every exported layer, evaluate all legal tuples (g_c,g_o,g_i) plus their sharing factor sigma and
stationary-bank plan. The paper's lower bound includes compute, input-panel rereads and output
bytes:

  W = b*k^2*c                 X = a_in*c/sigma
  input_bytes = min(ceil(X/M_b)*W + X, W + ceil(W/M_b)*X)
  T >= max(a*b*k^2*c/(R*eta_k), input_bytes/B, a*b/B_o)

Use B=B_o=8 bytes/cycle and the paper's 16-KB logical-bank parameter only within its stated model.
Measure real DMA arbitration, locks, padding and cascades. Select on maximum tile completion
time, not an average that hides a slow tile. Keep g_i partial sums unrounded until they are
combined in the prescribed order. [SPLICE-V Lemma 1]

The sixteen-entry catalog itself is not enumerated in the PDF. Build and freeze legal entries from the
actual 17×2 placement, supported neighbor sharing, cascade direction, bank capacity and DMA
routing. Test that changing among legal tuples leaves all output codes unchanged. Retain row
slicing and output-channel slicing as ablations.

8. PL engines in Vitis HLS C++

8.1 Decision arithmetic is still arithmetic
The paper's phrase “every multiply–accumulate runs on the array” should not be read as “the PL
performs no multiplication.” Its detailed exchange lanes explicitly use DSP58 wide products,
recurrences, reciprocals and comparisons. Implement dense neural tensor products on AIE,
while preserving the specified numerical arithmetic inside PL decision engines. DSP mode is a static
hardware choice; a wide-product lane cannot become a three-term dot-product slice merely
because a descriptor changes. [SPLICE-V §3, pp. 4, 7]

Start with one or a few lanes. Prove the numerical function and protocol. Scale to 32, 64 and 128
only after reports show the actual multiplication mapping, memory-port demand, initiation interval
and schedule. A handwritten claim of one DSP per lane is not enough: unconstrained wide
multiplications or aggressive unrolling can instantiate more DSPs.

8.2 Reserve and certified linkage
Implement the P3 reserve selector with deterministic ties and the paper's P4-parent window
convention. For linkage, maintain window-pair sum, count and support-guard statistics. Compare
average scores by adequately widened integer cross-products instead of floating division. Keys are
the largest anchor index in a cluster; pair ordering is decreasing score followed by ascending
ordered pair key.

For every round, find both the first eligible neighbor and the first guard-failed neighbor. Commit only
mutual pairs that satisfy the certified rule: the round's first eligible pair, or a pair with no earlier
guard-failed incident rival. Merge disjoint committed pairs in a deterministic order. Plain MNN
merging, a smallest-index key, or ignoring a failed support guard is a different algorithm. [SPLICE-V
Lemma 2]

Compare uncapped partition equivalence with the sequential rule using exhaustive small windows
and tie-rich cases. For budgeted behavior, compare against the reference executing the same
counted-work branch; do not claim an unfinished partition equals unconstrained full linkage.
Preserve the current valid membership when the work budget is exhausted.

8.3 Early gate, shared mask
Immediately after linkage, fuse already available anchor hypotheses into fragment hypotheses
using the specified precision weighting. Evaluate each unordered pair's Mahalanobis statistic
against the density-stratum threshold. Store the symmetric admissibility result once and send G to
AIE. The exchange reads that same stored mask.

At F=256, an explicitly stored full F×F mask needs 65,536 bits, or 8,192 bytes, before packet
headers/alignment. This derives from the paper's dimensions; whether the transport sends a full
square or packed triangle is an ABI decision. Mask row order, padding bits, symmetry, diagonal
handling and frame ID must be fixed. [SPLICE-V §3, early gate; Fig. 3]

The modeled gate count is (5*F^2 + 12*1856)/n + 2*c_f, or 2,798 cycles at 128 lanes and F=256.
Measure the actual pipeline, stream and RAM stalls separately. An early gate that arrives late must
stall the forecast, not enable an ungated fallback.

8.4 Posterior and screened trust
Implement the elementary-symmetric-polynomial prefix/suffix recurrence with 24-bit block-floating
mantissas, explicit exponents and normalization after every specified operation. Keep prior
weighting and truncation to min(Kmax,N_f) exact to the reference. Use independent enumeration on
small rows to validate marginals, then high-precision checks at the cap. Record every exceptional
normalization/denominator branch. [SPLICE-V Eq. (1); Prop. 2]

Form the mutual minimum, screen with the paper's conservative rounding-inflated inequality and
choose up to eight survivors in the manifest's total order. Implement the primal active-set solve
from lambda=0, with Q8.39 storage/directions and at most 16 iterations. Compute feasibility and
the KKT residual independently. A capped row returns its last feasible iterate and is counted; it is not
a converged optimum.

The new [0,2] stored-iterate clamp is not the previous roadmap's [0,1] assumption. Neither the
clamp nor nonnegativity alone proves the exact optimum's strict trust-mass bound for an
interrupted solve. Freeze the intended capped-row policy and monitor both clamp hits and mass
predicates; do not quietly renormalize or replace the optimizer. [SPLICE-V Definition 1; Prop. 2]

8.5 Wide-product lane and source products
For the paper's 48×47-bit products, explicitly account for the four partial-product passes and the
full 95-bit recombination before final rounding. Validate the signs of every partial product and guard
bits. Use controlled operator sharing and check the synthesized netlist against the allocation;
memory banks and recurrence dependencies can dominate the intended II.

After the source list is ready, submit the translation/value/check GEMMs to AIE as one priority unit.
Feed the returned products into the remaining checks, consensus and second hop. Preserve the pre-
solve Phi dependency identified in section 2. Finalize whether the lane pool is shared among
linkage/gate/exchange phases and account for all resource overlap; do not simply add 64 linkage
lanes to 128 exchange lanes and still claim a 160-DSP total. [SPLICE-V Fig. 3; Prop. 4]

8.6 Work budgets and lane enabling
A work budget must be a deterministic function of logical operations and input, not FIFO contention,
Linux scheduling or processor clocks. Generate work-charge tables and branch points for the
reference and PL. Record actual cycles, stream waits and deadlines in separate telemetry. Hardware
watchdogs detect malfunction but must not silently produce a different valid inference branch.

Instantiate the validated maximum lane count once. An active-lane mask selects 32/64/128 without
changing per-row operation order. The energy controller applies the paper's objective using
measured parameters and the declared constraint g2(n)+w <= A3a. It must also check that the early
gate reaches the forecast by its dependency deadline; minimizing exchange energy alone cannot
authorize a late mask. Treat inactive lanes as clock-enabled/disabled resources unless board-
qualified power control proves more; do not promise independent voltage-rail power gating from
Python. [SPLICE-V Prop. 4, Table 2]

9. Schedule and compact interface ABI

9.1 Required phase ordering
 Phase         AIE work                                      PL work / dependency

 A1            Backbone, ownership heads, walks,             Reserve at H0; receive P4/window data and anchor
               bands, anchor hypotheses                      hypotheses.

 A2a           Top-down path                                 Linkage runs concurrently; publish IDs/order at H2.

 A2b, first    Restriction and per-fragment heads after      Fuse hypotheses and compute G; G must match the IDs'
 part          H2                                            frame generation.

 A2b,          Gated forecast after G is complete            Exchange consumes the same G and H3 state.
 second
 part

 A3a           Forecast synthesis, bottom-up path, box       Posterior/trust; issue H3' source list for priority AIE GEMMs;
               branch                                        receive H3''; complete exchange/H4.

 A3b           Class band, int16 late residual projection/   Candidate-list merge, checks and completion.
               fill, exact gate/candidates

No exact residual is an input to A3a. No forecast may start with an absent or stale G. Do not defer
anchor hypotheses until H3 because that would put the early gate behind its intended consumer.
Do not allow Python to sit between H3' and the priority GEMM. [SPLICE-V Fig. 4; Props. 4–6]

9.2 Boundary payloads to define byte-for-byte
 Label    Direction     Payload from Figure 3                             Displayed KB

 H0       AIE → PL      P3 eligibility/thin-structure data                        12.8

 H1       AIE → PL      eligibility, thin cue, keys, logits, hypotheses          139.7

 P        AIE → PL      P4 window scores                                          19.2

 H2       PL → AIE      per-cell u16 membership IDs and order                     25.0

 G        PL → AIE      admissible mask bits                                       8.2

 H3       AIE → PL      H, band descriptor, head outputs                         100.0

 H3'      PL → AIE      selected-source lists                                      4.1

 H3''     AIE → PL      translated values and checks                             106.5

 H4       PL → AIE      delta-V, boxes, xi                                        20.0

 H5       AIE → PL      compact candidates                                       100.0

These rounded labels total 535.5 KB. The transport must account for alignment, headers, masks
and legal shape-dependent lengths. A four-lane 64-bit interface at 400 MHz has 12.8 GB/s
aggregate ideal bandwidth per direction; the quoted serialization times do not include every routing
or scheduling stall. The extra four weight streams are separate resources. [SPLICE-V Fig. 3]

9.3 Proposed control descriptor
The paper lists descriptor concepts, not a complete binary ABI. A recommended descriptor contains
protocol version, manifest hash fragment, frame/generation ID, slot ID, phase/unit type, layer/band,
legal partition ID, active lanes, input/output shape and byte counts, exponent IDs, buffer offsets,
dependency tokens and work-budget ID.

Specify little-endian byte packing in the manifest; do not DMA a native C++ struct whose padding
has not been fixed. A protocol test must round-trip the same bytes through Python, HLS and AIE
parsers. Reject mismatched manifest versions and lengths before executing a frame.

A buffer transitions through allocated → inputs-ready → issued → computed → DMA-drained → all-
consumers-finished → reusable. Credits return only after both transfers and all consumers finish.
Use generation IDs to reject late results; never clear a slot merely because the producer has
finished computing. Priority-unit preemption occurs only at validated unit boundaries, not inside an
arbitrary kernel instruction sequence. [SPLICE-V shared-workers description and Fig. 3; proposed ABI
details]

10. Memory, clocks and feasibility gates
The paper budgets 7.38 MB of frame storage against approximately 8.91 MB of memory-tile
capacity. Seventeen 512-KiB tiles give 8,912,896 bytes in total, but a total-byte sum does not prove
a placeable allocation. Produce a per-tile allocation with lifetimes, DMA ownership, lock IDs,
alignment, halos, descriptors, compiler-reserved areas and shared buffers. Count retained C3–C5/
top-down outputs and the active working set together. [SPLICE-V §3, memory and streams]

Begin at the defined 640×640 batch-one operating point. The paper says its 704² slot exceeds
available memory and its 1,024² case is outside the on-chip regime. Treat larger shapes as separate
configurations requiring renewed allocation and performance checks, not a runtime shape
argument to the same proven layout.

The accelerator-RAM feeder uses two 256-bit ports in the model: at 300 MHz their aggregate ideal
throughput is 19.2 GB/s; one is 9.6 GB/s, below the modeled 12.8-GB/s four-stream weight intake.
Confirm that the real platform exposes the required ports and clocks. Accelerator RAM removes
DDR traffic but does not remove tile intake or local-bank contention. [SPLICE-V p. 9]

The intended rates are AIE 1 GHz, decision PL 300 MHz and PLIO 400 MHz. Check the exact device/
speed-grade operating limits and platform clocks. Implement proper clock-domain crossings and
resets; never infer safety from a five-nanosecond HLS compile used in a different project. A slower
first build is a valid debug configuration, but it is not evidence for the paper's timing model.

11. Python PS runtime and system build

11.1 PS responsibilities
Keep the PS application in Python: deployment manifest validation; XCLBIN loading; graph setup;
buffer allocation; model/descriptor initialization; frame submission; bounded completion waits;
output collection; evidence logging; power sampling coordination and offline metrics.

Keep per-fragment posterior decisions, trust iterations, descriptor dispatch and microsecond
scheduling out of Python. One high-level submission may trigger the complete frame schedule. Use
preallocated reusable buffers and frame IDs; the PL sequencer performs intra-frame control. Python
overhead must be timed separately from on-device execution. [SPLICE-V tool flow, Fig. 3; engineering
recommendation]

Do not assume package availability from an old Linux image. Verify the installed binding's device,
BO, kernel and graph APIs, the driver and firmware versions, user permissions and XCLBIN
metadata. Derive kernel names, arguments, memory groups and graph name from the built release
rather than hardcoded names copied from this document. XRT documents host-controlled kernels,
buffer synchronization and graph execution, but the deployed Python build must be tested. [D5]

11.2 Proposed runtime modules
ps/runtime.py manages the loaded device, graph lifetime and frame state. ps/buffers.py handles
allocation and synchronization. ps/manifest.py validates hashes and types. ps/submit.py creates frame
commands. ps/telemetry.py parses counted branches and timing. ps/measure.py records energy and
timestamps. ps/evaluate.py runs accuracy/conformance datasets. These are work-package names;
the pack does not claim these deployed modules already exist.

For the one-worker smoke, arm the output sink before starting the source so finite output FIFOs
cannot prevent progress. For a persistent dispatcher graph, start the validated graph lifecycle once,
not once per fragment. Use acknowledgments and timeouts for frame boundaries. Resetting the
whole AIE array can invalidate the loaded configuration; follow the qualified runtime reset sequence
instead of using resets as a generic retry. [D5]

11.3 Verified tool-flow pattern, not a fictitious ready build
The official flow compiles AIE C++ to libadf.a, PL HLS C++ to kernel objects, links them with an
extensible platform to a fixed system, then packages the system for the selected boot/runtime
method. AMD documents terminal-driven flows and both Vitis-linker and Vivado-export variants.
Choose one for this project and record it. [D6–D8]

The following are command templates after source files and a qualified XPFM exist. They
are not commands that can compile the planning pack as-is:

  v++ -c --mode aie --target hw --platform "$XPFM" \
      --config aie/aiecompiler.cfg aie/graph.cpp

  v++ -c --mode hls --config pl/configs/early_gate.cfg \
      --work_dir build/hls/early_gate

  v++ --link --target hw --platform "$XPFM" \
      --config platform/connectivity.cfg \
      build/aie/libadf.a build/pl/*.xo -o build/system/splicev.xsa

  v++ --package --target hw --platform "$XPFM" \
      --config platform/package.cfg \
      build/system/splicev.xsa build/aie/libadf.a \
      -o artifacts/release/splicev.xclbin

Pin actual output paths and object lists; do not assume the AIE archive lands under build/aie unless
the config explicitly sets it. Generate the HLS files, package format, port names, clocks, bank
assignments and software packaging options for the pinned tool release. Before a full build, make
the one-kernel smoke exercise these exact compile/link/package commands. The commands alone
cannot supply a missing TE0950 platform. [D6–D8]

11.4 HLS and simulation discipline
Use the release's documented HLS config or batch commands. Legacy vitis_hls absence does not
imply HLS is missing; check the supported vitis-run/v++ flow. For Tcl component creation, distinguish
legal component names from filesystem locations. Register all translation units and binary test
data, and verify the actual compile/link lines.

Stage test vectors with add_files -tb or use a runtime environment path. Avoid embedding absolute
paths into a quoted -D macro across Tcl, make and the compiler. Capture the producer exit code
immediately after a tee pipeline; a successfully created log is not a successful build.

Start RTL co-simulation with one complete transaction and bounded simulation progress. Then test
cached/repeated transactions, resets and backpressure without losing the original evidence. No-
wave or limited-trace simulation is appropriate for normal regressions; enable targeted tracing for
diagnosis. A progress counter of 1/2 is not a numerical pass, and an elapsed wall-clock duration
cannot identify a simulator bug. Check handshake/data availability and retain the failed trace before
changing the test.

12. Stage-gated implementation roadmap
The stages below specify dependencies and acceptance evidence rather than optimistic calendar
durations. Independent platform and reference work can proceed in parallel. Do not combine the
first test of a new algorithm with the first test of a new platform.

 Gat    Work                                   Required exit evidence
 e

 G00    Archive current folder and freeze      Source/checkpoint inventory; unchanged older project; requirements and
        SPLICE-V identity                           unresolved list.

 G01    Host and board platform                Exact device/tool versions; candidate XPFM; Linux/XRT API report; no
        inventory                              guessed platform.

 G02    One-worker AIE + PL movers +           Hardware bytes exact, repeated frames/reboot, bounded waits.
        Python

 G03    Complete manifest and reference        No deployment-critical nulls; generated ABI/LUTs; arithmetic/branch
        primitives                             fixtures exact.

 G04    Arm-(e) model, calibration and         Hash-bound trained models, DFP32/Db/M comparisons, early hypotheses
        integer export                         and Phi dependency closed.

 G05    One-worker AIE numerical kernels       All walk/band/forecast/residual elements exact; program and data budgets
                                               fit.

 G06    PL linkage, gate, posterior and        Exact oracle cases; one-call RTL and repeat/reset tests; measured resource
        trust                                  schedules.

 G07    Masked fragment-loop integration       Same G in both engines; no circular dependency; exact source lists/delta-
                                               V/residual.

 G08    Multiworker partition and              Two-/four-/34-worker sequence; sixteen legal tuples documented; no
        memory plan                            overflow/deadlock.

 G09    Persistent schedule and priority       All H0–H5/G events correct; no r→box dependency; credits/generation
        products                               tests pass.

 G10    Native and enhanced full detector      Six-boundary parity on complete frames, cap cases and forced budget
                                               exits.

 G11    Timing/resource closure                Actual platform clocks; no critical DRCs; post-route allocation incl. shell
                                               and debug.

 G12    Accuracy, load and energy study        P1–P9 results with source tags, paired intervals, common loads, auditable
                                               rail samples.

 G13    Reproducible release                   One manifest/model/platform identity; clean-build logs; deploy procedure;
                                               checksum inventory.

G03/G04 can be developed beside G01/G02, but full hardware execution requires both tracks. The
earliest useful algorithmic milestone is a small exact vertical slice: toy features → walk/bands →
memberships → early mask → gated forecast → PL trust → late residual → class output. Precomputed
toy memberships are acceptable for the first slice only when clearly labeled; full SPLICE
conformance requires replacing them with certified linkage.

13. Test plan mapped to the manuscript

13.1 P1 internal conformance
Compare the six required boundaries on identical frame input, calibrated tables, parameters and
counted-work branch trace. For each, save expected/actual sizes, hashes, mismatch counts and
first mismatch location. Add exact forecast/residual traces so a class-logit mismatch can be
localized.

Required cases include ordinary validation frames, empty admissibility, F cap, N cap, tie-rich
linkage, duplicate-unit extremes, forced linkage budget exits, QP iteration cap, clamp hits, every
lane enable count and legal partition changes. P1 is empirical execution conformance, not
something established by “same formulas” or a generated header alone. [SPLICE-V §4; Table 6]

13.2 P2 and P7 accuracy separation
P2 compares DFP32 to M while retaining Db as an intermediate diagnostic. The paper's one-sided
non-inferiority margins are AP greater than -0.1 and MR−2 less than +0.2. P7 compares Dpub to
DFP32 with margins -0.3 AP and +0.5 MR−2. Evaluate the correct direction for each metric.

Use the paper's image-paired bootstrap plan with 10,000 replicates, the declared three-seed/
calibration selection rule, seed spread, and the family correction for twelve tests. Do not choose the
best seed on the final test set. Reproduce descriptor and forecast arms separately; P1 pass cannot
imply P2/P7 pass. [SPLICE-V §4, Tables 6–7]

13.3 P3, P4, P6 and P8 timing
P3 uses the same detector and threshold for the T4 and TE0950 comparison. Record both
deployment threshold and evaluation-threshold regimes; skipping may disappear at the latter. P4
compares predicted versus actual frame latency with p99 absolute error below 0.1 ms. P6 checks
the observed throughput/EDP knee against the model. P8 measures movement ceilings against
attained per-layer efficiency, including exceptions rather than hiding them in an overall average.
[SPLICE-V Tables 5–6]

Instrument A1/A2a/A2b/A3a/A3b, linkage/gate/exchange work, priority wait, H0–H5/G byte counts
and timestamps, input/output stalls, NoC traffic and each frame's enabled lanes. Clock-domain
counters need a common time conversion and documented synchronization. Report CPU wall-clock
submission/completion latency separately from hardware stage time.

The paper's 2.88/2.55-ms cap latencies and 351/397-frame/s knees are analytical operating points
in Table 3. Use them as hypotheses to test on this board, not hardcoded timers or status values. If
the early mask is not hidden or g1/g2 exceed their AIE intervals, compute and report the exposed
gap. [SPLICE-V Prop. 6; Table 3]

13.4 P5 energy and P9 recall
P5's revised comparison is at 270 frames/s, not each system's own best throughput. The
manuscript gives 47.3 versus 47.2 mJ/frame and a criterion no worse than 1.05× native. Measure
mean energy over completed frames from synchronized power samples:

  E_frame = integral(P(t) dt) / completed_frame_count

Record whole-board versus SoC-rail scope, idle baseline, sampling rate, clock gating, sensor
calibration, thermal conditions and dropped/rejected frames. The paper names a Tria SOM; TE0950
power monitors and regulator exposure must be inventoried independently. If the needed rails are
unavailable, use a calibrated external measurement and state its scope. A T4 TDP envelope is not
measured GPU energy. [SPLICE-V §4; Tables 3, 5–6]

P9 requires heavy-occlusion recall loss no larger than 0.5 percentage point. Test the detector's gate
effect separately from exact skipping. Do not discard difficult frames, cap hits, aborted rows or
thermal throttling from denominators.

14. Baselines, optimization and resource choices
Preserve the paper's four-way comparison: native on PL (N-P), enhanced on PL (E-P), native on
AIE+PL (N-A), enhanced on AIE+PL (E-A). The DSP58-only analytical comparator uses 432 DSPs for
its configurable arithmetic pool and reserves 32; it is not a claim that all 464 slices are available
exclusively for convolution. A deployable PL-only build is a separate implementation and must be
measured as such. [SPLICE-V Table 3 and §4]

Evaluate the same model/manifest/threshold at a common offered load before attributing gains to a
device. Separate the cost of enhancement from the advantage of AIE. Keep the native graph/
placement comparable and document unavoidable differences.

Optimize in this order: correctness and ABI; first-worker program fit; legal placement and bank
conflicts; movement-aware partitions; removal of repeated DDR activation traffic; priority
scheduling; counted budgets; resource scaling; lane enabling; accelerator-RAM feeder; then low-
load power management. Further widening a hidden PL stage cannot improve end-to-end latency
once AIE is the critical resource.

At each optimization, retain before/after conformance and mechanism counters. Increasing kernel
efficiency can expose a formerly hidden PL gap; a faster AIE kernel does not prove the total latency
follows the old ideal formula. The paper already recognizes this above its stated hiding range. [SPLICE-V
Props. 4, 6; Tables 2, 4]

15. First commands and immediate next decision

15.1 Install without overwriting earlier work
Download SPLICEV_Implementation_Pack.zip to ~/Downloads. Extract it there; install through its
integrity-checking Python utility. No shell script executable permission is needed.

  unzip -q -n "$HOME/Downloads/SPLICEV_Implementation_Pack.zip" \
      -d "$HOME/Downloads"

  python3 "$HOME/Downloads/SPLICEV_Implementation_Pack/tools/install_workspace.py" \
      --root ~/projects

  cd ~/projects/splicev
  python3 -m unittest discover -s tests -v

The included 23 tests cover host arithmetic/structural invariants and formula calculations. They do
not execute HLS, AIE, model training or hardware.

  python3 tools/audit_paper_numbers.py
  python3 tools/check_contract.py
  python3 tools/generate_paper_constants.py

The contract remains intentionally non-deployable. Listing missing fields is useful evidence, not an
installation error. Generated paper constants do not close those fields.

15.2 Capture the host and board inventories
From the desktop, source the existing 2025.2 tools for the preferred-track inventory:

  source /opt/AMD/2025.2/Vitis/settings64.sh
  python3 tools/preflight.py --target host \
      --output evidence/preflight/host.json

This does not upgrade a platform or change the board. Review all paths and versions. If the
expected settings file is absent on the current host, locate the actual installed version rather than
creating a fake path.

When the board is booted into its working Linux image, copy the same read-only script and run it
there:

  scp tools/preflight.py petalinux@10.161.0.15:/home/petalinux/splicev_preflight.py

  ssh petalinux@10.161.0.15 \
    'python3 /home/petalinux/splicev_preflight.py --target board --output /home/petalinux/
          splicev_board.json'

  scp petalinux@10.161.0.15:/home/petalinux/splicev_board.json \
      evidence/preflight/board.json

The address is the earlier working project address, not newly verified network state. In JTAG-only
boot there may be no Linux or SSH: restore the board's recorded Linux boot configuration safely, or
use the serial console for inventory. Do not program a full JTAG PDI while relying on the live Linux
SSH session.

15.3 Next action after inventory
Approve a platform track only when the actual XPFM and Linux/XRT combination is known. Then
build the one-AIE/PL/Python hardware smoke. In parallel, import the exact arm-(e) model and
complete the manifest/reference. Do not start a full 34-worker or 128-lane hardware build from this
planning pack.

16. Release acceptance and troubleshooting
rules
For every stage: identify the last operation executed successfully, preserve its inputs and output
logs, classify the failure, and change only the responsible boundary. Syntax, path, linkage,
simulator, protocol, numerical, model and platform failures are different categories.

 Symptom                        Correct response

 HLS path or component-         Repair the harness; no arithmetic conclusion has been reached.
 name error

 Missing vector file /          Stage data or use a runtime path; do not regenerate goldens.
 broken -D quoting

 CSim passes, RTL does          Inspect protocol/handshake and finite buffers; isolate one transaction; do not declare an
 not finish                     HLS arithmetic defect yet.

 CSim/RTL pass, device          Compare exact release hashes, packing, addresses, cache handling, control registers and
 differs                        timing/clock-domain behavior. Stale IP is one candidate, not the only possible cause.

 Both paths return the          Compare mismatch positions/values; equal counts alone do not prove identical
 same mismatch count            deterministic errors.

 mrd is blocked                 Validate the real map before using any force option; a generated address does not prove
                                that every downstream reset/clock is healthy.

 hw_server port is busy         Reuse the known listener; do not repeatedly start a second process.

 AIE code or buffers            Reduce specialization/unrolling or change a documented mapping; do not hide the
 exceed capacity                deviation from the shared-worker model.

 Mask or residual belongs       Stop the frame and report a protocol error; do not proceed with stale state.
 to an old frame

 A budget is hit                Execute the declared reference-matched branch and count it; do not omit the frame from
                                metrics.

The final release gate requires all six boundary comparisons, negative tests and branch traces,
repeat/reset testing, timing/resource reports, a bounded runtime, model/manifest consistency and
actual measurement provenance. A routed image or 23 passing host tests is only a partial
milestone.

Recommended next milestone: exact, repeatable Python → PL → AIE → PL → Python loopback on
a qualified TE0950 platform, alongside a complete arm-(e) integer-reference contract. Those two
foundations make subsequent failures local and prevent the long chain of unrelated rebuilds from
the earlier project.

Sources and reading order
[SPLICE-V] SPLICE-V.pdf, SPLICE-V: A Representation-Guided Architecture for Graph-
Wavelet Object Detection on Versal, 12 pages. Main implementation anchors: §2 pp. 2–3; Definition
1 pp. 4–5; Fig. 3 p. 5; Lemma 1 p. 6; linkage/gate/exchange p. 7; Lemma 3/Props. 5–6 p. 8; §4 pp. 9–
11; Tables 3–7 pp. 10–11. This manuscript is the primary method source.

[D1] Trenz Electronic, TE0950 Test Board. Board-specific reference-design identity and release
context. https://wiki.trenz-electronic.de/display/PD/TE0950+Test+Board

[D2] Trenz Electronic, TE0950 Extensible Platform. Board-platform construction and AIE/PL
integration example; its documented tool version must not be silently treated as 2025.2. https://
wiki.trenz-electronic.de/display/PD/TE0950+Extensible+Platform

[D3] AMD, AIE-ML Memory Module, AM020. Local-bank organization and DMA/lock hardware.
https://docs.amd.com/r/en-US/am020-versal-aie-ml/AIE-ML-Memory-Module

[D4] AMD, AI Engine Data Memory, UG1603, 2025.2. Program/data memory limits, banks and
neighborhood access. https://docs.amd.com/r/2025.2-English/ug1603-ai-engine-ml-kernel-graph/AI-
Engine-Data-Memory

[D5] AMD/Xilinx, XRT Native APIs, 2025.2. Runtime buffer, kernel and graph lifecycle; Python
availability must be checked in the installed build. https://xilinx.github.io/XRT/2025.2/html/
xrt_native_apis.html

[D6] AMD, Compiling using v++ (Unified Compiler), UG1076, 2025.2; and v++ Mode HLS, UG1702,
2025.2. https://docs.amd.com/r/2025.2-English/ug1076-ai-engine-environment/Compiling-using-v-
Unified-Compiler and https://docs.amd.com/r/2025.2-English/ug1702-vitis-accelerated-reference/v-
Mode-HLS

[D7] AMD, Versal Custom Thin Platform Extensible System, XD324, 2025.2. End-to-end terminal
build pattern, not a TE0950 board drop-in. https://docs.amd.com/r/2025.2-English/Vitis-Tutorials-
Vitis-System-Design/Versal-Custom-Thin-Platform-Extensible-System

[D8] AMD, Design Planning Considerations for AI Engine and Programmable Logic Integration,
UG1387, 2025.2. Archive/object integration, fixed platform and packaging. https://docs.amd.com/r/
en-US/ug1387-acap-hardware-ip-platform-dev-methodology/Design-Planning-Considerations-for-AI-
Engine-and-Programmable-Logic-Integration

External references were checked for implementation tooling. They do not supply missing SPLICE
checkpoints, calibration tables or evidence for the manuscript's numerical results.
```
