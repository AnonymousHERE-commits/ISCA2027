# Work packages still open

Ordered by dependency.  Each package lists its inputs, outputs and the evidence that closes it.

## WP-1 Platform (G01, G02) - board owner
- **In**: Trenz TE0950 reference design for the qualified tool version, Vitis/Vivado 2025.2,
  PetaLinux/EDF image with XRT + zocl + pyxrt.
- **Do**: `platform/README.md` steps 1-6; then `make system SYSTEM=smoke PLATFORM=...`, deploy,
  `ps/apps/g02_smoke.py --repeat 5`, reboot, repeat.
- **Out**: `evidence/preflight/{host,board,platform_report,pyxrt_probe}.json`, `evidence/G02/smoke_*.json`.

## WP-2 AIE fit and simulation (G05)
- `make aie AIE_SET=uniform1 AIE_TARGET=x86sim`, `make aie-sim ... SUITE=basic|toy0|toy1`.
- `make aie AIE_SET=uniform1 PLATFORM=...`: read the program-memory report.  If one tile cannot
  hold all eight bodies, build `split3` (or `split6`) - the PS routes by `workers.json`, the
  boundaries are unchanged (`tests/test_slice_runner.py`).
- Then `aiesimulator` (`AIE_TARGET=hw make aie-sim`) and the board run `ps/apps/g05_units.py`.
- Vectorised AIE API paths for GEMM/CONV/STENCIL must reproduce the scalar outputs exactly.

## WP-3 HLS engines (G06)
- `make hls-csim KERNEL=k CAPS=small` for sv_reserve, sv_linkage, sv_gate, sv_exchange1/2
  (exercises the `ap_int<128>` path that the host g++ build does not), then `make hls` and
  `make hls-cosim` with one complete transaction first (roadmap 11.4).
- Record II/latency/resources; the 128-bit dividers of the trust solver are the first
  optimisation target once results are exact.
- Memory: `sv_linkage` holds exact-capacity neighbour lists (254 entries per anchor, ~4.5 KB):
  the `small` set (192 anchors) needs ~0.86 MB and must be bound to UltraRAM; the `full` set
  (1856 anchors, ~8.3 MB) cannot fit the VE2302.  Before G10 replace the per-anchor lists by a
  pooled list store bounded by 2E entries (same merge order, same overflow points), and prove it
  against the reference with the existing vectors.

## WP-4 Slice on the board (G07)
- `make system SYSTEM=slice1` (or `slice3`), deploy, `ps/apps/g05_units.py`, `g06_engines.py`,
  `g07_slice.py --seeds 0 1 2 --repeat 3`.

## WP-5 Model and manifest (G03, G04) - model owner
- Train/export arm-(e); write `contracts/manifest.model.json` (extends the paper seed) that
  replaces every SYN-xx and confirms or replaces every DD-xx (`generated/DECISIONS.md`); close the
  three nulls (`posterior.duplicate_edge_bound`, `forecast.probe_distribution`,
  `partition.catalog`) and the Phi_f pre-solve dependency (roadmap 2.2).
- `python3 tools/generate_sources.py --manifest contracts/manifest.model.json --out generated`
  regenerates every consumer; descriptors carry the new MANIFEST_HASH16.

## WP-6 Scale-out and schedule (G08, G09)
- Packet-switched 34-worker graph on the four handoff PLIOs; partition catalogue (16 tuples).
- PL sequencer for the device-scheduled mode (A1/A2a/A2b/A3a/A3b with H0-H5/G events,
  credits and generation tags); the host-orchestrated steps listed in
  `ps/splicev_ps/slice_runner.HOST_STEPS` move to the device.

## WP-7 Full detector, closure and study (G10-G13)
- Six-boundary parity on complete frames, timing/resource closure, P1-P9 study, release.
