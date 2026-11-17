# Platform track (gates G01, G02)

SPLICE-V needs one qualified **extensible Vitis platform (XPFM) for the Trenz TE0950-03**
(VE2302, `xcve2302-sfva784-1LP-e-S`) together with a matching Linux image, XRT and Python
binding.  This repository does **not** contain such a platform and nothing here pretends
otherwise: every hardware target of the top-level `Makefile` stops with an error until
`PLATFORM=<path to the .xpfm>` names a real file.

Preferred track (roadmap 3.1): Vivado/Vitis 2025.2 with a regenerated TE0950 platform.  If a
2025.2 platform cannot be built yet, qualify the complete 2024.2 stack on a separate branch and
rebuild *all* HLS/AIE/platform artifacts there; never mix 2025.2 AIE objects, older PL IP and an
older root file system in one "release".

## What the platform must provide (interface contract)

| item | requirement for SPLICE-V |
|---|---|
| part | `xcve2302-sfva784-1LP-e-S` (TE0950-03) |
| AI Engine | AIE-ML array exported to the platform (34 tiles, 17x2), mem tiles enabled |
| PL clock | one clock usable by all kernels; 300 MHz is the design point (`pl/kernels.json`). Record its clock **id** (`make link CLOCK_ID=<id>`) |
| PLIO | 64-bit AXI4-Stream between PL and AIE: 2 per worker (in/out). Smoke: 2, slice1: 2, slice3: 6, final 34-worker system: see G08 |
| memory | NoC/DDR (LPDDR/DDR4 as on the board) reachable from kernel `m_axi` ports; record the bank names (`sp=` lines are optional, the linker may choose) |
| control | `s_axilite` control through the platform's PS master; interrupt controller for kernel completion |
| software | Linux with XRT (zocl) and a Python binding (`pyxrt`) whose **graph** API is checked by `tools/probe_pyxrt.py`; otherwise build `ps/native` (XRT C++ graph control) |
| boot | SD boot files produced by `v++ --package` (BOOT.BIN with the PL+AIE PDI) |

## Procedure (summary; follow the referenced sources for the exact GUI/Tcl steps)

1. **Inventory (G01)**: `make preflight-host` on the desktop, `tools/preflight.py --target board`
   on the board (see `docs/BUILD_AND_RUN.md`).  Keep both JSON files in `evidence/preflight/`.
2. **Hardware design**: start from Trenz's TE0950 reference design for the chosen tool
   version [D1, D2].  Keep CIPS, DDR timing, reference clocks, pin constraints, resets, boot
   support and board management.  Trim video/peripheral logic only on a branch, after the base
   design boots.
3. **Expose platform interfaces** with `te0950/pfm_properties.tcl` (edit the cell names at the
   top to the names in *your* block design), then `write_hw_platform` (hardware and, if wanted,
   hw_emu XSA).  The tutorial XD324 [D7] shows the same steps on AMD boards; do not copy its
   board files.
4. **Software components**: build the Linux image (PetaLinux or the Yocto/EDF flow you use) with
   XRT, zocl, `python3`, `python3-numpy` and the XRT Python binding; export the sysroot, `Image`
   and `rootfs.ext4` used by `v++ --package` (`ROOTFS=`, `KERNEL_IMAGE=`).
5. **Platform component**: `te0950/create_platform.py` (Vitis Python API) packages XSA + Linux
   domain into the `.xpfm`.
6. **Platform report**: `python3 tools/platform_report.py --xpfm <xpfm> --xsa <xsa>` writes
   `evidence/preflight/platform_report.json` (hashes, platforminfo output: clocks, memories,
   AIE, PLIO).  Read the tool-generated address map; never reuse the earlier project's addresses
   (0xA4090000, 0xA4140000, 0xB0000000).
7. **First hardware acceptance (G02)**: `make system SYSTEM=smoke PLATFORM=...`, deploy, then
   `python3 ps/apps/g02_smoke.py --release artifacts/release/smoke/release.json --repeat 5`, and again
   after a reboot.

Only after G02 passes is the platform considered qualified; that is a platform result, not a
SPLICE result.

## Files

| file | purpose |
|---|---|
| `te0950/pfm_properties.tcl` | template: PFM clock / AXI / IRQ / AIE properties + `write_hw_platform` |
| `te0950/create_platform.py` | template: Vitis platform component from the XSA and the Linux artifacts |
| `package.cfg.in` | `v++ --package` configuration rendered by `tools/build_cfg.py package` |
| `platform_report_template.json` | fields of `platform_report.json` (roadmap 3.2) |

The connectivity files are rendered per build from `pl/kernels.json` and the generated
`workers.json` (`tools/build_cfg.py connectivity`), so kernel instance and PLIO names always match
the compiled graph.
