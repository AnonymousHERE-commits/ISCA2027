# Build and run

Paths follow the roadmap: desktop workspace `~/projects/splicev`,
Vitis 2025.2 at `/opt/AMD/2025.2`, board `petalinux@10.161.0.15` (the address
of the earlier project; verify it).  Commands are run from the repository root.

## 0. Get the repository

```bash
git clone https://github.com/AnonymousHERE-commits/ISCA2027.git ~/projects/splicev
cd ~/projects/splicev
```

Requirements on the desktop: Python 3.9+ with `numpy`, GNU make, g++ (C++17, x86-64 for
`__int128`).  The paper PDF is not part of the repository; keep it in `paper/` locally.

## 1. Host checks (no AMD tools)

```bash
make test             # Python reference, exact oracles, generated sources, PS paths (emulated device)
make check-generated  # generated/ matches contracts/manifest.dev.json
make host-cpp         # PL cores + kernel tops, AIE dispatcher, generated graphs: bit-exact vs reference
make emulate          # the four board applications on the emulated device
make contract         # what is still synthetic / undecided in the manifest
```

All of these must pass before any tool or board time is spent.  `make help` lists every target.

The detector is trained separately on a GPU cluster (`detector/hpc/README.md`); its export
(`artifacts/trained/<run>/`) feeds the manifest.  The board steps below run with the
dev-synthetic profile until then.

## 2. Inventory (G01)

```bash
source /opt/AMD/2025.2/Vitis/settings64.sh
make preflight-host                                 # -> evidence/preflight/host.json

scp tools/preflight.py petalinux@10.161.0.15:/home/petalinux/splicev_preflight.py
ssh petalinux@10.161.0.15 'python3 /home/petalinux/splicev_preflight.py --target board --output /home/petalinux/splicev_board.json'
scp petalinux@10.161.0.15:/home/petalinux/splicev_board.json evidence/preflight/board.json
```

Both scripts are read-only.  If the settings file is elsewhere, locate the installed version
instead of creating the path.

## 3. Platform

Follow `platform/README.md` to produce the TE0950 extensible platform and its Linux image.
From here on `PLATFORM` is the path of that `.xpfm`:

```bash
export PLATFORM=/path/to/te0950_splicev.xpfm
python3 tools/platform_report.py --xpfm $PLATFORM --board-json evidence/preflight/board.json
```

## 4. Optional simulations before hardware

```bash
# HLS C simulation of each decision engine with the toy-slice capacities (uses ap_int<128>)
for k in sv_reserve sv_linkage sv_gate sv_exchange1 sv_exchange2; do make hls-csim KERNEL=$k CAPS=small; done

# AIE worker graph in the x86 simulator, every unit suite
make aie     AIE_SET=uniform1 AIE_TARGET=x86sim
make aie-sim AIE_SET=uniform1 AIE_TARGET=x86sim SUITE=basic
make aie-sim AIE_SET=uniform1 AIE_TARGET=x86sim SUITE=toy0
```

Without `PLATFORM` the x86sim compile passes `--part` (VE2302); if your `v++` insists on a
platform for `--mode aie`, any AIE-ML base platform of the installation is acceptable for
x86sim only, because the simulated function does not depend on the board.
`aie-sim` writes `data/w<i>_in.txt` (units routed by worker class and padded with NOP units),
runs the simulator with `SV_SIM_UNITS` from `units.json`, and compares every output word with
the reference.  `AIE_TARGET=hw make aie-sim ...` uses `aiesimulator` on the hardware build.

## 5. First hardware test: smoke (G02)

```bash
make system SYSTEM=smoke PLATFORM=$PLATFORM      # movers + smoke graph, link, package, release
make bundle SYSTEM=smoke                          # build/bundle/splicev_smoke.tar.gz
```

`make system` = `hls-system` + `aie` + `link` + `package` + `release`; each step logs to
`logs/` and fails on the tool's exit status.  `release` reads the compute-unit names back from
the xclbin (`xclbinutil`) and refuses a build that lacks an expected instance.

**Deploy.**  On Versal without DFX the PL+AIE design is in the boot PDI: copy the `BOOT.BIN`
produced under `build/system/smoke/package/` (and any other files the package step lists for
the FAT partition) to the board's boot partition, keep the kernel/root file system that belong
to the platform, and reboot.  Then copy the bundle:

```bash
scp build/bundle/splicev_smoke.tar.gz petalinux@10.161.0.15:/home/petalinux/
ssh petalinux@10.161.0.15
mkdir -p ~/splicev && tar -xzf ~/splicev_smoke.tar.gz -C ~/splicev && cd ~/splicev
python3 tools/probe_pyxrt.py --xclbin artifacts/release/smoke/splicev.xclbin --graph smoke --kernels "sv_mm2s:{mm2s_0}" "sv_s2mm:{s2mm_0}"
python3 ps/apps/g02_smoke.py --release artifacts/release/smoke/release.json --repeat 5
```

The board needs `python3` with `numpy` and the XRT Python binding.  If the probe reports no
usable `pyxrt.graph`, either build the C++ helper on the board (`make -C ps/native`, needs the
XRT headers) or rebuild with `AIE_AUTOSTART=1` (graph runs from PDI load) and pass
`--graph-mode autostart`.  Repeat the G02 run after a reboot.  Evidence goes to
`evidence/G02/`; copy it back to the desktop repository.

## 6. Toy vertical slice on the board (G05, G06, G07)

```bash
make system SYSTEM=slice1 PLATFORM=$PLATFORM      # 1 full worker + movers + 5 PL engines (small caps)
make bundle SYSTEM=slice1
```

If the AIE compiler reports that the worker does not fit one tile's program memory, use
`SYSTEM=slice3` (three worker classes, `aie/worker_classes.json`); the PS routes units by the
`workers.json` recorded in the release.  After deploying as in step 5:

```bash
python3 ps/apps/g05_units.py   --release artifacts/release/slice1/release.json --suite basic toy0 toy1 --repeat 3
python3 ps/apps/g06_engines.py --release artifacts/release/slice1/release.json --repeat 3
python3 ps/apps/g07_slice.py   --release artifacts/release/slice1/release.json --seeds 0 1 2 --repeat 3
```

Each app exits 0 only when every compared word matches; the JSON evidence lists mismatch
counts, first mismatch positions, boundary hashes and bounded wait times.  `--backend emulated`
runs the same host code on the emulated device; it never writes board evidence.

## 7. Useful variables

| variable | default | meaning |
|---|---|---|
| `PLATFORM` | - | TE0950 `.xpfm` (required for hardware targets) |
| `SYSTEM` | `slice1` | `smoke`, `slice1`, `slice3` (`pl/kernels.json`) |
| `KERNEL`, `CAPS` | `sv_gate`, system's | HLS kernel and capacity set (`small`/`full`) |
| `AIE_SET`, `AIE_TARGET` | system's, `hw` | worker-class set; `hw` or `x86sim` |
| `AIE_PL_FREQ` | 300 | PLIO clock (MHz) given to the AIE compiler |
| `AIE_AUTOSTART` | 0 | 1 = package without `defer_aie_run` |
| `CLOCK_ID` | - | platform clock id for every kernel (link) |
| `ROOTFS`, `KERNEL_IMAGE` | - | Linux artifacts for a full SD image (else `no_image`) |
| `AIE_FLAGS`, `VPP_LINK_FLAGS`, `VPP_PKG_FLAGS` | - | extra tool options |

The tool command lines follow the 2025.2 unified `v++`/`vitis-run` flow (roadmap 11.3).  If
your installation rejects an option spelling, change it in the `Makefile` recipe (one place
per tool) and record the change; do not change the sources to work around a tool option.

## 8. Troubleshooting (roadmap 16, applied to this repository)

| symptom | response |
|---|---|
| `make` stops: `PLATFORM must name ...` | no qualified platform yet; finish `platform/README.md` |
| HLS path or component error | harness problem; no arithmetic conclusion yet.  Check the rendered `build/hls/<caps>/<k>/hls_config.cfg` |
| csim cannot read vectors | `make vectors`; the testbench reads `$SPLICEV_VECTORS/engines/pl_kernels.svb` at run time |
| csim passes, cosim does not finish | protocol/handshake or finite buffers; isolate one transaction, keep the failed trace |
| AIE program memory overflow | build a split worker-class set (`split3`, `split6`); do not drop bodies silently |
| simulator output differs | `tools/aie_sim_data.py check` prints the first differing word; compare with `make host-graphs`, which runs the same kernels on the host |
| app: `release manifest ... != host manifest` | the board bundle and the build come from different manifests; rebuild or re-bundle |
| app: `run ended in state TIMEOUT` | the sink waited for more words than arrived: wrong worker class, graph not running (`--graph-mode`), or PLIO/connectivity mismatch; check `probe_pyxrt` and the release's `xclbin_compute_units` |
| unit status `BAD_OPCODE` | the unit reached a worker class without that body: routing and `workers.json` disagree |
| `corrupt response frame` / `stale` | protocol error: stop, keep the evidence, do not retry blindly; reload the image before the next run |
| mask or residual from an old frame | stop the frame and report a protocol error (roadmap 16) |
| a linkage budget is hit | expected branch: it is executed and counted (`linkage.stats`), not dropped |
