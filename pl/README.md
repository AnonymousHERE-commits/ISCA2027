# PL (Vitis HLS C++)

| path | content |
|---|---|
| `include/` | bit-exact engine cores: `sv_int.h` (RNE, 128-bit containers), `sv_bf24.h`, `sv_reserve.h`, `sv_linkage.h`, `sv_gate.h`, `sv_posterior.h`, `sv_trust.h` |
| `engines/` | kernel tops with the word ABI of `reference/splicev_ref/pl_abi.py`: `sv_reserve`, `sv_linkage`, `sv_gate`, `sv_exchange1`, `sv_exchange2`; `sv_mm2s`/`sv_s2mm` movers |
| `tb/` | C-simulation testbenches (`tb_cores.cpp`, `tb_kernels.cpp`) reading `$SPLICEV_VECTORS` |
| `kernels.json` | kernel table, capacity sets (`full`, `small`) and systems (`smoke`, `slice1`, `slice3`) |
| `configs/hls.cfg.in` | HLS component configuration template (`tools/build_cfg.py hls`) |

Host builds use `__int128`; HLS builds define `SPLICEV_USE_AP_INT` and use `ap_int<128>`.  Both
hold the same integers, so the results are identical; the HLS C simulation is the check.
