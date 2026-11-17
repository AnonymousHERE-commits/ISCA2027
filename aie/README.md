# AIE-ML

| path | content |
|---|---|
| `kernels/sv_units.h` | the dispatcher: descriptor checks, eight unit bodies (CONV, GEMM, STENCIL, LUT, PERMUTE, SEGOPS, TAPMOD, HEAD), response framing; compiled for the AIE tile and for the host |
| `include/sv_aie_int.h` | integer helpers (RNE shifts, divisions, check words) |
| `templates/` | worker kernel, worker declarations and graph templates |
| `worker_classes.json` | worker-class sets (body masks per worker) |
| `smoke/` | G02 smoke kernel and graph (instance `smoke`) |
| `test/` | host conformance (`host_units.cpp`), mock ADF layer and host graph runners |

Generate a graph with `python3 tools/gen_aie_graph.py --set uniform1 --out build/aie/uniform1/src`
(the Makefile does this); the output `workers.json` lists every worker's class, kernel and PLIO
names and is copied into the release so the PS never hard-codes them.
