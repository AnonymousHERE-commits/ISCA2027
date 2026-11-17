# Evidence

Immutable JSON reports of executed checks (roadmap section 4).  One directory per gate
(`G02/`, `G05/`, ...) plus `preflight/` and `audit/`.  Every report carries the release identity
(manifest hash, git commit, xclbin hash), the mode (host-orchestrated / device-scheduled), the
backend and a provenance tag: S simulation/emulation, R post-route report, T training, M board
measurement.  Files are never edited after they are written; a new run writes a new file.

Board runs write here by default (`ps/apps/*.py --backend board`); emulated or reference runs
write only when `--evidence-dir` is given and are labelled S.
