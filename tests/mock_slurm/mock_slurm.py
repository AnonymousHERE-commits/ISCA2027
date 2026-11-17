#!/usr/bin/env python3
"""A small stand-in for Slurm, used to test the launcher's job chain without a cluster.

Commands (each a file in this folder that calls this module): sbatch, scancel, scontrol,
squeue, sacct, sinfo, sacctmgr, srun, nvidia-smi.  `sbatch` only queues a job; `run_queue()` then plays the
scheduler: it runs the queued jobs one at a time in submission order, honouring
--dependency=afterok|afterany, --array, scancel and `scontrol requeue` (a requeued task is
run again with SLURM_RESTART_COUNT incremented).  State lives in $MOCK_SLURM_DIR.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def state() -> Path:
    d = Path(os.environ["MOCK_SLURM_DIR"])
    for sub in ("queue", "done", "requeue", "cancel"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def next_id(d: Path) -> int:
    f = d / "counter"
    n = int(f.read_text()) + 1 if f.exists() else 1000
    f.write_text(str(n))
    return n


def sbatch(argv) -> int:
    d = state()
    spec = {"array": None, "dependency": [], "export": {}, "name": "job", "cpus": "1"}
    i, script = 0, None
    test_only = False
    while i < len(argv):
        a = argv[i]
        if a == "--parsable" or a == "--requeue":
            pass
        elif a == "--test-only":
            test_only = True
        elif a.startswith("--time="):
            spec["time"] = a.split("=", 1)[1]
        elif a.startswith("--dependency="):
            for dep in a.split("=", 1)[1].split(","):
                kind, jid = dep.split(":")
                spec["dependency"].append([kind, jid])
        elif a.startswith("--array="):
            rng = a.split("=", 1)[1].split("%")[0]
            lo, hi = rng.split("-")
            spec["array"] = [int(lo), int(hi)]
        elif a.startswith("--export="):
            for kv in a.split("=", 1)[1].split(","):
                if kv != "ALL" and "=" in kv:
                    k, v = kv.split("=", 1)
                    spec["export"][k] = v
        elif a.startswith("--job-name="):
            spec["name"] = a.split("=", 1)[1]
        elif a.startswith("--cpus-per-task="):
            spec["cpus"] = a.split("=", 1)[1]
        elif a.startswith("--"):
            pass
        else:
            script = a
            spec["args"] = argv[i + 1:]
            break
        i += 1
    if script is None:
        print("mock sbatch: no script", file=sys.stderr)
        return 1
    if test_only:                                          # nothing is queued
        bad = os.environ.get("MOCK_SBATCH_REJECT")
        if bad:
            print(f"sbatch: error: Batch job submission failed: {bad}", file=sys.stderr)
            return 1
        print("sbatch: Job 4242 to start at 2026-01-01T00:00:00 using 4 processors on nodes mocknode in partition gpu", file=sys.stderr)
        return 0
    jid = next_id(d)
    spec.update(id=jid, script=script, env={k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "BASH_FUNC"))})
    (d / "queue" / f"{jid}.json").write_text(json.dumps(spec))
    print(jid)
    return 0


def scancel(argv) -> int:
    d = state()
    for a in argv:
        if a.isdigit():
            (d / "cancel" / a).write_text("1")
    return 0


def scontrol(argv) -> int:
    d = state()
    if argv[:1] == ["requeue"] and len(argv) > 1:
        if os.environ.get("MOCK_REQUEUE_DENIED") == "1":       # a site where users cannot requeue running jobs
            print("mock scontrol: requeue denied", file=sys.stderr)
            return 1
        (d / "requeue" / argv[1]).write_text("1")
    return 0


def run_one(d: Path, spec: dict, task) -> int:
    jid = spec["id"]
    target = f"{jid}_{task}" if task is not None else str(jid)
    restart = 0
    while True:
        env = dict(spec["env"])
        env.update(spec["export"])
        env.update(SLURM_JOB_ID=str(jid * 1000 + task) if task is not None else str(jid), SLURM_RESTART_COUNT=str(restart),
                   SLURMD_NODENAME="mocknode", SLURM_CPUS_PER_TASK=str(spec["cpus"]), SLURM_JOB_NAME=spec["name"])
        if task is not None:
            env.update(SLURM_ARRAY_JOB_ID=str(jid), SLURM_ARRAY_TASK_ID=str(task))
        r = subprocess.run(["bash", spec["script"], *spec.get("args", [])], env=env, capture_output=True, text=True)
        (d / "done" / f"{target}.r{restart}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        marker = d / "requeue" / target
        if marker.exists() and restart < 20:
            marker.unlink()
            restart += 1
            continue
        return r.returncode


def run_queue(limit: int = 500) -> dict:
    """Run every queued job (and the jobs they submit) to completion; returns {job id: record}."""
    d = state()
    records = {}
    for _ in range(limit):
        pending = sorted((int(p.stem), p) for p in (d / "queue").glob("*.json"))
        if not pending:
            break
        progressed = False
        for jid, path in pending:
            spec = json.loads(path.read_text())
            deps = spec["dependency"]
            if any(dep not in records for _, dep in deps):
                continue                                    # a dependency has not finished yet
            path.unlink()
            progressed = True
            rec = {"name": spec["name"], "state": "COMPLETED", "codes": [], "time": spec.get("time"), "cpus": spec["cpus"]}
            if (d / "cancel" / str(jid)).exists():
                rec["state"] = "CANCELLED"
            elif any(kind == "afterok" and not records[dep]["ok"] for kind, dep in deps):
                rec["state"] = "DEPENDENCY_NEVER_SATISFIED"
            else:
                tasks = [None] if spec["array"] is None else list(range(spec["array"][0], spec["array"][1] + 1))
                rec["codes"] = [run_one(d, spec, t) for t in tasks]
                if any(rec["codes"]):
                    rec["state"] = "FAILED"
            rec["ok"] = rec["state"] == "COMPLETED"
            records[str(jid)] = rec
            (d / "done" / f"{jid}.json").write_text(json.dumps(rec))
            break
        if not progressed:
            raise RuntimeError(f"mock scheduler deadlock: {[j for j, _ in pending]} wait for unknown jobs")
    else:
        raise RuntimeError("mock scheduler: job limit reached")
    return records


def main() -> int:
    cmd, argv = Path(sys.argv[0]).name, sys.argv[1:]
    if cmd == "sbatch":
        return sbatch(argv)
    if cmd == "scancel":
        return scancel(argv)
    if cmd == "scontrol":
        return scontrol(argv)
    if cmd == "srun":
        os.execvp(argv[0], argv)
    if cmd == "sacct":
        print("JobID|JobName|State|ExitCode|Elapsed|Timelimit|NodeList|MaxRSS|MaxVMSize")
        return 0
    if cmd == "sinfo":
        limit = os.environ.get("MOCK_PARTITION_LIMIT", "7-00:00:00")
        print(limit if "%l" in argv else f"gpu {limit} 382 288 512000 gpu:gh200:4")
        return 0
    if cmd == "sacctmgr":
        print("\n".join(os.environ.get("MOCK_ACCOUNTS", "naiss2025-5-643-gpu").split()))
        return 0
    if cmd == "nvidia-smi":
        print("mock nvidia-smi: no GPU")
        return 0
    return 0                                                # squeue and anything else: silent


if __name__ == "__main__":
    raise SystemExit(main())
