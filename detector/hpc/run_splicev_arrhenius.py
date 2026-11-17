#!/usr/bin/env python3
"""SPLICE-V detector suite for the NAISS Arrhenius GH200 cluster (channel 'train').

One runner for every Slurm role of run_splicev.sh:

  --print_version / --print_identity         identity checked by the launcher
  --gpu_health_check_only                    GPU sanity before a worker starts
  --semantic_preflight_only                  unit tests and autograd self-checks, fails closed
  --prepare_dataset_only                     dataset views, pretrained weights, API probe
  --init_suite                               suite configuration and the task manifests
  --stage S --task T ...                     one task of a stage
  --stage_gate S                             PASS (0) / RETRY (10) / CONTINUE (11) / FAIL (20) of a stage

Stage graph (the launcher runs the chain in order, then the evaluation stages in parallel):

  baseline -> train -> calibrate -> qat -> { evaluate, diagnostics, export } -> reduce

  baseline     the published YOLO26-S checkpoint on COCO val2017 in this pipeline
  train        every registered run: dataset x model (native YOLO26-S, the seven readout arms
               a-g of SPLICE-B, the forecast-descriptor variants of arm e) x seed
  calibrate    calibration split: band exponents, conformal thresholds, partner prior,
               activation scales, measured sensitivity
  qat          quantization-aware training of the deployed arm (e) at the manifest's sites
  evaluate     FP32 accuracy of every run with and without the objectness gate; the QAT model
               with the rounding sites active ('manifest sites'); per-image matches and the
               per-frame regime variables
  diagnostics  forecast error and crossing share, admissibility false-rejection rate,
               Lemma 3 quantities, leakage, sensitivity
  export       integer export and contract check of the QAT models
  reduce       tables, the declared tests of the train channel (2, 3, 9, 10, 12 and the
               arm (f) comparison), parameter table, manuscript-target comparison, report

Manuscript numbers are carried only as targets for comparison and are never reported as
measurements; a test without its evidence is NOT_EVALUATED or NOT_QUALIFIED, never a pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

VERSION = "1.0.0-arrhenius"
VERSION_TAG = "SPLICE_V1_0_0"
EXPECTED_RUNNER_BASENAME = "run_splicev_arrhenius.py"
PAPER = "SPLICE-V"

EXIT_PREREQUISITE = 33
EXIT_ARTIFACT_MISSING = 74
EXIT_INFRASTRUCTURE = 75
EXIT_CONTINUE = 76
GATE_PASS, GATE_RETRY, GATE_CONTINUE, GATE_FAIL = 0, 10, 11, 20

CHAIN_STAGES = ["baseline", "train", "calibrate", "qat"]
PARALLEL_STAGES = ["evaluate", "diagnostics", "export"]
ALL_STAGES = CHAIN_STAGES + PARALLEL_STAGES + ["reduce"]
STAGE_DIRS = {"baseline": "010_BASELINE", "train": "020_TRAIN", "calibrate": "030_CALIBRATION", "qat": "040_QAT",
              "evaluate": "070_EVALUATE", "diagnostics": "080_DIAGNOSTICS", "export": "085_EXPORT", "reduce": "900_REDUCE"}

ARMS = ["a", "b", "c", "d", "e", "f", "g"]
DESCRIPTORS = ["no_discount", "learned_mix", "lowpass", "blind", "local", "single_step"]
REQUIRED_MODELS = {"native", "arm-a", "arm-e"}          # a failure of these fails the stage; the others are optional
DIAG_MODELS = ["arm-d", "arm-e", "arm-f"] + [f"desc-{d}" for d in DESCRIPTORS]

# Declared tests of the train channel (SPLICE-V Table 6) and their margins: the arithmetic may
# cost 5% and the co-design 15% of the gain of SPLICE-B over YOLO26-S (2.0 AP, 7.4 MR^-2).
ALPHA = 0.05 / 4
MARGINS = {"arithmetic": {"AP": 0.10, "MR": 0.37}, "codesign": {"AP": 0.30, "MR": 1.11}}
HEAVY_RECALL_DROP_MAX = 0.5                             # test 12, points
BASELINE_MIN_AP = 46.0                                  # the published checkpoint must reproduce in this pipeline

# Manuscript values (SPLICE-V Tables 5-7): targets for comparison only, never measurements.
PAPER_TARGETS = {
    "table7": {
        "native": {"AP": 47.6, "AP75": 51.7, "MR": 44.2, "MR_heavy": 62.3},
        "arm-a": {"AP": 49.6, "AP75": 53.9, "MR": 36.8, "MR_heavy": 51.4},
        "arm-b": {"AP": 49.3, "AP75": 53.5, "MR": 37.3, "MR_heavy": 52.3},
        "arm-c": {"AP": 49.1, "AP75": 53.3, "MR": 37.7, "MR_heavy": 53.0},
        "arm-d": {"AP": 49.6, "AP75": 53.8, "MR": 36.9, "MR_heavy": 51.9, "err": 0.41, "cross": 0.12},
        "arm-e": {"AP": 49.6, "AP75": 53.9, "MR": 36.7, "MR_heavy": 51.1, "err": 0.34, "cross": 0.02},
        "arm-f": {"AP": 49.4, "AP75": 53.6, "MR": 37.5, "MR_heavy": 53.2, "err": 0.34, "cross": 0.02},
        "arm-g": {"AP": 48.9, "AP75": 53.0, "MR": 39.4, "MR_heavy": 56.0},
        "arm-e+gate": {"AP": 49.5, "AP75": 53.8, "MR": 36.8, "MR_heavy": 51.4},
        "arm-e+gate+manifest": {"AP": 49.44, "AP75": 53.72, "MR": 36.91, "MR_heavy": 51.6},
        "desc-no_discount": {"AP": 49.6, "AP75": 53.9, "MR": 36.7, "MR_heavy": 51.3, "err": 0.34, "cross": 0.03},
        "desc-learned_mix": {"AP": 49.6, "AP75": 53.9, "MR": 36.7, "MR_heavy": 51.2, "err": 0.34, "cross": 0.03},
        "desc-lowpass": {"AP": 49.5, "AP75": 53.7, "MR": 36.9, "MR_heavy": 51.6, "err": 0.39, "cross": 0.04},
        "desc-blind": {"AP": 49.4, "AP75": 53.5, "MR": 37.1, "MR_heavy": 52.4, "err": 0.52, "cross": 0.07},
        "desc-local": {"AP": 49.3, "AP75": 53.4, "MR": 37.2, "MR_heavy": 52.6, "err": 0.58, "cross": 0.08},
        "desc-single_step": {"AP": 49.2, "AP75": 53.3, "MR": 37.6, "MR_heavy": 53.4, "err": 0.61, "cross": 0.09},
    },
    "table6": {"test2": {"effect": -0.06, "se": 0.012, "bound": -0.087}, "test3": {"effect": 0.11, "se": 0.027, "bound": 0.171},
               "test9": {"effect": -0.10, "se": 0.065, "bound": -0.246}, "test10": {"effect": 0.00, "se": 0.111, "bound": 0.249},
               "test12": {"drop": 0.3}},
    "table5": {"skipped_share": {"crowdhuman": 0.30, "coco": 0.50}, "F_mean": {"crowdhuman": 103, "coco": 41},
               "F_shape": {"crowdhuman": 0.45, "coco": 0.60}, "N_over_F": 0.29, "cap_frames_share": {"crowdhuman": 0.012},
               "abstain_share": 0.31, "sensitivity": 0.07, "false_rejection_rate": 0.01},
}

SCALES = {
    "full": {"seeds": [0, 1, 2], "epochs": 50, "qat_epochs": 12, "fraction": 1.0, "max_iters": None, "val_limit": None,
             "calib_batches": 200, "diag_batches": 60, "bootstrap": 2000, "descriptors": True,
             "arms_multi": ["a", "e", "f", "g"], "arms_single": ["b", "c", "d"], "native": {"coco": 1, "crowdhuman": 3}},
    "pilot": {"seeds": [0], "epochs": 2, "qat_epochs": 1, "fraction": 0.05, "max_iters": None, "val_limit": 500,
              "calib_batches": 20, "diag_batches": 10, "bootstrap": 200, "descriptors": False,
              "arms_multi": ["a", "e", "f", "g"], "arms_single": [], "native": {"coco": 0, "crowdhuman": 1}},
    "preflight": {"seeds": [0], "epochs": 1, "qat_epochs": 1, "fraction": 1.0, "max_iters": 3, "val_limit": 32,
                  "calib_batches": 2, "diag_batches": 2, "bootstrap": 20, "descriptors": False,
                  "arms_multi": ["a", "e"], "arms_single": ["c"], "native": {"coco": 0, "crowdhuman": 1}},
}

DRY = os.environ.get("SPLICE_DRY_RUN", "0") == "1"       # orchestration test: synthetic task results, no torch


# ----------------------------------------------------------------------------- small utilities

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(msg, flush=True)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_root(arg: Optional[str]) -> Path:
    """The repository snapshot this runner executes against: --code_root, the suite snapshot
    next to the runner (000_MANIFEST/code), or the repository that contains the runner."""
    here = Path(__file__).resolve()
    cands = [Path(arg)] if arg else []
    cands += [here.parent / "code", here.parents[2] if len(here.parents) > 2 else here.parent, here.parent / "SPLICE-V"]
    for c in cands:
        if (c / "detector" / "splice_det" / "__init__.py").is_file():
            return c
    raise SystemExit(f"[FATAL] SPLICE-V code not found (looked in {[str(c) for c in cands]}); pass --code_root")


def add_code_to_path(root: Path) -> None:
    for p in (root / "detector", root / "reference", root / "generated" / "py", root / "tests"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def identity() -> Dict:
    return {"version": VERSION, "version_tag": VERSION_TAG, "expected_runner_basename": EXPECTED_RUNNER_BASENAME,
            "actual_runner_basename": Path(__file__).name, "paper": PAPER}


def slurm_env() -> Dict:
    e = os.environ
    return {"job_id": e.get("SLURM_JOB_ID", "manual"), "array_job_id": e.get("SLURM_ARRAY_JOB_ID", ""),
            "array_task_id": e.get("SLURM_ARRAY_TASK_ID", ""), "restart_count": e.get("SLURM_RESTART_COUNT", "0"),
            "node": e.get("SLURMD_NODENAME", socket.gethostname())}


class Prerequisite(RuntimeError):
    """An upstream artifact or input is missing: not retriable, not a scientific failure."""


class ContinueLater(Exception):
    """The task stopped cleanly before its wall-clock limit and continues after a requeue."""

    def __init__(self, info: Dict):
        super().__init__("continue")
        self.info = info


# ----------------------------------------------------------------------------- suite layout

def model_spec(model: str) -> Tuple[str, str]:
    """model name -> (arm, descriptor); 'native' keeps YOLO26-S's own neck."""
    if model == "native":
        return "native", "bands"
    if model.startswith("arm-"):
        return model[4:], "bands"
    if model.startswith("desc-"):
        return "e", model[5:]
    raise ValueError(model)


def run_key(dataset: str, model: str, seed: int) -> str:
    return f"{dataset}__{model}__s{seed}"


def parse_key(task: str) -> Dict:
    parts = task.split("__")
    if len(parts) < 3 or not parts[2].startswith("s"):
        raise ValueError(f"task name {task!r} is not <dataset>__<model>__s<seed>[__<variant>]")
    return {"dataset": parts[0], "model": parts[1], "seed": int(parts[2][1:]), "variant": parts[3] if len(parts) > 3 else None,
            "key": "__".join(parts[:3])}


class Suite:
    def __init__(self, path: str):
        self.root = Path(path)
        self.manifest = self.root / "000_MANIFEST"
        self.cfg = read_json(self.manifest / "suite_config.json", {})

    def stage_dir(self, stage: str) -> Path:
        return self.root / STAGE_DIRS[stage]

    def tasks(self, stage: str) -> List[Tuple[str, str]]:
        f = self.manifest / f"tasks_{stage}.tsv"
        if not f.exists():
            return []
        rows = [r.split("\t") for r in f.read_text().splitlines()[1:] if r.strip()]
        return [(r[0], r[1]) for r in rows]

    def result_dir(self, stage: str, ordinal: str, task: str) -> Path:
        return self.stage_dir(stage) / "runs" / f"{ordinal}_{task}" / "RESULT"

    def artifacts(self, key: str) -> Path:
        d = self.root / "ARTIFACTS" / key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ckpt_dir(self, key: str) -> Path:
        d = Path(self.cfg["ckpt_root"]) / key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def dataset(self, name: str) -> Dict:
        recs = read_json(self.manifest / "datasets.json", {})
        if name not in recs:
            raise Prerequisite(f"dataset {name!r} was not prepared for this suite ({self.manifest / 'datasets.json'})")
        return recs[name]

    def summaries(self, stage: str) -> Dict[str, Dict]:
        out = {}
        for ordinal, task in self.tasks(stage):
            s = read_json(self.result_dir(stage, ordinal, task) / "summary.json")
            if isinstance(s, dict):
                s["_dir"] = str(self.result_dir(stage, ordinal, task))
                out[task] = s
        return out

    def optional(self, task: str) -> bool:
        if task == "reduce" or "__" not in task:
            return False
        return parse_key(task)["model"] not in REQUIRED_MODELS and parse_key(task)["model"] != "native_pub"


def registered_runs(cfg: Dict) -> List[Tuple[str, str, int]]:
    runs = []
    for ds in cfg["datasets"]:
        for s in cfg["seeds"][:cfg["native"].get(ds, 0)]:
            runs.append((ds, "native", s))
        for arm in cfg["arms_multi"]:
            runs += [(ds, f"arm-{arm}", s) for s in cfg["seeds"]]
        for arm in cfg["arms_single"]:
            runs.append((ds, f"arm-{arm}", cfg["seeds"][0]))
        if cfg["descriptors"] and ("e" in cfg["arms_multi"] or "e" in cfg["arms_single"]):
            runs += [(ds, f"desc-{d}", cfg["seeds"][0]) for d in DESCRIPTORS]
    seen, out = set(), []
    for r in runs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def init_suite(args) -> int:
    suite = Path(args.suite)
    manifest = suite / "000_MANIFEST"
    manifest.mkdir(parents=True, exist_ok=True)
    cfg = dict(SCALES[args.scale])
    cfg["scale"] = args.scale
    cfg["datasets"] = [d for d in args.datasets.split(",") if d]
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.arms:
        sel = [a for a in args.arms.split(",") if a]
        bad = [a for a in sel if a not in ARMS]
        if bad:
            raise SystemExit(f"[FATAL] unknown arm(s) {bad}; choose from {ARMS}")
        cfg["arms_multi"] = [a for a in cfg["arms_multi"] if a in sel]
        cfg["arms_single"] = [a for a in sel if a not in cfg["arms_multi"]]
    if args.descriptors is not None:
        cfg["descriptors"] = bool(int(args.descriptors))
    for k in ("epochs", "qat_epochs", "val_limit", "max_iters"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = int(v) if int(v) > 0 or k in ("qat_epochs",) else None
    if args.fraction is not None:
        cfg["fraction"] = float(args.fraction)
    if args.native_coco is not None:
        cfg["native"] = dict(cfg["native"], coco=int(args.native_coco))
    cfg.update(batch=args.batch, imgsz=args.imgsz, lr=args.lr, lr_pretrained=args.lr_pretrained, base_seed=args.seed,
               ckpt_root=str(Path(args.ckpt_root)), weights=str(Path(args.weights)), dry_run=DRY, paper=PAPER,
               alpha=ALPHA, margins=MARGINS, runner_version=VERSION)
    runs = registered_runs(cfg)
    cfg["runs"] = [run_key(*r) for r in runs]
    write_json(manifest / "suite_config.json", cfg)
    proto = None
    for cand in (Path(args.code_root or ".") / "planning" / "protocol.json", manifest / "code" / "planning" / "protocol.json"):
        if cand.is_file():
            proto = {"path": str(cand), "sha256": sha256_file(cand)}
            break
    write_json(manifest / "declared_tests.json", {
        "source": f"{PAPER} Table 6 and section 4 ('Claims and the declared tests'), train channel",
        "alpha_one_sided": ALPHA, "margins": MARGINS, "heavy_recall_drop_max_points": HEAVY_RECALL_DROP_MAX,
        "bootstrap": {"levels": ["seeds", "whole images"], "replicates": cfg["bootstrap"]},
        "protocol_json": proto or "not present in the code snapshot (planning/protocol.json); its hash is not bound here",
        "tests": {"2": "AP (COCO), manifest sites - FP32, arm (e) with gate: lower bound > -0.10",
                  "3": "MR^-2 (CrowdHuman), manifest sites - FP32, arm (e) with gate: upper bound < +0.37",
                  "9": "AP (COCO), arm (e) with gate - arm (a): lower bound > -0.30",
                  "10": "MR^-2 (CrowdHuman), arm (e) with gate - arm (a): upper bound < +1.11",
                  "12": "gate: recall on heavily occluded persons drops by at most 0.5 pt",
                  "f_vs_e": "arm (f) non-inferior to arm (e) at the co-design margins on AP, MR^-2 and heavy MR^-2"}})

    def emit(stage: str, names: List[str]) -> None:
        lines = ["ordinal\ttask\tstage"] + [f"{i + 1:04d}\t{n}\t{stage}" for i, n in enumerate(names)]
        (manifest / f"tasks_{stage}.tsv").write_text("\n".join(lines) + "\n")
        log(f"[INIT] tasks_{stage}.tsv: {len(names)} task(s)")

    emit("baseline", ["coco__native_pub__s0"] if "coco" in cfg["datasets"] else [])
    emit("train", [run_key(*r) for r in runs])
    splice = [r for r in runs if r[1] != "native"]
    emit("calibrate", [run_key(*r) for r in splice])
    qat = [r for r in splice if r[1] == "arm-e"] if cfg["qat_epochs"] else []
    emit("qat", [run_key(*r) for r in qat])
    emit("evaluate", [run_key(*r) + "__fp32" for r in runs] + [run_key(*r) + "__manifest" for r in qat])
    s0 = cfg["seeds"][0]
    emit("diagnostics", [run_key(*r) for r in splice if r[1] in DIAG_MODELS and r[2] == s0])
    emit("export", [run_key(*r) for r in qat])
    emit("reduce", ["reduce"])
    log(f"[INIT] scale={args.scale} datasets={cfg['datasets']} seeds={cfg['seeds']} runs={len(runs)} dry_run={DRY}")
    return 0


# ----------------------------------------------------------------------------- GPU health, preflight

def gpu_health(args) -> int:
    """A small deterministic GPU computation checked against the CPU; exit 75 when the device
    is unusable so that the launcher requeues the task on another node."""
    rec = {"version": VERSION, "status": "UNKNOWN", "timestamp_utc": now(), **slurm_env()}
    if DRY:
        rec.update(status="PASS", note="dry run: no GPU used")
        write_json(Path(args.gpu_health_out), rec)
        return 0
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        n = int(args.gpu_health_matrix_size)
        g = torch.Generator().manual_seed(1234)
        a = torch.randn(n, n, generator=g, dtype=torch.float64)
        ref = (a @ a.T).diagonal().sum().item()
        worst = 0.0
        for _ in range(int(args.gpu_health_repeats)):
            b = a.cuda()
            got = (b @ b.T).diagonal().sum().item()
            worst = max(worst, abs(got - ref) / max(abs(ref), 1e-30))
            del b
        torch.cuda.synchronize()
        rec.update(status="PASS" if worst < 1e-9 else "FAIL", relative_error=worst, device=torch.cuda.get_device_name(0),
                   memory_gb=torch.cuda.get_device_properties(0).total_memory / 2 ** 30, torch=torch.__version__)
    except Exception as e:                                     # any CUDA failure is an infrastructure failure
        rec.update(status="FAIL", error=f"{type(e).__name__}: {e}")
    write_json(Path(args.gpu_health_out), rec)
    log(f"[GPU_HEALTH] {rec['status']} {rec.get('device', '')} {rec.get('error', '')}")
    return 0 if rec["status"] == "PASS" else EXIT_INFRASTRUCTURE


# Test modules of the code snapshot that the prepare job runs (the snapshot holds
# detector/splice_det, reference, generated/py, contracts, tools, tests and planning).
PREFLIGHT_TESTS = ["test_detector_core", "test_detector_metrics", "test_detector_data"]
PREFLIGHT_TESTS_TORCH = ["test_detector_torch"]                 # on real torch (CPU) in the prepare job


def semantic_preflight(args, root: Path) -> int:
    """Unit tests of the code snapshot and, with real torch, the autograd self-checks of the
    SPLICE neck (every arm) and of the differentiable exchange.  Fails closed, except with
    --advisory (the smoke scale), where a failure is recorded and reported but does not stop
    the suite."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"version": VERSION, "started_utc": now(), "code_root": str(root), "dry_run": DRY, "checks": {}}
    tests = list(PREFLIGHT_TESTS)
    if not DRY:
        tests += PREFLIGHT_TESTS_TORCH
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "tests"), str(root / "detector")]))
    ok = True
    for t in tests:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "unittest", f"tests.{t}"], cwd=str(root), env=env, capture_output=True, text=True)
        (out / f"{t}.log").write_text(r.stdout + "\n" + r.stderr)
        tail = (r.stderr.strip().splitlines() or [""])[-1]
        report["checks"][t] = {"returncode": r.returncode, "seconds": round(time.time() - t0, 1), "result": tail}
        log(f"[PREFLIGHT] {t}: rc={r.returncode} {tail}")
        ok &= r.returncode == 0
    if DRY and "preflight:fail" in os.environ.get("SPLICE_DRY_FAULTS", ""):       # orchestration tests only
        report["checks"]["injected"] = {"returncode": 1, "result": "injected preflight failure (dry run)"}
        ok = False
    if not DRY:
        try:
            add_code_to_path(root)
            from splice_det import selfcheck
            res = selfcheck.run(log=log)
            report["checks"]["autograd_selfcheck"] = res
            ok &= bool(res["passed"])
        except Exception as e:
            report["checks"]["autograd_selfcheck"] = {"passed": False, "error": f"{type(e).__name__}: {e}",
                                                      "traceback": traceback.format_exc()}
            ok = False
            log(f"[PREFLIGHT] autograd self-check raised: {type(e).__name__}: {e}")
    report.update(passed=bool(ok), advisory=bool(args.advisory), finished_utc=now())
    write_json(out / "PREFLIGHT_REPORT.json", report)
    log(f"[PREFLIGHT] {'PASS' if ok else 'FAIL'}: {out / 'PREFLIGHT_REPORT.json'}")
    if not ok and args.advisory:
        log("[PREFLIGHT] advisory mode (smoke scale): the failure is recorded and the suite goes on, so that one pass "
            "shows every problem; pilot and full suites stop here")
        return 0
    return 0 if ok else 1


# ----------------------------------------------------------------------------- datasets, weights, API probe

WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"


def ensure_weights(path: Path) -> Dict:
    """The published YOLO26-S checkpoint at a fixed place (downloaded once if the node is online)."""
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f"[WEIGHTS] {path} missing; downloading {WEIGHT_URL}")
        try:
            import urllib.request
            tmp = path.with_name(path.name + ".part")
            urllib.request.urlretrieve(WEIGHT_URL, tmp)
            tmp.replace(path)
        except Exception as e:
            raise Prerequisite(f"pretrained weights not found at {path} and the download failed ({type(e).__name__}: {e}). "
                               f"Place yolo26s.pt there (e.g. on the login node: curl -L -o {path} {WEIGHT_URL}).")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def api_probe(weights: str, out: Path, advisory: bool = False) -> Dict:
    """One training and one inference step of SPLICE-B (arms e and c) and of the native model
    on a random 320 x 320 batch: checks the installed torch / Ultralytics / YOLO26 interfaces
    (layout taps, raw head format, loss, decode) before any long job.  Each model is probed on
    its own; a failure stops the suite unless `advisory` (the smoke scale), where it is
    recorded and the suite goes on."""
    import torch
    from splice_det import engine, losses
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = {"device": str(dev), "torch": torch.__version__, "failed": []}
    import ultralytics
    rep["ultralytics"] = ultralytics.__version__
    torch.manual_seed(0)
    img = torch.rand(2, 3, 320, 320, device=dev)
    batch = {"img": img, "batch_idx": torch.tensor([0.0, 0.0, 1.0]), "cls": torch.tensor([[0.0], [0.0], [0.0]]),
             "bboxes": torch.tensor([[0.3, 0.4, 0.2, 0.5], [0.7, 0.5, 0.25, 0.6], [0.5, 0.5, 0.4, 0.7]])}

    def probe(arm: str) -> Dict:
        model = engine.build_model(weights, dev, arm, "bands", 0, 320, nc=1)
        try:
            model.train()
            boxes = losses.batch_boxes(batch) if arm != "native" else None
            raw, nk = model(img, boxes, probe=True)
            det, _items = model.detection_loss(raw, engine.to_device(batch, dev))
            loss = det.sum()
            if nk is not None:
                aux = losses.aux_losses(nk, boxes, model.neck.strides, model.neck.cfg.hyp_weight)
                loss, _ = losses.total_loss(loss.float(), nk, aux, model.neck.cfg, 2)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
            model.eval()
            with torch.no_grad():
                y, _ = model(img, gate="both") if arm != "native" else model(img)
            ys = y if isinstance(y, dict) else {"native": y}
            res = {"status": "PASS", "loss": float(loss), "grads_finite": finite, "params_with_grad": len(grads),
                   "raw_format": type(raw).__name__ + (":" + ",".join(sorted(raw.keys())) if isinstance(raw, dict) else ""),
                   "decoded": {k: list(v.shape) for k, v in ys.items()},
                   "end2end": bool(getattr(model.yolo.model[-1], "end2end", False)),
                   "layout": getattr(model, "layout", None)}
            if not (finite and math.isfinite(float(loss))):
                raise RuntimeError(f"non-finite loss or gradients ({res})")
            return res
        finally:
            if getattr(model, "neck", None) is not None and model.neck._pool is not None:
                model.neck._pool.close()

    for name, arm in (("splice_arm_e", "e"), ("splice_arm_c", "c"), ("native", "native")):
        try:
            rep[name] = probe(arm)
            log(f"[API_PROBE] {name}: loss={rep[name]['loss']:.4f} raw={rep[name]['raw_format']} decoded={rep[name]['decoded']}")
        except Exception as e:
            rep[name] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            rep["failed"].append(name)
            log(f"[API_PROBE] {name}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
    rep["status"] = "FAILED" if rep["failed"] else "PASS"
    write_json(out / "API_PROBE.json", rep)
    if rep["failed"] and not advisory:
        raise RuntimeError(f"API probe failed for {rep['failed']}: see {out / 'API_PROBE.json'}")
    if rep["failed"]:
        log("[API_PROBE] advisory mode (smoke scale): the failure is recorded and the suite goes on")
    return rep


def prepare_dataset(args, root: Path) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite = Path(args.suite)
    names = [d for d in args.datasets.split(",") if d]
    records, report = {}, {"version": VERSION, "datasets": {}, "dry_run": DRY}
    try:
        if DRY:
            for n in names:
                records[n] = {"name": n, "path": str(out / n), "dry_run": True, "nc": 80 if n == "coco" else 1,
                              "counts": {"train_fit": {"images": 0}, "calib": {"images": 0}, "val": {"images": 40}}}
            report["weights"] = {"path": args.weights, "sha256": None, "dry_run": True}
        else:
            add_code_to_path(root)
            from splice_det import data as D
            views = Path(args.data_root) / "splice_views"
            for n in names:
                src = Path(args.coco_root or Path(args.data_root) / "coco") if n == "coco" else \
                    Path(args.crowdhuman_root or Path(args.data_root) / "crowdhuman")
                try:
                    records[n] = D.prepare_coco(src, views, log) if n == "coco" else D.prepare_crowdhuman(src, views, log)
                except D.DataError as e:
                    raise Prerequisite(str(e))
            report["weights"] = ensure_weights(Path(args.weights))
            report["api_probe"] = api_probe(args.weights, out, advisory=bool(args.advisory))
        for n, r in records.items():
            report["datasets"][n] = {"path": r["path"], "counts": r.get("counts"), "nc": r.get("nc")}
        write_json(suite / "000_MANIFEST" / "datasets.json", records)
        report["status"] = "PREPARED"
        write_json(out / "DATASET_REPORT.json", report)
        log(f"[DATA] prepared: {json.dumps(report['datasets'])}")
        return 0
    except Prerequisite as e:
        report.update(status="FAILED_PREREQUISITE", message=str(e))
        write_json(out / "DATASET_REPORT.json", report)
        log(f"[FATAL][PREREQUISITE] {e}")
        return EXIT_PREREQUISITE


# ----------------------------------------------------------------------------- stage tasks

def _deadline(args) -> Optional[float]:
    for v in (args.deadline_epoch, os.environ.get("SLURM_JOB_END_TIME")):
        try:
            if v and float(v) > 0:
                return float(v) - 600.0                       # leave ten minutes for the checkpoint and the summary
        except ValueError:
            pass
    return None


def _cpus_allowed() -> int:
    """Cores this process may run on (the job step's binding), not the node's core count."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _workers(args) -> Tuple[int, int]:
    """(data-loader workers, CPU processes of the neck) from the task's CPU allocation."""
    n = max(2, min(int(args.threads), _cpus_allowed()))
    data = max(2, min(8, n // 4))
    return data, max(1, n - data - 2)


def _logger(run_dir: Path):
    f = open(run_dir / "task_log.jsonl", "a")

    def emit(rec: Dict) -> None:
        rec = dict(rec, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        if rec.get("event") or rec.get("stage") in ("setup", "calibrate", "qat_attach", "export") or rec.get("iter", 1) % 200 == 0:
            log(json.dumps(rec, default=str))
    return emit


def _device(args):
    import torch
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available (infrastructure)")
    torch.set_num_threads(max(1, min(8, int(args.threads))))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _meta(suite: Suite, k: Dict, data: Dict) -> Dict:
    arm, desc = model_spec(k["model"])
    return {"weights": suite.cfg["weights"], "nc": int(data["nc"]), "seed": k["seed"], "dataset": k["dataset"], "arm": arm,
            "descriptor": desc, "model_name": k["model"], "suite": str(suite.root), "runner_version": VERSION}


def _seed(suite: Suite, k: Dict) -> int:
    return int(suite.cfg.get("base_seed", 1)) * 1000 + k["seed"]


def _fake_ckpt(path: Path, what: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"DRY RUN placeholder for {what}\n")


def stage_train(suite: Suite, args, run_dir: Path) -> Dict:
    k = parse_key(args.task)
    cfg = suite.cfg
    ck_dir, art = suite.ckpt_dir(k["key"]), suite.artifacts(k["key"])
    final = ck_dir / "trained.pt"
    done = read_json(art / "train.json")
    if done and done.get("status") == "DONE" and final.exists():
        return {"status": "ALREADY_TRAINED", "checkpoint": str(final), **{x: done.get(x) for x in ("epochs", "sha256")}}
    if DRY:
        _fake_ckpt(final, k["key"])
        res = {"status": "DONE", "epochs": cfg["epochs"], "checkpoint": str(final), "sha256": sha256_file(final), "dry_run": True}
        write_json(art / "train.json", res)
        return res
    from splice_det import engine
    from splice_det.data import load_dataset
    dev = _device(args)
    data = load_dataset(suite.dataset(k["dataset"]))
    arm, desc = model_spec(k["model"])
    dl, lw = _workers(args)
    emit = _logger(run_dir)
    meta = _meta(suite, k, data)
    model = engine.build_model(cfg["weights"], dev, arm, desc, lw, cfg["imgsz"], data["nc"])
    emit({"stage": "setup", "task": args.task, "device": str(dev), **meta, "layout": getattr(model, "layout", None),
          "data_workers": dl, "neck_workers": lw, "cpus_requested": int(args.threads), "cpus_allowed": _cpus_allowed(),
          "epochs": cfg["epochs"]})
    if _cpus_allowed() < int(args.threads):
        log(f"[WARN] the task may run on {_cpus_allowed()} core(s) but {args.threads} were requested: the job step is "
            f"bound to fewer cores than the job (check SRUN_CPUS_PER_TASK / the site's task binding)")
    try:
        st = engine.train_epochs(model, data, ck_dir / "last.pt", epochs=cfg["epochs"], batch=cfg["batch"], workers=dl,
                                 imgsz=cfg["imgsz"], lr=cfg["lr"], lr_pre=cfg["lr_pretrained"], seed=_seed(suite, k),
                                 fraction=cfg["fraction"], log=emit, deadline=_deadline(args), max_iters=cfg["max_iters"],
                                 warmup=1000 if cfg["max_iters"] is None else 2, meta=meta,
                                 close_mosaic=min(10, max(0, cfg["epochs"] - 1)))
    finally:
        if getattr(model, "neck", None) is not None and model.neck._pool is not None:
            model.neck._pool.close()
    if st["status"] == "CONTINUE":
        raise ContinueLater(st)
    engine.save_checkpoint(final, model, {**meta, "final": True, "epoch": st["epoch"], "epochs": st["epochs"]})
    res = {"status": "DONE", "epochs": st["epochs"], "iterations": st["it"], "epoch_seconds": st["epoch_seconds"],
           "checkpoint": str(final), "sha256": sha256_file(final)}
    write_json(art / "train.json", res)
    return res


def _need(path: Path, what: str) -> Path:
    if not Path(path).is_file():
        raise Prerequisite(f"{what} not found: {path} (its upstream task did not complete)")
    return Path(path)


def stage_calibrate(suite: Suite, args, run_dir: Path) -> Dict:
    k = parse_key(args.task)
    cfg = suite.cfg
    ck_dir, art = suite.ckpt_dir(k["key"]), suite.artifacts(k["key"])
    trained = _need(ck_dir / "trained.pt", "trained checkpoint")
    final = ck_dir / "calibrated.pt"
    if DRY:
        _fake_ckpt(final, k["key"])
        rec = {"gate_tau": [13.3] * 4, "prior": [0.4, 0.45, 0.15], "band_exponents": [[4, 4, 3, 3]] * 3,
               "sensitivity": {"q_over_response": 0.1, "q_over_probe": 0.05}, "split": "calib", "dry_run": True}
        write_json(art / "calibration.json", rec)
        return {"status": "DONE", "checkpoint": str(final), "sha256": sha256_file(final), "calibration": rec, "dry_run": True}
    from splice_det import engine
    from splice_det.data import load_dataset
    dev = _device(args)
    data = load_dataset(suite.dataset(k["dataset"]))
    dl, lw = _workers(args)
    emit = _logger(run_dir)
    model, ck = engine.load_model(str(trained), dev, lw, cfg["imgsz"])
    try:
        rec = engine.calibrate_model(model, data, split="calib", batches=cfg["calib_batches"], batch=cfg["batch"], workers=dl,
                                     imgsz=cfg["imgsz"], log=emit)
    finally:
        if model.neck._pool is not None:
            model.neck._pool.close()
    engine.save_checkpoint(final, model, {**_meta(suite, k, data), "final": True, "epoch": ck.get("epoch"), "calibration": rec})
    write_json(art / "calibration.json", rec)
    return {"status": "DONE", "checkpoint": str(final), "sha256": sha256_file(final),
            "calibration": {x: v for x, v in rec.items() if x != "act_lsb"}}


def stage_qat(suite: Suite, args, run_dir: Path) -> Dict:
    k = parse_key(args.task)
    cfg = suite.cfg
    ck_dir, art = suite.ckpt_dir(k["key"]), suite.artifacts(k["key"])
    calibrated = _need(ck_dir / "calibrated.pt", "calibrated checkpoint")
    final = ck_dir / "qat.pt"
    done = read_json(art / "qat.json")
    if done and done.get("status") == "DONE" and final.exists():
        return {"status": "ALREADY_TRAINED", "checkpoint": str(final), "sha256": done.get("sha256")}
    if DRY:
        _fake_ckpt(final, k["key"])
        res = {"status": "DONE", "epochs": cfg["qat_epochs"], "checkpoint": str(final), "sha256": sha256_file(final), "dry_run": True}
        write_json(art / "qat.json", res)
        return res
    from splice_det import engine, qat
    from splice_det.data import load_dataset
    dev = _device(args)
    data = load_dataset(suite.dataset(k["dataset"]))
    dl, lw = _workers(args)
    emit = _logger(run_dir)
    model, ck = engine.load_model(str(calibrated), dev, lw, cfg["imgsz"])
    calib = ck.get("calibration") or {}
    summary = qat.attach(model, calib.get("act_lsb"))
    emit({"stage": "qat_attach", **summary})
    meta = _meta(suite, k, data)
    try:
        st = engine.train_epochs(model, data, ck_dir / "last_qat.pt", epochs=cfg["qat_epochs"], batch=cfg["batch"], workers=dl,
                                 imgsz=cfg["imgsz"], lr=cfg["lr"] * 0.1, lr_pre=cfg["lr_pretrained"] * 0.1,
                                 warmup=100 if cfg["max_iters"] is None else 2, seed=_seed(suite, k) + 500,
                                 fraction=cfg["fraction"], log=emit, deadline=_deadline(args), max_iters=cfg["max_iters"],
                                 tag="qat", use_ema=False, close_mosaic=cfg["qat_epochs"], meta=meta)
    finally:
        if model.neck._pool is not None:
            model.neck._pool.close()
    if st["status"] == "CONTINUE":
        raise ContinueLater(st)
    qat.detach(model)
    engine.save_checkpoint(final, model, {**meta, "final": True, "epoch": ck.get("epoch"), "calibration": calib, "qat": summary,
                                          "qat_epochs": st["epochs"]})
    res = {"status": "DONE", "epochs": st["epochs"], "checkpoint": str(final), "sha256": sha256_file(final), "sites": summary}
    write_json(art / "qat.json", res)
    return res


def _dry_eval(k: Dict, variant: str, out: Path) -> Dict:
    """Synthetic detections through the real matchers and file formats (orchestration test)."""
    import numpy as np
    add = {"native": 0.0, "native_pub": 0.0, "arm-g": 0.06, "arm-f": 0.10, "arm-e": 0.12, "arm-a": 0.12}.get(k["model"], 0.09)
    from splice_det import metrics as M
    n_img = 40
    rec = {"dataset": k["dataset"], "images": n_img, "variants": {}, "limit": n_img, "dry_run": True}
    scene = np.random.default_rng(zlib.crc32(k["dataset"].encode()))              # the same images for every model
    noise = np.random.default_rng(zlib.crc32(f"{k['key']}|{variant}".encode()))
    gts, base = [], []
    for _ in range(n_img):
        m = int(scene.integers(2, 12))
        xy = scene.uniform(0, 500, (m, 2))
        b = np.concatenate([xy, xy + scene.uniform(20, 160, (m, 2))], 1)
        gts.append({"boxes": b, "ign": scene.uniform(0, 1, m) < 0.1, "vis": scene.uniform(0.2, 1.0, m),
                    "cat": scene.integers(1, 4, m)})
        base.append(scene.uniform(0, 1, m))
    keys = ["native"] if k["model"].startswith("native") else ["gate", "nogate"]
    for key in keys:
        p_hit = 0.62 + add - (0.01 if key == "gate" else 0.0) - (0.005 if variant == "manifest" else 0.0)
        dets = []
        for g, u in zip(gts, base):
            hit = u < p_hit
            d = g["boxes"][hit] + noise.normal(0, 3, (int(hit.sum()), 4))
            sc = np.round(noise.uniform(0.3, 1.0, len(d)), 4)
            dets.append(np.concatenate([d, sc[:, None], g["cat"][hit][:, None].astype(float)], 1))
        if k["dataset"] == "coco":
            truth = {"cats": [1, 2, 3], "gt": {i: {} for i in range(n_img)}}
            for i, g in enumerate(gts):
                for b, c in zip(g["boxes"], g["cat"]):
                    truth["gt"][i].setdefault(int(c), []).append(tuple(b) + (0,))
            arr = M.coco_match_numpy({i: d for i, d in enumerate(dets)}, truth, list(range(n_img)))
            np.savez_compressed(out / f"matches_{key}.npz", **arr)
            rec["variants"][key] = {**M.CocoAccumulator(arr)(), "matcher": "numpy (dry run)"}
        else:
            v = {}
            for stratum in ("all", "heavy"):
                gg = [{"boxes": g["boxes"], "ign": g["ign"] | ((g["vis"] >= 0.65) if stratum == "heavy" else False)} for g in gts]
                arr = M.crowd_match([d[:, :5] for d in dets], gg)
                m = M.crowd_metrics(arr)
                v[stratum] = {"AP": m["AP"], "MR": m["MR"], "recall": m["recall"], "gt": int(arr["n_gt"].sum())}
                np.savez_compressed(out / f"matches_{key}_{stratum}.npz", **arr)
            rec["variants"][key] = v
    if keys != ["native"]:
        rec["regime"] = {"frames": n_img, "F_mean": 60.0, "F_p50": 55.0, "F_p99": 200.0, "F_max": 256.0, "F_log_std": 0.5, "F_cv": 0.5,
                         "cap_frames_share": 0.0, "budget_exit_share": 0.0, "rounds_mean": 16.0, "N_f_mean": 18.0,
                         "N_over_F_mean": 0.3, "N_f_max": 60, "skipped_share": {"block1": 0.5, "block2": 0.4, "block4": 0.3}}
    return rec


def _evaluate(suite: Suite, args, run_dir: Path, k: Dict, variant: str) -> Dict:
    cfg = suite.cfg
    if DRY:
        add_code_to_path(code_root(args.code_root))
        return _dry_eval(k, variant, run_dir)
    from splice_det import engine, qat
    from splice_det.data import load_dataset
    from splice_det.evaluate import evaluate_model
    dev = _device(args)
    data = load_dataset(suite.dataset(k["dataset"]))
    dl, lw = _workers(args)
    if k["model"] == "native_pub":
        model = engine.build_model(cfg["weights"], dev, "native")
        src = Path(cfg["weights"])
    else:
        name = {"fp32": "trained.pt" if k["model"] == "native" else "calibrated.pt", "manifest": "qat.pt"}[variant]
        src = _need(suite.ckpt_dir(k["key"]) / name, f"{variant} checkpoint")
        model, ck = engine.load_model(str(src), dev, lw, cfg["imgsz"])
        if variant == "manifest":
            qat.attach(model, (ck.get("calibration") or {}).get("act_lsb"))
    try:
        rec = evaluate_model(model, data, dev, run_dir, cfg["batch"], dl, cfg["imgsz"], cfg["val_limit"])
    finally:
        if getattr(model, "neck", None) is not None and model.neck._pool is not None:
            model.neck._pool.close()
    rec.update(checkpoint=str(src), checkpoint_sha256=sha256_file(src))
    return rec


def stage_baseline(suite: Suite, args, run_dir: Path) -> Dict:
    k = parse_key(args.task)
    rec = _evaluate(suite, args, run_dir, k, "fp32")
    ap = rec["variants"]["native"].get("AP")
    full = suite.cfg["scale"] == "full" and suite.cfg["val_limit"] is None and not DRY
    rec["criterion"] = {"min_AP_points": BASELINE_MIN_AP, "applies": full,
                        "note": "the published YOLO26-S checkpoint must reproduce in this evaluation pipeline "
                                "(47.8 AP reported for the official checkpoint)"}
    if full and not (ap is not None and 100.0 * ap >= BASELINE_MIN_AP):
        raise AssertionError(f"baseline check failed: YOLO26-S reaches {100.0 * (ap or 0):.2f} AP on COCO val2017, "
                             f"below {BASELINE_MIN_AP}; the evaluation pipeline or the data view is wrong")
    return {"status": "DONE", "evaluation": rec}


def stage_evaluate(suite: Suite, args, run_dir: Path) -> Dict:
    k = parse_key(args.task)
    return {"status": "DONE", "variant": k["variant"], "evaluation": _evaluate(suite, args, run_dir, k, k["variant"])}


def stage_diagnostics(suite: Suite, args, run_dir: Path) -> Dict:
    k = parse_key(args.task)
    cfg = suite.cfg
    src = _need(suite.ckpt_dir(k["key"]) / "calibrated.pt", "calibrated checkpoint")
    if DRY:
        err = {"arm-d": 0.41, "arm-e": 0.34, "arm-f": 0.34}.get(k["model"], 0.45)
        d = {"frames": 32, "forecast": {"err_median": err, "cross_share": 0.03, "abstain_share": 0.3, "clip_share": 0.05},
             "admissibility": {"delta": 0.01, "false_rejection_rate": {"1": {"rejected": 1, "pairs": 400, "rate": 0.0025}}},
             "regime": {"N_over_F_mean": 0.3}, "lemma3": {}, "leakage": {}, "sensitivity": {"q_over_response": 0.1}}
        return {"status": "DONE", "splits": {"calib": d, "val": d}, "dry_run": True}
    from splice_det import diagnostics, engine
    from splice_det.data import load_dataset, make_loader
    dev = _device(args)
    data = load_dataset(suite.dataset(k["dataset"]))
    dl, lw = _workers(args)
    emit = _logger(run_dir)
    model, _ = engine.load_model(str(src), dev, lw, cfg["imgsz"])
    out = {}
    try:
        for split in ("calib", "val"):
            loader = make_loader(data, split, "val", cfg["batch"], dl, cfg["imgsz"], shuffle=False)
            out[split] = diagnostics.run(model, loader, cfg["diag_batches"], exact_images=200 if split == "calib" else 0, log=emit)
            write_json(run_dir / f"diagnostics_{split}.json", out[split])
    finally:
        if model.neck._pool is not None:
            model.neck._pool.close()
    return {"status": "DONE", "splits": out, "checkpoint_sha256": sha256_file(src)}


def stage_export(suite: Suite, args, run_dir: Path) -> Dict:
    k = parse_key(args.task)
    src = _need(suite.ckpt_dir(k["key"]) / "qat.pt", "QAT checkpoint")
    if DRY:
        return {"status": "DONE", "export": {"dry_run": True}, "contract": {"returncode": 0}}
    root = code_root(args.code_root)
    from splice_det import export
    rep = export.export_torch(str(src), str(run_dir / "export"), run=k["key"])
    r = subprocess.run([sys.executable, str(root / "tools" / "check_contract.py"), "--manifest",
                        str(run_dir / "export" / "manifest.model.json")], capture_output=True, text=True, cwd=str(root))
    (run_dir / "check_contract.txt").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise AssertionError(f"the exported manifest violates the contract (tools/check_contract.py rc={r.returncode}); "
                             f"see {run_dir / 'check_contract.txt'}")
    return {"status": "DONE", "export": rep, "contract": {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-6:]},
            "checkpoint_sha256": sha256_file(src)}


# ----------------------------------------------------------------------------- reduce

def _stats(vals: Sequence[float]) -> Optional[Dict]:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not v:
        return None
    return {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)}


def _metric(rec: Dict, dataset: str, variant: str, name: str) -> Optional[float]:
    """A metric in points (x 100) from an evaluation record; None when it is not there."""
    v = (rec or {}).get("variants", {}).get(variant)
    if not v:
        return None
    if dataset == "coco":
        src = v.get("pycocotools", v)
        x = src.get(name)
    else:
        key, field = {"MR": ("all", "MR"), "MR_heavy": ("heavy", "MR"), "AP": ("all", "AP"), "recall_heavy": ("heavy", "recall"),
                      "recall": ("all", "recall")}[name]
        x = v.get(key, {}).get(field)
    return None if x is None else 100.0 * float(x)


def _fmt(s: Optional[Dict], digits: int = 2) -> str:
    if not s:
        return "NOT_RUN"
    spread = f" [{s['min']:.{digits}f}, {s['max']:.{digits}f}]" if s["n"] > 1 else ""
    return f"{s['mean']:.{digits}f}{spread} (n={s['n']})"


def _metric_fn(result_dir: str, dataset: str, variant: str, name: str):
    """(callable weights -> metric in points, number of images) from the kept matches."""
    import numpy as np
    from splice_det import metrics as M
    d = Path(result_dir)
    if dataset == "coco":
        f = d / f"matches_{variant}.npz"
        if not f.is_file():
            return None, 0
        arr = dict(np.load(f))
        acc = M.CocoAccumulator(arr)
        return (lambda w: 100.0 * acc(w)[name]), len(arr["img_ids"])
    stratum = "heavy" if name == "MR_heavy" else "all"
    f = d / f"matches_{variant}_{stratum}.npz"
    if not f.is_file():
        return None, 0
    arr = dict(np.load(f))
    order = np.argsort(-arr["score"], kind="mergesort")          # sort once: every replicate then sorts sorted data
    arr.update({k: arr[k][order] for k in ("score", "img", "tp")})
    return (lambda w: 100.0 * M.crowd_metrics(arr, w)["MR"]), len(arr["n_gt"])


def stage_reduce(suite: Suite, args, run_dir: Path) -> Dict:
    add_code_to_path(code_root(args.code_root))
    from splice_det import metrics as M
    cfg = suite.cfg
    ev = suite.summaries("evaluate")
    base = suite.summaries("baseline")
    diag = suite.summaries("diagnostics")
    stage_status = {st: {"tasks": len(suite.tasks(st)), "completed": len(suite.summaries(st))} for st in ALL_STAGES if st != "reduce"}

    # evaluation records: evals[dataset][model][variant][seed] = record (with its result folder)
    evals: Dict = {}
    for task, s in ev.items():
        k = parse_key(task)
        rec = dict(s["evaluation"], _dir=s["_dir"])
        evals.setdefault(k["dataset"], {}).setdefault(k["model"], {}).setdefault(k["variant"], {})[k["seed"]] = rec
    for task, s in base.items():
        k = parse_key(task)
        evals.setdefault(k["dataset"], {}).setdefault("native_pub", {}).setdefault("fp32", {})[0] = dict(s["evaluation"], _dir=s["_dir"])

    def values(dataset: str, model: str, run: str, variant: str, name: str) -> Dict[int, float]:
        out = {}
        for seed, rec in evals.get(dataset, {}).get(model, {}).get(run, {}).items():
            x = _metric(rec, dataset, variant, name)
            if x is not None:
                out[seed] = x
        return out

    # ---- Table 7 (measured)
    rows = [("YOLO26-S (published checkpoint)", "native_pub", "fp32", "native", "native"),
            ("YOLO26-S (trained with this schedule)", "native", "fp32", "native", "native")]
    rows += [(f"SPLICE-B, arm ({a})", f"arm-{a}", "fp32", "nogate", f"arm-{a}") for a in ARMS]
    rows += [("(e) with gate, FP32", "arm-e", "fp32", "gate", "arm-e+gate"),
             ("(e) with gate, manifest sites", "arm-e", "manifest", "gate", "arm-e+gate+manifest")]
    rows += [(f"descriptor: {d}", f"desc-{d}", "fp32", "nogate", f"desc-{d}") for d in DESCRIPTORS]
    cols = [("coco", "AP"), ("coco", "AP75"), ("crowdhuman", "MR"), ("crowdhuman", "MR_heavy")]
    table, targets_cmp = [], []
    for label, model, run, variant, tkey in rows:
        line = {"row": label, "model": model, "run": run, "variant": variant}
        for ds, name in cols:
            st = _stats(values(ds, model, run, variant, name).values())
            line[f"{ds}_{name}"] = st
            tgt = PAPER_TARGETS["table7"].get("native" if tkey == "native" else tkey, {}).get(name)
            if tgt is not None and not (model == "native_pub" and ds == "crowdhuman") and not (model == "native" and ds == "coco"):
                targets_cmp.append({"table": "7", "row": label, "dataset": ds, "metric": name, "manuscript_target": tgt,
                                    "measured_mean": None if st is None else round(st["mean"], 4),
                                    "measured_seeds": 0 if st is None else st["n"],
                                    "status": "NOT_RUN" if st is None else "MEASURED"})
        for ds in ("crowdhuman", "coco"):
            d = diag.get(run_key(ds, model, cfg["seeds"][0]), {}).get("splits", {}).get("val", {}).get("forecast", {})
            line[f"{ds}_err"] = d.get("err_median")
            line[f"{ds}_cross"] = d.get("cross_share")
        for name in ("err", "cross"):
            tgt = PAPER_TARGETS["table7"].get(tkey, {}).get(name)
            if tgt is not None and variant == "nogate":
                got = line.get(f"crowdhuman_{name}")
                targets_cmp.append({"table": "7", "row": label, "dataset": "crowdhuman", "metric": name, "manuscript_target": tgt,
                                    "measured_mean": None if got is None else round(float(got), 4),
                                    "measured_seeds": 0 if got is None else 1, "status": "NOT_RUN" if got is None else "MEASURED"})
        table.append(line)

    # ---- declared tests (train channel)
    n_boot = int(cfg.get("bootstrap", 2000))
    boot_workers = max(1, min(16, int(args.threads), _cpus_allowed()) - 1)       # replicates run in forked processes
    tests: Dict[str, Dict] = {}

    def boot(tid: str, desc: str, dataset: str, name: str, cand: Tuple[str, str, str], ref: Tuple[str, str, str],
             margin: float, higher_better: bool) -> None:
        ca = evals.get(dataset, {}).get(cand[0], {}).get(cand[1], {})
        rb = evals.get(dataset, {}).get(ref[0], {}).get(ref[1], {})
        seeds = sorted(set(ca) & set(rb))
        t = {"description": desc, "dataset": dataset, "metric": name, "margin_points": margin, "seeds": seeds,
             "candidate": "/".join(cand), "reference": "/".join(ref), "alpha_one_sided": ALPHA}
        fa, fb, n_img = [], [], 0
        for s in seeds:
            a_fn, na = _metric_fn(ca[s]["_dir"], dataset, cand[2], name)
            b_fn, nb = _metric_fn(rb[s]["_dir"], dataset, ref[2], name)
            if a_fn is None or b_fn is None or na != nb:
                fa = []
                break
            fa.append(a_fn); fb.append(b_fn); n_img = na
        if not fa:
            t.update(decision="NOT_EVALUATED", reason="the evaluations of both arms (same images) are not available")
        else:
            st = M.two_level_bootstrap(fa, fb, n_img, n_boot, seed=int(cfg.get("base_seed", 1)), alpha=ALPHA,
                                       workers=boot_workers)
            t.update(st, images=n_img, **M.non_inferiority(st, margin, higher_better, min_seeds=3))
        tests[tid] = t

    E, A, F_ = "arm-e", "arm-a", "arm-f"
    boot("2", "AP, manifest sites - FP32 (arm e with gate): bound > -0.10", "coco", "AP", (E, "manifest", "gate"), (E, "fp32", "gate"),
         MARGINS["arithmetic"]["AP"], True)
    boot("3", "MR^-2, manifest sites - FP32 (arm e with gate): bound < +0.37", "crowdhuman", "MR", (E, "manifest", "gate"),
         (E, "fp32", "gate"), MARGINS["arithmetic"]["MR"], False)
    boot("9", "AP, (e) with gate - (a): bound > -0.30", "coco", "AP", (E, "fp32", "gate"), (A, "fp32", "nogate"),
         MARGINS["codesign"]["AP"], True)
    boot("10", "MR^-2, (e) with gate - (a): bound < +1.11", "crowdhuman", "MR", (E, "fp32", "gate"), (A, "fp32", "nogate"),
         MARGINS["codesign"]["MR"], False)
    boot("f_vs_e_AP", "arm (f) - arm (e), AP, co-design margin", "coco", "AP", (F_, "fp32", "gate"), (E, "fp32", "gate"),
         MARGINS["codesign"]["AP"], True)
    boot("f_vs_e_MR", "arm (f) - arm (e), MR^-2, co-design margin", "crowdhuman", "MR", (F_, "fp32", "gate"), (E, "fp32", "gate"),
         MARGINS["codesign"]["MR"], False)
    boot("f_vs_e_heavy", "arm (f) - arm (e), MR^-2 on heavily occluded persons, co-design margin", "crowdhuman", "MR_heavy",
         (F_, "fp32", "gate"), (E, "fp32", "gate"), MARGINS["codesign"]["MR"], False)
    r_no, r_g = values("crowdhuman", E, "fp32", "nogate", "recall_heavy"), values("crowdhuman", E, "fp32", "gate", "recall_heavy")
    common = sorted(set(r_no) & set(r_g))
    t12 = {"description": f"gate: recall on heavily occluded persons drops by at most {HEAVY_RECALL_DROP_MAX} pt (arm e, FP32)",
           "seeds": common, "criterion_points": HEAVY_RECALL_DROP_MAX}
    if not common:
        t12.update(decision="NOT_EVALUATED", reason="arm (e) on CrowdHuman was not evaluated")
    else:
        drop = sum(r_no[s] - r_g[s] for s in common) / len(common)
        ok = drop <= HEAVY_RECALL_DROP_MAX
        t12.update(drop_points=drop, per_seed={str(s): r_no[s] - r_g[s] for s in common},
                   decision=("PASS" if ok else "FAIL") if len(common) >= 3 else "NOT_QUALIFIED", would_pass=bool(ok))
    tests["12"] = t12
    fve = [tests[k]["decision"] for k in ("f_vs_e_AP", "f_vs_e_MR", "f_vs_e_heavy")]
    tests["f_vs_e"] = {"description": "declared comparison: if arm (f) is non-inferior to arm (e) on AP, MR^-2 and the heavy stratum, "
                                      "the exact exchange leaves the inference path",
                       "decision": "PASS" if all(d == "PASS" for d in fve) else "FAIL" if any(d == "FAIL" for d in fve)
                       else "NOT_QUALIFIED" if all(d in ("PASS", "NOT_QUALIFIED") for d in fve) else "NOT_EVALUATED",
                       "consequence_if_pass": "arm (f) is deployed: posterior, trust solve and translation leave inference"}
    core = [tests[k]["decision"] for k in ("2", "3", "9", "10", "12")]
    release = "DRY_RUN_NOT_A_MEASUREMENT" if cfg.get("dry_run") else \
        "TRAIN_CHANNEL_TESTS_PASS" if all(d == "PASS" for d in core) else \
        "TRAIN_CHANNEL_TESTS_FAIL" if any(d == "FAIL" for d in core) else "NOT_QUALIFIED"
    for tid, key in (("2", "test2"), ("3", "test3"), ("9", "test9"), ("10", "test10")):
        tg, t = PAPER_TARGETS["table6"][key], tests[tid]
        targets_cmp.append({"table": "6", "row": f"test {tid}", "dataset": t["dataset"], "metric": "effect; SE; bound",
                            "manuscript_target": f"{tg['effect']}; {tg['se']}; {tg['bound']}",
                            "measured_mean": None if "effect" not in t else
                            f"{t['effect']:.3f}; {t['se']:.3f}; {(t['lower'] if tid in ('2', '9') else t['upper']):.3f}",
                            "measured_seeds": len(t["seeds"]), "status": t["decision"]})
    targets_cmp.append({"table": "6", "row": "test 12", "dataset": "crowdhuman", "metric": "heavy recall drop (pt)",
                        "manuscript_target": PAPER_TARGETS["table6"]["test12"]["drop"],
                        "measured_mean": t12.get("drop_points"), "measured_seeds": len(common), "status": t12["decision"]})

    # ---- measured gain and the margins the rule would give
    gain = {}
    for ds, name, base_model in (("coco", "AP", "native_pub"), ("crowdhuman", "MR", "native")):
        b = _stats(values(ds, base_model, "fp32", "native", name).values())
        a = _stats(values(ds, A, "fp32", "nogate", name).values())
        if a and b:
            g = a["mean"] - b["mean"] if name == "AP" else b["mean"] - a["mean"]
            gain[ds] = {"metric": name, "yolo26s": b["mean"], "splice_b_arm_a": a["mean"], "gain_points": g,
                        "rule_margin_arithmetic_5pct": 0.05 * g, "rule_margin_codesign_15pct": 0.15 * g,
                        "declared_margins": {"arithmetic": MARGINS["arithmetic"][name], "codesign": MARGINS["codesign"][name]}}

    # ---- parameters of the train channel (Table 5 and section 3)
    params = []
    s0 = cfg["seeds"][0]
    for ds in cfg["datasets"]:
        reg = (evals.get(ds, {}).get(E, {}).get("fp32", {}).get(s0) or {}).get("regime") or {}
        dval = diag.get(run_key(ds, E, s0), {}).get("splits", {}).get("val", {})
        dcal = diag.get(run_key(ds, E, s0), {}).get("splits", {}).get("calib", {})
        tgt5 = PAPER_TARGETS["table5"]

        def put(name, value, target=None, source="evaluate, arm e"):
            params.append({"dataset": ds, "parameter": name, "measured": value, "manuscript_target": target, "source": source,
                           "status": "NOT_RUN" if value is None else "MEASURED"})

        for b in (1, 2, 4):
            put(f"skipped share phi at theta_det=0.25, {b}x{b}-cell blocks", (reg.get("skipped_share") or {}).get(f"block{b}"),
                tgt5["skipped_share"].get(ds))
        put("F mean", reg.get("F_mean"), tgt5["F_mean"].get(ds))
        put("F median", reg.get("F_p50"))
        put("F p99", reg.get("F_p99"))
        put("F log-std (shape)", reg.get("F_log_std"), tgt5["F_shape"].get(ds))
        put("F coefficient of variation", reg.get("F_cv"))
        put("N_f / F mean", reg.get("N_over_F_mean"), tgt5["N_over_F"])
        put("frames at the fragment cap (share)", reg.get("cap_frames_share"), tgt5["cap_frames_share"].get(ds))
        put("linkage budget exits (share)", reg.get("budget_exit_share"))
        put("forecast abstention share", (dval.get("forecast") or {}).get("abstain_share"), tgt5["abstain_share"], "diagnostics val")
        put("forecast clip share", (dval.get("forecast") or {}).get("clip_share"), None, "diagnostics val")
        put("sensitivity ||Q dN|| / ||dN|| (calibration split)", (dcal.get("sensitivity") or {}).get("q_over_response"),
            tgt5["sensitivity"], "diagnostics calib")
        put("sensitivity ||Q dN|| / ||M v|| (calibration split)", (dcal.get("sensitivity") or {}).get("q_over_probe"), None,
            "diagnostics calib")
        frr = (dval.get("admissibility") or {}).get("false_rejection_rate") or {}
        rates = [v["rate"] for v in frr.values() if v.get("rate") is not None]
        put("admissibility false-rejection rate, worst stratum (validation)", max(rates) if rates else None,
            tgt5["false_rejection_rate"], "diagnostics val")
        for lv in ("P3", "P4", "P5"):
            l3 = (dcal.get("lemma3") or {}).get(lv) or {}
            put(f"kappa_d at {lv} (max over images)", (l3.get("kappa_d") or {}).get("max"), None, "diagnostics calib")
            put(f"||Xi||_inf at {lv} (max over images)", (l3.get("xi_inf") or {}).get("max"), None, "diagnostics calib")
            lk = (dcal.get("leakage") or {}).get(lv) or {}
            put(f"low-pass leakage at {lv} (max / min over images)",
                None if not lk.get("lowpass") else f"{lk['lowpass']['max']:.5f} / {lk['lowpass']['min']:.5f}", None, "diagnostics calib")
    for p in params:
        if p["manuscript_target"] is not None:
            targets_cmp.append({"table": "5", "row": p["parameter"], "dataset": p["dataset"], "metric": "value",
                                "manuscript_target": p["manuscript_target"], "measured_mean": p["measured"],
                                "measured_seeds": 0 if p["measured"] is None else 1, "status": p["status"]})

    # ---- claim ledger
    ledger = []
    for ds, g in gain.items():
        ledger.append({"claim": f"SPLICE-B (arm a) against YOLO26-S on {ds}: {g['metric']} gain", "measured_points": g["gain_points"],
                       "status": "MEASURED"})
    for ds in ("coco", "crowdhuman"):
        if ds not in gain:
            ledger.append({"claim": f"SPLICE-B (arm a) against YOLO26-S on {ds}", "status": "NOT_EVALUATED",
                           "reason": "one of the two models was not evaluated on this dataset"})
    for tid in ("2", "3", "9", "10", "12", "f_vs_e"):
        t = tests[tid]
        ledger.append({"claim": f"test {tid}: {t['description']}", "status": t["decision"],
                       **{x: t[x] for x in ("effect", "se", "lower", "upper", "drop_points", "reason") if x in t}})
    ledger.append({"claim": "accuracy under the integer manifest (tests 2, 3)", "status": "PROXY",
                   "note": "'manifest sites' is the QAT model with the manifest's rounding sites active (fake quantization); "
                           "the bit-exact integer model M of the full detector needs the unit programs of gate G10"})

    # ---- files
    def write_csv(name: str, fieldnames: List[str], data: List[Dict]) -> None:
        with open(run_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in data:
                w.writerow(r)

    flat = []
    for line in table:
        r = {"row": line["row"], "model": line["model"], "run": line["run"], "variant": line["variant"]}
        for ds, name in cols:
            st = line[f"{ds}_{name}"]
            r.update({f"{ds}_{name}_mean": None if not st else round(st["mean"], 4), f"{ds}_{name}_min": None if not st else round(st["min"], 4),
                      f"{ds}_{name}_max": None if not st else round(st["max"], 4), f"{ds}_{name}_seeds": 0 if not st else st["n"]})
        r.update({k: line[k] for k in ("crowdhuman_err", "crowdhuman_cross", "coco_err", "coco_cross")})
        flat.append(r)
    write_csv("table7_measured.csv", list(flat[0].keys()), flat)
    write_csv("paper_target_comparison.csv", ["table", "row", "dataset", "metric", "manuscript_target", "measured_mean",
                                              "measured_seeds", "status"], targets_cmp)
    write_csv("train_channel_parameters.csv", ["dataset", "parameter", "measured", "manuscript_target", "source", "status"], params)
    write_json(run_dir / "tests.json", tests)
    write_json(run_dir / "claim_ledger.json", ledger)
    write_json(run_dir / "measured_gain.json", gain)

    md = ["# SPLICE-V detector suite: measured report (channel 'train')", "",
          f"suite: `{suite.root}`  ", f"scale: **{cfg['scale']}**, datasets {cfg['datasets']}, seeds {cfg['seeds']}, "
          f"epochs {cfg['epochs']}, QAT epochs {cfg['qat_epochs']}, bootstrap replicates {n_boot}  ",
          f"train-channel decision: **{release}**", ""]
    if cfg.get("dry_run"):
        md += ["> **DRY RUN.** Every number below comes from synthetic task results that only exercise the orchestration and",
               "> the reduction. Nothing here is a measurement.", ""]
    if cfg["scale"] != "full":
        md += [f"> Scale `{cfg['scale']}` is a shortened schedule on a subset; its numbers do not qualify any claim.", ""]
    md += ["## Stages", "", "| stage | tasks | completed |", "|---|---|---|"]
    md += [f"| {st} | {v['tasks']} | {v['completed']} |" for st, v in stage_status.items()]
    md += ["", "## Accuracy by readout arm and forecast descriptor (Table 7)", "",
           "Mean over seeds [min, max] (n seeds); AP in points, MR^-2 in points (lower is better). err.: median relative error of the "
           "forecast; cross: share of its transport crossing instances (CrowdHuman validation, first seed).", "",
           "| row | COCO AP | COCO AP75 | CrowdHuman MR^-2 | heavy MR^-2 | err. | cross |", "|---|---|---|---|---|---|---|"]
    f3 = lambda x: "-" if x is None else f"{float(x):.3f}"
    for line in table:
        md.append(f"| {line['row']} | {_fmt(line['coco_AP'])} | {_fmt(line['coco_AP75'])} | {_fmt(line['crowdhuman_MR'])} | "
                  f"{_fmt(line['crowdhuman_MR_heavy'])} | {f3(line['crowdhuman_err'])} | {f3(line['crowdhuman_cross'])} |")
    md += ["", "## Declared tests of the train channel (Table 6)", "",
           f"Two-level bootstrap over seeds and whole images, one-sided level {ALPHA}; a test with fewer than three seeds per arm is "
           "NOT_QUALIFIED, a test without its evaluations NOT_EVALUATED.", "",
           "| test | criterion | effect | SE | bound | margin | seeds | decision |", "|---|---|---|---|---|---|---|---|"]
    for tid in ("2", "3", "9", "10", "f_vs_e_AP", "f_vs_e_MR", "f_vs_e_heavy"):
        t = tests[tid]
        if "effect" in t:
            bound = t["lower"] if t["metric"] == "AP" else t["upper"]
            md.append(f"| {tid} | {t['description']} | {t['effect']:+.3f} | {t['se']:.3f} | {bound:+.3f} | {t['margin_points']} | "
                      f"{len(t['seeds'])} | {t['decision']} |")
        else:
            md.append(f"| {tid} | {t['description']} | - | - | - | {t['margin_points']} | {len(t['seeds'])} | {t['decision']} |")
    md.append(f"| 12 | {t12['description']} | {f3(t12.get('drop_points'))} pt | - | - | {HEAVY_RECALL_DROP_MAX} | {len(common)} | "
              f"{t12['decision']} |")
    md.append(f"| f vs e | {tests['f_vs_e']['description']} | - | - | - | - | - | {tests['f_vs_e']['decision']} |")
    if gain:
        md += ["", "## Measured gain of SPLICE-B over YOLO26-S and the margins of the rule", "",
               "| dataset | metric | YOLO26-S | SPLICE-B (a) | gain | 5% (arithmetic) | 15% (co-design) | declared margins |",
               "|---|---|---|---|---|---|---|---|"]
        for ds, g in gain.items():
            md.append(f"| {ds} | {g['metric']} | {g['yolo26s']:.2f} | {g['splice_b_arm_a']:.2f} | {g['gain_points']:.2f} | "
                      f"{g['rule_margin_arithmetic_5pct']:.3f} | {g['rule_margin_codesign_15pct']:.3f} | "
                      f"{g['declared_margins']['arithmetic']} / {g['declared_margins']['codesign']} |")
    md += ["", "## Parameters of the train channel (Table 5, section 3)", "", "| dataset | parameter | measured | source |", "|---|---|---|---|"]
    for p in params:
        val = "NOT_RUN" if p["measured"] is None else (f"{p['measured']:.4f}" if isinstance(p["measured"], float) else str(p["measured"]))
        md.append(f"| {p['dataset']} | {p['parameter']} | {val} | {p['source']} |")
    md += ["", "## How the numbers were obtained", "",
           "- COCO: val2017, 640 x 640 letterbox, FP32 inference, pycocotools matching and summary.",
           "- CrowdHuman: validation set, full-body boxes clipped to the image (YOLO labels cannot leave it), official matching rule "
           "with ignore regions; 'heavy' = visible ratio below 0.65, the other persons treated as ignore regions.",
           "- Calibration split: held out of the training set by a hash of the file name (COCO 1/24, CrowdHuman 1/12); the validation "
           "sets are used for the reported metrics only.",
           "- 'manifest sites': the QAT model evaluated with the manifest's rounding sites active (fake quantization). It is a proxy "
           "for the integer model M until the full-detector unit programs exist (gate G10).",
           "- The skip rule's block size is not fixed by the manuscript; the skipped share is reported for 1x1, 2x2 and 4x4 cells.",
           "- Manuscript values appear only in paper_target_comparison.csv, as targets.", ""]
    (run_dir / "MEASURED_REPORT.md").write_text("\n".join(md) + "\n")
    top = suite.stage_dir("reduce")
    for name in ("MEASURED_REPORT.md", "paper_target_comparison.csv", "table7_measured.csv", "train_channel_parameters.csv",
                 "tests.json", "claim_ledger.json", "measured_gain.json"):
        shutil.copyfile(run_dir / name, top / name)
    summary = {"release": {"decision": release, "core_tests": dict(zip(("2", "3", "9", "10", "12"), core))},
               "stages": stage_status, "evaluations": len(ev), "tests": {k: v["decision"] for k, v in tests.items()}}
    write_json(top / "summary.json", {"version": VERSION, **summary})
    return {"status": "DONE", **summary}


# ----------------------------------------------------------------------------- task wrapper, gate

STAGE_FUNCS = {"baseline": stage_baseline, "train": stage_train, "calibrate": stage_calibrate, "qat": stage_qat,
               "evaluate": stage_evaluate, "diagnostics": stage_diagnostics, "export": stage_export, "reduce": stage_reduce}


def classify(e: BaseException) -> Tuple[str, bool, int]:
    name, msg = type(e).__name__, str(e)
    if isinstance(e, Prerequisite):
        return "PREREQUISITE", False, EXIT_PREREQUISITE
    if "OutOfMemory" in name or "out of memory" in msg.lower():
        return "RESOURCE", False, 1
    infra = ("CUDA", "cuDNN", "cudaError", "NVML", "device-side", "GPU has fallen", "CUDA required but not available")
    if isinstance(e, RuntimeError) and any(s in msg for s in infra):
        return "INFRASTRUCTURE", True, EXIT_INFRASTRUCTURE
    if isinstance(e, OSError) and getattr(e, "errno", None) in (5, 110, 116, 121):
        return "INFRASTRUCTURE", True, EXIT_INFRASTRUCTURE
    if isinstance(e, (AssertionError, FloatingPointError)):
        return "SCIENTIFIC", False, 1
    return "CODE", False, 1


def _dry_fault(suite: Suite, stage: str, task: str) -> None:
    """Orchestration tests only (SPLICE_DRY_RUN=1): SPLICE_DRY_FAULTS="stage:task:kind[:times];..." makes a
    task fail its first `times` attempts.  kind: infra (retriable), continue (stops before the
    time limit), crash (dies without a summary), term (the job is ended by SIGTERM), fatal
    (non-retriable failure)."""
    for item in filter(None, os.environ.get("SPLICE_DRY_FAULTS", "").split(";")):
        parts = item.split(":")
        if len(parts) < 3 or parts[0] != stage or parts[1] != task:
            continue
        kind, times = parts[2], int(parts[3]) if len(parts) > 3 else 1
        counter = suite.manifest / "dry_faults" / f"{stage}__{task}__{kind}.count"
        counter.parent.mkdir(parents=True, exist_ok=True)
        done = int(counter.read_text()) if counter.exists() else 0
        if done >= times:
            continue
        counter.write_text(str(done + 1))
        if kind == "infra":
            raise RuntimeError("CUDA error: injected infrastructure fault (dry run)")
        if kind == "continue":
            raise ContinueLater({"epoch": done, "note": "injected continuation (dry run)"})
        if kind == "crash":
            os._exit(137)
        if kind == "term":                                   # Slurm ends the job: SIGTERM to the batch shell and the step
            import signal
            os.kill(os.getppid(), signal.SIGTERM)
            os._exit(143)
        raise AssertionError("injected non-retriable failure (dry run)")


def run_stage(args, root: Path) -> int:
    suite = Suite(args.suite)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    add_code_to_path(root)
    t0 = time.time()
    base = {"version": VERSION, "stage": args.stage, "task": args.task, "ordinal": args.ordinal, "started_utc": now(),
            "dry_run": DRY, "scale": suite.cfg.get("scale"), **slurm_env()}
    try:
        if not suite.cfg:
            raise Prerequisite(f"suite configuration missing: {suite.manifest / 'suite_config.json'}")
        if DRY:
            _dry_fault(suite, args.stage, args.task)
        res = STAGE_FUNCS[args.stage](suite, args, run_dir)
        for stale in ("failure_summary.json", "CONTINUE.json"):
            (run_dir / stale).unlink(missing_ok=True)
        write_json(run_dir / "summary.json", {**base, **res, "seconds": round(time.time() - t0, 1), "finished_utc": now(),
                                              "scientific_status": "DRY_RUN_NOT_A_MEASUREMENT" if DRY else "MEASURED"})
        log(f"[TASK] {args.stage}/{args.task}: {res.get('status', 'DONE')} in {time.time() - t0:.0f} s")
        return 0
    except ContinueLater as c:
        write_json(run_dir / "CONTINUE.json", {**base, "status": "CONTINUE", **c.info, "timestamp_utc": now()})
        log(f"[TASK] {args.stage}/{args.task}: stopped before the time limit after epoch {c.info.get('epoch')}; continues after requeue")
        return EXIT_CONTINUE
    except Exception as e:
        cls, retriable, code = classify(e)
        write_json(run_dir / "failure_summary.json", {**base, "status": f"FAILED_{cls}", "failure_class": cls, "retriable": retriable,
                                                      "scientific_status": "NOT_EVALUATED", "message": f"{type(e).__name__}: {e}",
                                                      "traceback": traceback.format_exc(), "timestamp_utc": now()})
        log(f"[FATAL][{cls}] {args.stage}/{args.task}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return code


def stage_gate(args) -> int:
    suite = Suite(args.suite)
    stage = args.stage_gate
    gdir = suite.stage_dir(stage) / "GATE"
    gdir.mkdir(parents=True, exist_ok=True)
    retry_count = 0
    try:
        retry_count = int((gdir / "RETRY_COUNT.txt").read_text().strip())
    except Exception:
        pass
    completed, retry, cont, fatal, optional_failed = [], [], [], [], []
    for ordinal, task in suite.tasks(stage):
        rd = suite.result_dir(stage, ordinal, task)
        parent = rd.parent
        ok = read_json(rd / "summary.json")
        fail = read_json(rd / "failure_summary.json") or read_json(rd / "shell_failure_summary.json")
        row = {"ordinal": ordinal, "task": task}
        if isinstance(ok, dict) and not fail:
            completed.append(row)
            continue
        opt = suite.optional(task)
        if (parent / "INFRA_RETRY_PENDING.json").exists():
            row.update(failure_class="INFRASTRUCTURE", message="infrastructure fault; the requeue was not possible")
            retry.append(row)
        elif isinstance(fail, dict):
            row.update(failure_class=fail.get("failure_class"), message=str(fail.get("message", fail.get("command", "")))[:400])
            if fail.get("retriable") or str(fail.get("failure_class", "")).upper() == "INFRASTRUCTURE":
                retry.append(row)
            else:
                (optional_failed if opt else fatal).append(row)
        elif (parent / "SKIPPED.txt").exists():
            row.update(failure_class="PREREQUISITE", message="skipped: the previous stage's gate did not pass")
            (optional_failed if opt else fatal).append(row)
        elif (parent / "INFRA_RETRY_EXHAUSTED.json").exists() or (parent / "CONTINUE_EXHAUSTED.json").exists():
            row.update(failure_class="RETRY_EXHAUSTED", message="requeue or continuation budget exhausted")
            (optional_failed if opt else fatal).append(row)
        elif (parent / "CONTINUE_PENDING.json").exists():
            row.update(failure_class="CONTINUE", message="stopped before its time limit; continues from its checkpoint")
            cont.append(row)
        else:
            row.update(failure_class="MISSING", message="no summary.json (not run, killed or timed out)")
            retry.append(row)
    budget_left = retry_count < int(args.stage_retries)
    if retry and not budget_left and all(suite.optional(r["task"]) for r in retry) and not fatal:
        optional_failed += [dict(r, message=r["message"] + " (retry budget exhausted; optional task)") for r in retry]
        retry = []
    if fatal:
        decision, code = "FAIL", GATE_FAIL
    elif retry:
        decision, code = "RETRY", GATE_RETRY                 # open continuations ride along with the retry
    elif cont:
        decision, code = "CONTINUE", GATE_CONTINUE           # only continuations are open: no retry budget is spent
    else:
        decision, code = ("PASS_WITH_OPTIONAL_FAILURES" if optional_failed else "PASS"), GATE_PASS
    total = len(suite.tasks(stage))
    write_json(gdir / "gate.json", {"stage": stage, "decision": decision, "completed": len(completed), "total": total,
                                    "fatal": fatal, "optional_failed": optional_failed, "retry": retry, "continue": cont,
                                    "retry_count": retry_count, "timestamp_utc": now(), "version": VERSION})
    lines = [f"stage {stage}: {decision} ({len(completed)}/{total} tasks completed)"]
    for label, rows in (("FATAL", fatal), ("OPTIONAL FAILURE", optional_failed), ("RETRY", retry), ("CONTINUE", cont)):
        lines += [f"  {label}: {r['ordinal']} {r['task']} [{r.get('failure_class')}] {r.get('message', '')}" for r in rows]
    (gdir / "gate.txt").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    for name in ("PASS", "FAIL", "RETRY"):
        (gdir / name).unlink(missing_ok=True)
    if code == GATE_PASS:
        (gdir / "PASS").write_text(decision + "\n")
    elif code in (GATE_RETRY, GATE_CONTINUE):
        (gdir / "RETRY").write_text(decision + "\n")
        (gdir / "missing_tasks.tsv").write_text("\n".join(["ordinal\ttask\tstage"] + [f"{r['ordinal']}\t{r['task']}\t{stage}"
                                                                                       for r in retry + cont]) + "\n")
    else:
        (gdir / "FAIL").write_text("FAIL\n")
    return code


# ----------------------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print_version", action="store_true")
    ap.add_argument("--print_identity", action="store_true")
    ap.add_argument("--code_root")
    ap.add_argument("--gpu_health_check_only", action="store_true")
    ap.add_argument("--gpu_health_out")
    ap.add_argument("--gpu_health_repeats", type=int, default=2)
    ap.add_argument("--gpu_health_matrix_size", type=int, default=1024)
    ap.add_argument("--semantic_preflight_only", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--advisory", action="store_true",
                    help="smoke scale: record a failed preflight / API probe and go on instead of stopping")
    ap.add_argument("--out_dir")
    ap.add_argument("--prepare_dataset_only", action="store_true")
    ap.add_argument("--data_root")
    ap.add_argument("--coco_root")
    ap.add_argument("--crowdhuman_root")
    ap.add_argument("--weights")
    ap.add_argument("--init_suite", action="store_true")
    ap.add_argument("--suite")
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--datasets", default="coco,crowdhuman")
    ap.add_argument("--seeds")
    ap.add_argument("--arms")
    ap.add_argument("--descriptors")
    ap.add_argument("--epochs")
    ap.add_argument("--qat_epochs")
    ap.add_argument("--val_limit")
    ap.add_argument("--max_iters")
    ap.add_argument("--fraction")
    ap.add_argument("--native_coco")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr_pretrained", type=float, default=0.001)
    ap.add_argument("--ckpt_root")
    ap.add_argument("--stage", choices=ALL_STAGES)
    ap.add_argument("--task")
    ap.add_argument("--ordinal", default="0000")
    ap.add_argument("--run_dir")
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--deadline_epoch")
    ap.add_argument("--stage_gate", choices=ALL_STAGES)
    ap.add_argument("--stage_retries", type=int, default=2)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_version:
        print(VERSION)
        return 0
    if args.print_identity:
        print(json.dumps(identity(), sort_keys=True))
        return 0
    if args.gpu_health_check_only:
        return gpu_health(args)
    if args.stage_gate:
        return stage_gate(args)
    if args.init_suite:
        return init_suite(args)
    root = code_root(args.code_root)
    if args.semantic_preflight_only:
        return semantic_preflight(args, root)
    if args.prepare_dataset_only:
        return prepare_dataset(args, root)
    if args.stage:
        return run_stage(args, root)
    print("nothing to do; see --help", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
