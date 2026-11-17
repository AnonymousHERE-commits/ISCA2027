#!/bin/bash -l
#SBATCH -A naiss2025-5-643-gpu
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH -t 00:20:00
#SBATCH -J splice_V1_0_0_parent
#SBATCH -o /nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V/logs/SPLICE_V1_0_0_PARENT_%j.out
#SBATCH -e /nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V/logs/SPLICE_V1_0_0_PARENT_%j.err

# SPLICE-V detector suite V1.0.0: launcher for the NAISS Arrhenius GH200 cluster.
# Trains, calibrates, quantizes, evaluates and exports the SPLICE detector of
# "SPLICE-V: A Representation-Guided Architecture for Graph-Wavelet Object Detection on
#  Versal" on COCO 2017 and CrowdHuman, and reduces the results of the paper's
#  'train' channel (Table 7, tests 2, 3, 9, 10, 12 of Table 6, the train rows of Table 5).
#
# Fresh full run:       sbatch run_splicev.sh
# Small pilot run:      sbatch run_splicev.sh pilot
# Orchestration smoke:  sbatch run_splicev.sh smoke   (a few iterations of every stage on the real data)
# Resume a suite:       sbatch run_splicev.sh resume /absolute/output/folder
# Re-reduce + repack:   sbatch run_splicev.sh postprocess /absolute/output/folder
#
# Server conventions (project, partition, storage addresses, live logs, venv, modules,
# numbered suites, gates with retries, GPU-health requeue, finalizer ZIP) are those of the
# group's other Arrhenius launchers; the storage root is the SPLICE-V project folder.
#
# Allocation: compute project naiss2025-5-643 (GPU nodes need the account suffix -gpu and the
# partition gpu), storage project naiss2025-6-428 under /nobackup/proj/disk.  The header has
# no -N line on purpose: on Arrhenius, naming -N anywhere allocates the whole node (four
# GH200) to the job; a job of this suite is one task on one GPU.  These #SBATCH lines are
# also the defaults of every job the suite submits (each overrides time, CPUs and memory).
# Check everything from the login node, without submitting:  bash run_splicev.sh check
SPLICE_RELEASE="SPLICE_V_CODE_V1_0_0"
export SPLICE_RELEASE
set -Eeuo pipefail
umask 002
# Preserve the original terminal/Slurm descriptors so that nested tee
# processes attached later never duplicate output.
exec 7>&1 8>&2

BASE="${SPLICE_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V}"
CODE_DIR="${BASE}/code"
DATA_DIR="${BASE}/datasets"
CACHE_DIR="${BASE}/cache"
LOGS_DIR="${BASE}/logs"
OUTPUTS_DIR="${BASE}/outputs"
CKPT_DIR="${BASE}/checkpoints"
# Stable live logs: every parent, worker, gate and finalizer mirrors its exact
# log into this pair, so one fixed tail command works across numbered suites.
LIVE_OUT="${LOGS_DIR}/SPLICE_V1_0_0_LIVE.out"
LIVE_ERR="${LOGS_DIR}/SPLICE_V1_0_0_LIVE.err"
LIVE_SUITE_FILE="${LOGS_DIR}/SPLICE_V1_0_0_LIVE_SUITE.txt"
VENV_DIR="${CODE_DIR}/venv_arrhenius_gh200"
PYTHON="${SPLICE_PYTHON:-${VENV_DIR}/bin/python}"
# The repository (github.com/AnonymousHERE-commits/ISCA2027) cloned into the code folder; every suite runs
# from its own snapshot of it (000_MANIFEST/code).
REPO_DIR="${SPLICE_REPO:-${CODE_DIR}/SPLICE-V}"
PY_SCRIPT_SOURCE="${CODE_DIR}/run_splicev_arrhenius.py"
EXPECTED_VERSION="1.0.0-arrhenius"
EXPECTED_VERSION_TAG="SPLICE_V1_0_0"
EXPECTED_RUNNER_BASENAME="run_splicev_arrhenius.py"
EXPECTED_SHELL_BASENAME="run_splicev.sh"
# Datasets (raw downloads; the suite builds its own views under datasets/splice_views)
COCO_ROOT="${SPLICE_COCO_ROOT:-${DATA_DIR}/coco}"
CROWDHUMAN_ROOT="${SPLICE_CROWDHUMAN_ROOT:-${DATA_DIR}/crowdhuman}"
WEIGHTS="${SPLICE_WEIGHTS:-${CKPT_DIR}/pretrained/yolo26s.pt}"
SPLICE_CACHE="${CACHE_DIR}/splice_V1_0_0"
# Slurm executes a spool copy (`slurm_script`), so $0 is not a reliable
# identity: nested submissions and snapshots use the canonical file in CODE_DIR.
RUNTIME_SHELL="$(readlink -f "$0")"
SELF="${SPLICE_CANONICAL_SHELL:-${CODE_DIR}/${EXPECTED_SHELL_BASENAME}}"
SHELL_IMPLEMENTATION_VERSION="1.0.0"
SHELL_VERSION_TAG="SPLICE_V1_0_0"

SCALE="${SPLICE_SCALE:-full}"
SEED="${SPLICE_SEED:-1}"
MAX_CONCURRENT="${SPLICE_MAX_CONCURRENT:-6}"
WORKER_CPUS="${SPLICE_WORKER_CPUS:-32}"
ADMIN_GPUS="${SPLICE_ADMIN_GPUS:-1}"
STAGE_RETRIES="${SPLICE_STAGE_RETRIES:-2}"
MAX_INFRA_REQUEUES="${SPLICE_MAX_INFRA_REQUEUES:-3}"
MAX_CONTINUATIONS="${SPLICE_MAX_CONTINUATIONS:-8}"
GPU_HEALTH_RETRIES="${SPLICE_GPU_HEALTH_RETRIES:-3}"
GPU_HEALTH_RETRY_SLEEP="${SPLICE_GPU_HEALTH_RETRY_SLEEP:-10}"
# Protocol selection (space separated lists; empty = the scale's registered protocol)
DATASETS="${SPLICE_DATASETS:-coco crowdhuman}"
SEEDS="${SPLICE_SEEDS:-}"
ARMS="${SPLICE_ARMS:-}"
DESCRIPTORS="${SPLICE_DESCRIPTORS:-}"
EPOCHS="${SPLICE_EPOCHS:-}"
QAT_EPOCHS="${SPLICE_QAT_EPOCHS:-}"
BATCH="${SPLICE_BATCH:-16}"
FRACTION="${SPLICE_FRACTION:-}"
VAL_LIMIT="${SPLICE_VAL_LIMIT:-}"
NATIVE_COCO="${SPLICE_NATIVE_COCO:-}"
DRY_RUN="${SPLICE_DRY_RUN:-0}"
PREREQUISITE_EXIT_CODE=33
ORCHESTRATION_EXIT_CODE=70
ARTIFACT_MISSING_EXIT_CODE=74
INFRASTRUCTURE_EXIT_CODE=75
CONTINUE_EXIT_CODE=76
GATE_PASS=0
GATE_RETRY=10
GATE_CONTINUE=11
GATE_FAIL=20

# Stage graph.  The chain runs strictly in order; after the last chain stage the
# evaluation stages run in parallel and are joined before the reduction.
CHAIN_STAGES=(baseline train calibrate qat)
PARALLEL_STAGES=(evaluate diagnostics export)
ALL_STAGES=("${CHAIN_STAGES[@]}" "${PARALLEL_STAGES[@]}" reduce)

stage_dir_of() {
  case "$1" in
    baseline) echo 010_BASELINE ;;
    train) echo 020_TRAIN ;;
    calibrate) echo 030_CALIBRATION ;;
    qat) echo 040_QAT ;;
    evaluate) echo 070_EVALUATE ;;
    diagnostics) echo 080_DIAGNOSTICS ;;
    export) echo 085_EXPORT ;;
    reduce) echo 900_REDUCE ;;
    *) echo "[FATAL] unknown stage $1" >&2; return 2 ;;
  esac
}

stage_number_of() { local d; d="$(stage_dir_of "$1")"; printf '%s\n' "${d%%_*}"; }

next_chain_stage() {
  case "$1" in
    baseline) echo train ;;
    train) echo calibrate ;;
    calibrate) echo qat ;;
    *) echo "" ;;
  esac
}

prev_chain_stage() {
  case "$1" in
    train) echo baseline ;;
    calibrate) echo train ;;
    qat) echo calibrate ;;
    *) echo "" ;;
  esac
}

is_chain_stage() { local s; for s in "${CHAIN_STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
is_parallel_stage() { local s; for s in "${PARALLEL_STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
# Stages that use the GPU (health preflight, --require_cuda); reduce is CPU work.
is_gpu_stage() { case "$1" in reduce) return 1 ;; *) return 0 ;; esac; }

# Default wall time / memory / CPUs per stage on one GH200 (72 cores, 96 GB HBM per GPU);
# each can be overridden with SPLICE_TIME_<STAGE> / SPLICE_MEM_<STAGE> / SPLICE_CPUS_<STAGE>.
stage_time_default() {
  if [[ "${SCALE}" == preflight ]]; then echo 01:00:00; return; fi
  if [[ "${SCALE}" == pilot ]]; then
    case "$1" in train|qat) echo 08:00:00 ;; *) echo 03:00:00 ;; esac
    return
  fi
  # train / qat: one job is one chunk of at most this long; a training that needs more stops
  # cleanly before the limit and continues from its checkpoint (up to SPLICE_MAX_CONTINUATIONS).
  case "$1" in
    baseline) echo 02:00:00 ;;
    train) echo 24:00:00 ;;
    calibrate) echo 04:00:00 ;;
    qat) echo 24:00:00 ;;
    evaluate) echo 04:00:00 ;;
    diagnostics) echo 08:00:00 ;;
    export) echo 01:00:00 ;;
    reduce) echo 08:00:00 ;;
    *) echo 12:00:00 ;;
  esac
}

stage_mem_default() {
  case "$1" in
    reduce|export) echo 64G ;;
    *) echo 110G ;;
  esac
}

stage_cpus_default() {
  case "$1" in
    reduce|export) echo 16 ;;
    *) echo "${WORKER_CPUS}" ;;
  esac
}

# Slurm time string (M, M:S, H:M:S, D-H, D-H:M, D-H:M:S) -> seconds; empty for anything else.
time_to_seconds() {
  local t="$1" d=0 h=0 m=0 s=0 rest
  [[ "${t}" =~ ^[0-9:-]+$ ]] || return 0
  if [[ "${t}" == *-* ]]; then
    d="${t%%-*}"; rest="${t#*-}"
    IFS=: read -r h m s <<< "${rest}"
  else
    case "${t//[^:]/}" in
      "") m="${t}" ;;
      ":") IFS=: read -r m s <<< "${t}" ;;
      "::") IFS=: read -r h m s <<< "${t}" ;;
      *) return 0 ;;
    esac
  fi
  printf '%d\n' "$(( 10#${d:-0} * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${s:-0} ))"
}

seconds_to_time() { local t="$1"; printf '%d-%02d:%02d:%02d\n' "$((t / 86400))" "$((t % 86400 / 3600))" "$((t % 3600 / 60))" "$((t % 60))"; }

# The partition's wall-time limit in seconds (empty when unknown or unlimited).  The limits
# of Arrhenius' partitions are read from sinfo; the answer is cached per suite.
partition_limit_seconds() {
  local cache="" limit="" part="${SBATCH_PARTITION:-${SPLICE_PARTITION:-gpu}}"
  [[ -n "${SPLICE_SUITE:-}" && -d "${SPLICE_SUITE}/000_MANIFEST" ]] && cache="${SPLICE_SUITE}/000_MANIFEST/partition_time_limit.txt"
  if [[ -n "${cache}" && -s "${cache}" ]]; then
    limit="$(head -n 1 "${cache}")"
  elif command -v sinfo >/dev/null 2>&1; then
    limit="$(sinfo -h -p "${part}" -o %l 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
    [[ -n "${cache}" && -n "${limit}" ]] && printf '%s\n' "${limit}" > "${cache}"
  fi
  time_to_seconds "${limit}"
}

stage_time() {
  local u want limit want_s
  u="SPLICE_TIME_$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  want="${!u:-$(stage_time_default "$1")}"
  limit="$(partition_limit_seconds)"; want_s="$(time_to_seconds "${want}")"
  if [[ -n "${limit}" && -n "${want_s}" ]] && (( limit > 0 && want_s > limit )); then
    want="$(seconds_to_time "${limit}")"
  fi
  printf '%s\n' "${want}"
}
stage_mem() { local u; u="SPLICE_MEM_$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"; printf '%s\n' "${!u:-$(stage_mem_default "$1")}"; }
stage_cpus() { local u; u="SPLICE_CPUS_$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"; printf '%s\n' "${!u:-$(stage_cpus_default "$1")}"; }

case "${SPLICE_ROLE:-orchestrator}:${1:-full}" in
  orchestrator:check|orchestrator:help|orchestrator:-h|orchestrator:--help) ;;      # read-only modes create nothing
  *) mkdir -p "${LOGS_DIR}" "${OUTPUTS_DIR}" "${SPLICE_CACHE}" "${DATA_DIR}" "${CKPT_DIR}/pretrained" ;;
esac

# -----------------------------------------------------------------------------
# live monitor and per-stage log attachment
# -----------------------------------------------------------------------------
init_live_monitor() {
  local mode="${1:-full}"
  : > "${LIVE_OUT}"
  : > "${LIVE_ERR}"
  printf 'SPLICE-V V1.0.0 LIVE MONITOR\nstarted_utc=%s\nparent_job=%s\nmode=%s\nstatus=INITIALIZING\n\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_JOB_ID:-manual}" "${mode}" >> "${LIVE_OUT}"
  printf '%s\n' "${SLURM_JOB_ID:-manual}" > "${LIVE_SUITE_FILE}.parent"
}

attach_stage_logs() {
  local exact_out="$1" exact_err="$2" stage_label="${3:-stage}"
  mkdir -p "$(dirname "${exact_out}")" "$(dirname "${exact_err}")"
  touch "${exact_out}" "${exact_err}" "${LIVE_OUT}" "${LIVE_ERR}"
  printf '\n===== %s | job=%s | task=%s | utc=%s =====\n' \
    "${stage_label}" "${SLURM_JOB_ID:-manual}" "${SLURM_ARRAY_TASK_ID:-NA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LIVE_OUT}"
  exec > >(tee -a "${exact_out}" "${LIVE_OUT}" >&7) \
       2> >(tee -a "${exact_err}" "${LIVE_ERR}" >&8)
}

attach_live_only() {
  local stage_label="${1:-bootstrap}"
  touch "${LIVE_OUT}" "${LIVE_ERR}"
  printf '\n===== %s | job=%s | utc=%s =====\n' \
    "${stage_label}" "${SLURM_JOB_ID:-manual}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LIVE_OUT}"
  exec > >(tee -a "${LIVE_OUT}" >&7) 2> >(tee -a "${LIVE_ERR}" >&8)
}

# -----------------------------------------------------------------------------
# truthful shell-level failure artifacts (a task that dies before the Python
# runner writes failure_summary.json still leaves a machine-readable record)
# -----------------------------------------------------------------------------
CURRENT_RUN_DIR=""
CURRENT_RUN_PARENT=""
CURRENT_LOG_ERR=""
CURRENT_REGISTRY_ACTIVE=0
SHELL_ERROR_HANDLER_ACTIVE=0

write_shell_failure_artifact() {
  local rc="$1" line="$2" cmd="$3" failure_class="$4" retriable="$5"
  local target="${CURRENT_RUN_DIR}"
  if [[ -z "${target}" && -n "${SPLICE_SUITE:-}" ]]; then
    target="${SPLICE_SUITE}/000_MANIFEST/SHELL_FAILURES"
  fi
  [[ -n "${target}" ]] || return 0
  mkdir -p "${target}"
  local py="${PYTHON}"
  [[ -x "${py}" ]] || py="$(command -v python3 || true)"
  if [[ -n "${py}" ]]; then
    SPLICE_FAILURE_TARGET="${target}" SPLICE_FAILURE_RC="${rc}" SPLICE_FAILURE_LINE="${line}" \
    SPLICE_FAILURE_COMMAND="${cmd}" SPLICE_FAILURE_CLASS="${failure_class}" \
    SPLICE_FAILURE_RETRIABLE="${retriable}" SPLICE_FAILURE_LOG_ERR="${CURRENT_LOG_ERR}" \
    RUNTIME_SHELL="${RUNTIME_SHELL}" SELF="${SELF}" "${py}" - <<'PYFAIL'
import json, os, socket
from datetime import datetime, timezone
from pathlib import Path

target = Path(os.environ['SPLICE_FAILURE_TARGET'])
obj = {
  'version': '1.0.0-arrhenius',
  'status': 'FAILED_SHELL',
  'failure_class': os.environ.get('SPLICE_FAILURE_CLASS', 'ORCHESTRATION'),
  'retriable': os.environ.get('SPLICE_FAILURE_RETRIABLE', 'false').lower() == 'true',
  'scientific_status': 'NOT_EVALUATED',
  'exit_code': int(os.environ.get('SPLICE_FAILURE_RC', '1')),
  'line': os.environ.get('SPLICE_FAILURE_LINE', 'unknown'),
  'command': os.environ.get('SPLICE_FAILURE_COMMAND', 'unknown'),
  'role': os.environ.get('SPLICE_ROLE', 'orchestrator'),
  'stage': os.environ.get('SPLICE_STAGE', ''),
  'task': os.environ.get('TASK', ''),
  'ordinal': os.environ.get('ORDINAL', ''),
  'log_err': os.environ.get('SPLICE_FAILURE_LOG_ERR', ''),
  'job_id': os.environ.get('SLURM_JOB_ID', 'manual'),
  'array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID', ''),
  'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID', ''),
  'restart_count': os.environ.get('SLURM_RESTART_COUNT', '0'),
  'node': os.environ.get('SLURMD_NODENAME', socket.gethostname()),
  'runtime_shell': os.environ.get('RUNTIME_SHELL', ''),
  'canonical_shell': os.environ.get('SELF', ''),
  'timestamp_utc': datetime.now(timezone.utc).isoformat(),
}
if target.name == 'RESULT':
    names = ['shell_failure_summary.json']
    if not (target / 'failure_summary.json').exists():
        names.insert(0, 'failure_summary.json')
else:
    names = [f"shell_failure_{obj['job_id']}_{obj['array_task_id'] or 'NA'}.json"]
for name in names:
    tmp = target / (name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(target / name)
PYFAIL
  else
    printf 'status=FAILED_SHELL\nclass=%s\nrc=%s\nline=%s\ncommand=%s\n' \
      "${failure_class}" "${rc}" "${line}" "${cmd}" > "${target}/shell_failure_summary.txt"
  fi
}

failure_class_of_rc() {
  case "$1" in
    "${INFRASTRUCTURE_EXIT_CODE}") echo "INFRASTRUCTURE true" ;;
    "${PREREQUISITE_EXIT_CODE}") echo "PREREQUISITE false" ;;
    "${ARTIFACT_MISSING_EXIT_CODE}") echo "ARTIFACT_MISSING false" ;;
    *)
      # 128+signal: the process was killed (out of memory, node fault, time limit,
      # preemption).  The stage gate resubmits such a task within its retry budget;
      # a training task then continues from its last checkpoint.
      if [[ "$1" =~ ^[0-9]+$ ]] && (( $1 > 128 )); then echo "KILLED_BY_SIGNAL true"; else echo "ORCHESTRATION false"; fi ;;
  esac
}

on_shell_error() {
  local rc=$? line="${BASH_LINENO[0]:-unknown}" cmd="${BASH_COMMAND:-unknown}"
  if [[ "${SHELL_ERROR_HANDLER_ACTIVE}" == 1 ]]; then
    exit "${rc}"
  fi
  SHELL_ERROR_HANDLER_ACTIVE=1
  trap - ERR
  local failure_class retriable
  read -r failure_class retriable < <(failure_class_of_rc "${rc}")
  write_shell_failure_artifact "${rc}" "${line}" "${cmd}" "${failure_class}" "${retriable}" || true
  if [[ "${CURRENT_REGISTRY_ACTIVE}" == 1 && -n "${SPLICE_SUITE:-}" ]]; then
    append_registry "FAILED_SHELL_${failure_class}" "${CURRENT_RUN_DIR:-UNKNOWN}" || true
  fi
  echo "[FATAL][SHELL] version=${SHELL_IMPLEMENTATION_VERSION} role=${SPLICE_ROLE:-orchestrator} class=${failure_class} rc=${rc} line=${line}" >&2
  echo "[FATAL][SHELL] command=${cmd}" >&2
  echo "[FATAL][SHELL] runtime_shell=${RUNTIME_SHELL}" >&2
  echo "[FATAL][SHELL] canonical_shell=${SELF}" >&2
  package_after_gate_crash || true
  exit "${rc}"
}
trap on_shell_error ERR

# A crashed gate would otherwise stall the chain silently: package what exists
# (the suite can later be continued with the 'resume' mode).
package_after_gate_crash() {
  [[ "${SPLICE_ROLE:-}" == gate && -n "${SPLICE_SUITE:-}" ]] || return 0
  local marker="${SPLICE_SUITE}/000_MANIFEST/GATE_CRASH_FINALIZER_SUBMITTED"
  [[ -f "${marker}" ]] && return 0
  command -v sbatch >/dev/null 2>&1 || return 0
  local jid
  jid="$(sbatch --parsable --job-name=splice_V1_0_0_pack --time=04:00:00 --cpus-per-task=8 --mem=64G \
    --gpus="${ADMIN_GPUS}" --output=/dev/null --error=/dev/null \
    --export="ALL,SPLICE_CANONICAL_SHELL=${SELF},SPLICE_ROLE=finalizer,SPLICE_SUITE=${SPLICE_SUITE}" "${SELF}" 2>/dev/null)" || return 0
  printf '%s\n' "${jid%%;*}" > "${marker}"
  echo "[FATAL][SHELL] gate crashed; finalizer ${jid%%;*} submitted to package the partial suite" >&2
}

on_shell_exit() {
  local rc=$?
  if [[ ${rc} -ne 0 && "${SHELL_ERROR_HANDLER_ACTIVE}" != 1 ]]; then
    local failure_class retriable
    read -r failure_class retriable < <(failure_class_of_rc "${rc}")
    write_shell_failure_artifact "${rc}" "EXIT" "explicit shell exit" "${failure_class}" "${retriable}" || true
    if [[ "${CURRENT_REGISTRY_ACTIVE}" == 1 && -n "${SPLICE_SUITE:-}" ]]; then
      append_registry "FAILED_SHELL_${failure_class}" "${CURRENT_RUN_DIR:-UNKNOWN}" || true
      CURRENT_REGISTRY_ACTIVE=0
    fi
    package_after_gate_crash || true
  fi
}
trap on_shell_exit EXIT

# SIGTERM: Slurm ends a job at its time limit, on preemption and on scancel.  The exit trap
# alone sees status 0 then, so the reason is recorded here; the stage gate resubmits such a
# task within its retry budget and a training continues from its last checkpoint.
on_shell_signal() {
  local signal="$1" rc="$2"
  trap - TERM
  if [[ "${SHELL_ERROR_HANDLER_ACTIVE}" != 1 ]]; then
    SHELL_ERROR_HANDLER_ACTIVE=1
    if [[ -z "${CURRENT_RUN_DIR}" || ! -f "${CURRENT_RUN_DIR}/summary.json" ]]; then
      write_shell_failure_artifact "${rc}" "SIG${signal}" \
        "job received SIG${signal} (time limit, preemption or cancellation)" KILLED_BY_SIGNAL true || true
    fi
    if [[ "${CURRENT_REGISTRY_ACTIVE}" == 1 && -n "${SPLICE_SUITE:-}" ]]; then
      append_registry "KILLED_BY_SIGNAL" "${CURRENT_RUN_DIR:-UNKNOWN}" || true
      CURRENT_REGISTRY_ACTIVE=0
    fi
    echo "[FATAL][SHELL] role=${SPLICE_ROLE:-orchestrator} received SIG${signal} (time limit, preemption or cancellation)" >&2
    package_after_gate_crash || true
  fi
  exit "${rc}"
}
trap 'on_shell_signal TERM 143' TERM

strip_jobid() { local x="$1"; printf '%s\n' "${x%%;*}"; }

current_slurm_task_id() {
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    printf '%s_%s\n' "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}"
  else
    printf '%s\n' "${SLURM_JOB_ID:-unknown}"
  fi
}

# Unix time at which Slurm ends this job (0 when unknown): long trainings stop
# cleanly before it and continue after a requeue.
job_deadline_epoch() {
  local end="" epoch=""
  if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then printf '%s\n' "${SLURM_JOB_END_TIME}"; return 0; fi
  if command -v squeue >/dev/null 2>&1; then
    end="$(squeue -h -j "$(current_slurm_task_id)" -o %e 2>/dev/null | head -n 1 || true)"
    if [[ -n "${end}" && "${end}" != "N/A" && "${end}" != "NONE" ]]; then
      epoch="$(date -d "${end}" +%s 2>/dev/null || true)"
    fi
  fi
  [[ "${epoch}" =~ ^[0-9]+$ ]] || epoch=0
  printf '%s\n' "${epoch}"
}

# Returns 2 when a completed result already exists (resume), 0 otherwise; an
# incomplete earlier attempt is archived under ATTEMPTS/ rather than deleted.
archive_previous_attempt() {
  local run_parent="$1" run_dir="${run_parent}/RESULT" restart="${SLURM_RESTART_COUNT:-0}" stamp
  if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    if [[ -f "${run_dir}/summary.json" && ! -f "${run_dir}/failure_summary.json" ]]; then
      return 2
    fi
    stamp="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${run_parent}/ATTEMPTS"
    mv "${run_dir}" "${run_parent}/ATTEMPTS/RESULT_restart${restart}_${stamp}_$$"
  fi
  return 0
}

infrastructure_failure_detected() {
  local run_dir="$1" log_err="$2" rc="${3:-1}"
  [[ "${rc}" -eq "${INFRASTRUCTURE_EXIT_CODE}" ]] && return 0
  if [[ -f "${run_dir}/failure_summary.json" ]]; then
    if "${PYTHON}" - "${run_dir}/failure_summary.json" <<'PYINFRA'
import json, sys
try:
    x = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
fc = str(x.get('failure_class', '')).upper(); status = str(x.get('status', '')).upper()
raise SystemExit(0 if fc == 'INFRASTRUCTURE' or 'INFRASTRUCTURE' in status or bool(x.get('retriable', False)) else 1)
PYINFRA
    then
      return 0
    fi
  fi
  if [[ -f "${log_err}" ]]; then
    grep -Eiq 'cudaErrorDevicesUnavailable|CUDA-capable device\(s\) is/are busy or unavailable|uncorrectable ECC error|CUDA driver initialization failed|CUDA initialization error|NVIDIA driver is shutting down|GPU has fallen off the bus|device is lost|system not yet initialized|failed call to cuInit|driver/library version mismatch' "${log_err}" && return 0
  fi
  return 1
}

requeue_current_task() {
  local run_parent="$1" reason="$2" target count_file lock_file count node
  target="$(current_slurm_task_id)"
  count_file="${run_parent}/INFRA_REQUEUE_COUNT.txt"
  lock_file="${run_parent}/INFRA_REQUEUE_COUNT.lock"
  mkdir -p "${run_parent}"
  exec 6>"${lock_file}"; flock -x 6
  count=0; [[ -s "${count_file}" ]] && count="$(cat "${count_file}")"
  [[ "${count}" =~ ^[0-9]+$ ]] || count=0
  if (( count >= MAX_INFRA_REQUEUES )); then
    cat > "${run_parent}/INFRA_RETRY_EXHAUSTED.json" <<EOF
{"version":"1.0.0-arrhenius","status":"INFRA_RETRY_EXHAUSTED","task":"${target}","attempts":${count},"reason":$(printf '%s' "${reason}" | "${PYTHON}" -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}
EOF
    flock -u 6
    echo "[INFRA] retry budget exhausted for ${target}: ${reason}" >&2
    return 1
  fi
  count=$((count+1)); printf '%s\n' "${count}" > "${count_file}"
  flock -u 6
  node="${SLURMD_NODENAME:-$(hostname -s)}"
  cat > "${run_parent}/INFRA_REQUEUE_${count}.json" <<EOF
{"version":"1.0.0-arrhenius","status":"INFRA_REQUEUE_REQUESTED","task":"${target}","attempt":${count},"max_attempts":${MAX_INFRA_REQUEUES},"node":"${node}","reason":$(printf '%s' "${reason}" | "${PYTHON}" -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}
EOF
  echo "[INFRA] requeueing ${target}; attempt ${count}/${MAX_INFRA_REQUEUES}; node=${node}; reason=${reason}" >&2
  scontrol update JobId="${target}" ExcNodeList="${node}" >/dev/null 2>&1 || true
  if scontrol requeue "${target}"; then
    exit 0
  fi
  # No requeue on this system or for this job: the stage gate retries the task instead.
  printf '{"version":"1.0.0-arrhenius","status":"INFRA_RETRY_PENDING","task":"%s","attempt":%s}\n' \
    "${target}" "${count}" > "${run_parent}/INFRA_RETRY_PENDING.json"
  echo "[INFRA] scontrol requeue failed; the stage gate retries this task" >&2
  return 1
}

# A training that stopped cleanly before its wall-clock limit (runner exit 76,
# checkpoint written) is requeued and continues from that checkpoint.
continue_current_task() {
  local run_parent="$1" target count_file count
  target="$(current_slurm_task_id)"
  count_file="${run_parent}/CONTINUATION_COUNT.txt"
  mkdir -p "${run_parent}"
  count=0; [[ -s "${count_file}" ]] && count="$(cat "${count_file}")"
  [[ "${count}" =~ ^[0-9]+$ ]] || count=0
  if (( count >= MAX_CONTINUATIONS )); then
    printf '{"version":"1.0.0-arrhenius","status":"CONTINUE_EXHAUSTED","task":"%s","continuations":%s}\n' \
      "${target}" "${count}" > "${run_parent}/CONTINUE_EXHAUSTED.json"
    echo "[CONTINUE] continuation budget exhausted for ${target} (${count}/${MAX_CONTINUATIONS}); raise SPLICE_MAX_CONTINUATIONS or SPLICE_TIME_<STAGE> and resume" >&2
    return 1
  fi
  count=$((count+1)); printf '%s\n' "${count}" > "${count_file}"
  printf '{"version":"1.0.0-arrhenius","status":"CONTINUE_REQUEUE_REQUESTED","task":"%s","continuation":%s,"max":%s}\n' \
    "${target}" "${count}" "${MAX_CONTINUATIONS}" > "${run_parent}/CONTINUE_${count}.json"
  echo "[CONTINUE] requeueing ${target} to continue from its checkpoint; continuation ${count}/${MAX_CONTINUATIONS}" >&2
  if ! scontrol requeue "${target}"; then
    # No requeue on this system or for this job: leave a marker; the stage gate resubmits the
    # task (without spending the stage's retry budget) and it continues from its checkpoint.
    printf '{"version":"1.0.0-arrhenius","status":"CONTINUE_PENDING","task":"%s","continuation":%s}\n' \
      "${target}" "${count}" > "${run_parent}/CONTINUE_PENDING.json"
    echo "[CONTINUE] scontrol requeue failed; the stage gate resubmits this task and it continues from its checkpoint" >&2
  fi
  exit 0
}

gpu_health_preflight() {
  local py_snapshot="$1" run_dir="$2" attempt rc=1
  for attempt in $(seq 1 "${GPU_HEALTH_RETRIES}"); do
    if "${PYTHON}" "${py_snapshot}" --gpu_health_check_only \
      --gpu_health_out "${run_dir}/GPU_HEALTH_PREFLIGHT.json" \
      --gpu_health_repeats 2 --gpu_health_matrix_size 1024; then
      rc=0
    else
      rc=$?
    fi
    [[ ${rc} -eq 0 ]] && return 0
    echo "[GPU_HEALTH] attempt ${attempt}/${GPU_HEALTH_RETRIES} failed rc=${rc}" >&2
    (( attempt < GPU_HEALTH_RETRIES )) && sleep "${GPU_HEALTH_RETRY_SLEEP}"
  done
  return "${rc}"
}

usage() {
  cat <<'EOF'
SPLICE-V V1.0.0 launcher (SPLICE detector on COCO 2017 and CrowdHuman, SPLICE-V 'train' channel)

Fresh full suite (registered protocol: seven readout arms, descriptor variants, three seeds):
  sbatch run_splicev.sh
Small end-to-end pilot on the cluster (every stage, 2 epochs on 5% of the training images):
  sbatch run_splicev.sh pilot
Orchestration smoke test (every stage and gate on the real data with a few iterations;
checks modules, venv, datasets, the Ultralytics/YOLO26 interfaces and the Slurm chain in
about an hour; its numbers carry no scientific meaning):
  sbatch run_splicev.sh smoke
Resume an existing suite from its first incomplete stage (completed tasks are kept, trainings
continue from their checkpoints):
  sbatch run_splicev.sh resume /absolute/output/folder
Re-run the reduction and repackage an existing suite:
  sbatch run_splicev.sh postprocess /absolute/output/folder
Check storage, directories, code, account, partition and datasets from the login node (submits
nothing, writes nothing):
  bash run_splicev.sh check

Environment overrides:
  SPLICE_BASE=/path (default: the fixed SPLICE-V project folder)
  SPLICE_SCALE=full|pilot   SPLICE_SEED=1
  SPLICE_DATASETS="coco crowdhuman"   SPLICE_SEEDS="0 1 2"   SPLICE_ARMS="a e f g"
  SPLICE_DESCRIPTORS=0|1 (forecast-descriptor variants of arm e)   SPLICE_NATIVE_COCO=0|1
  SPLICE_EPOCHS=50   SPLICE_QAT_EPOCHS=12   SPLICE_BATCH=16   SPLICE_FRACTION=1.0   SPLICE_VAL_LIMIT=0
  SPLICE_MAX_CONCURRENT=6   SPLICE_WORKER_CPUS=32   SPLICE_ADMIN_GPUS=1
  SPLICE_STAGE_RETRIES=2    SPLICE_MAX_INFRA_REQUEUES=3   SPLICE_MAX_CONTINUATIONS=8
  SPLICE_TIME_<STAGE>=HH:MM:SS  SPLICE_MEM_<STAGE>=110G  SPLICE_CPUS_<STAGE>=32   (e.g. SPLICE_TIME_TRAIN=48:00:00)
    A train or qat task is one chunk of at most SPLICE_TIME_<STAGE> (default 24 h, lowered to the
    partition's limit); a longer training stops before the limit and continues from its checkpoint.
  SPLICE_COCO_ROOT=/path  SPLICE_CROWDHUMAN_ROOT=/path  SPLICE_WEIGHTS=/path/yolo26s.pt  SPLICE_REPO=/path/SPLICE-V
  SBATCH_ACCOUNT / SBATCH_PARTITION override the account and partition of every job of the suite

The directory/address layout is fixed under:
  /nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V
    code/         SPLICE-V/ (git clone of github.com/AnonymousHERE-commits/ISCA2027), run_splicev.sh,
                  run_splicev_arrhenius.py, venv_arrhenius_gh200/
    datasets/     coco/ (train2017, val2017, annotations), crowdhuman/ (odgt files, images),
                  splice_views/ (built by the suite: train_fit / calib / val roles)
    checkpoints/  pretrained/yolo26s.pt, SPLICE_V1_0_0_S####/<run>/{last,trained,calibrated,qat}.pt
    cache/        splice_V1_0_0/ (suite counter, venv lock), torch/, pip/, ultralytics/
    logs/         SPLICE_V1_0_0_LIVE.out/.err and one ordered log pair per job
    outputs/      SPLICE_V1_0_0_SUITE_####/
EOF
}

# -----------------------------------------------------------------------------
# runtime
# -----------------------------------------------------------------------------
runtime_exports() {
  # One BLAS/OpenMP thread per process: the neck's CPU steps run in a process pool and
  # the runner sets torch's own thread count from --threads.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
  unset PYTHONPATH || true
  export TORCH_HOME="${CACHE_DIR}/torch" XDG_CACHE_HOME="${CACHE_DIR}/xdg"
  export PIP_CACHE_DIR="${CACHE_DIR}/pip"
  export YOLO_CONFIG_DIR="${CACHE_DIR}/ultralytics" YOLO_AUTOINSTALL=false
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${YOLO_CONFIG_DIR}"
}

load_common_runtime() {
  if [[ "${DRY_RUN}" != 1 ]]; then
    module --force purge >/dev/null 2>&1 || module purge || true
    module load GPU/Python/3.13.5-bare-gcc-2025b-eb
  fi
  runtime_exports
}

load_gpu_runtime() {
  if [[ "${DRY_RUN}" != 1 ]]; then
    module --force purge >/dev/null 2>&1 || module purge || true
    module load GPU/buildenv-nvhpc/25.9-cu13.0
    module load GPU/Python/3.13.5-bare-gcc-2025b-eb
  fi
  runtime_exports
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
}

bootstrap_environment() {
  load_gpu_runtime
  if [[ "${DRY_RUN}" == 1 ]]; then
    echo "[ENV] dry run: using ${PYTHON}; no venv, no installs"
    return 0
  fi
  mkdir -p "${CODE_DIR}" "${CACHE_DIR}/pip"
  local lock="${SPLICE_CACHE}/venv.lock" venv_lock_fd
  exec {venv_lock_fd}>"${lock}"
  flock -x "${venv_lock_fd}"
  if [[ ! -x "${PYTHON}" ]]; then
    echo "[ENV] creating/repairing persistent ARM64 venv once: ${VENV_DIR}"
    if [[ -e "${VENV_DIR}" ]]; then
      mv "${VENV_DIR}" "${VENV_DIR}.incomplete.${SLURM_JOB_ID:-$$}"
    fi
    local building="${VENV_DIR}.building.${SLURM_JOB_ID:-$$}"
    rm -rf "${building}"
    python3 -m venv "${building}"
    "${building}/bin/python" -m pip install --upgrade pip setuptools wheel
    mv "${building}" "${VENV_DIR}"
  fi
  local missing
  missing="$(${PYTHON} - <<'PYCHECK'
import importlib.util
mods = {'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy', 'scipy': 'scipy', 'yaml': 'pyyaml',
        'PIL': 'pillow', 'ultralytics': 'ultralytics>=8.4.0', 'pycocotools': 'pycocotools', 'matplotlib': 'matplotlib',
        'pandas': 'pandas', 'tqdm': 'tqdm'}
print(' '.join(pkg for mod, pkg in mods.items() if importlib.util.find_spec(mod) is None))
PYCHECK
)"
  # torch is pinned as on the other Arrhenius projects (aarch64 CUDA 13.0 wheels come from the
  # PyTorch index); everything installed afterwards is constrained to that build.
  local constraints="${SPLICE_CACHE}/pip_constraints.txt"
  printf 'torch==2.12.1+cu130\n' > "${constraints}"
  if [[ " ${missing} " == *" torch "* || " ${missing} " == *" torchvision "* ]]; then
    if ! "${PYTHON}" -m pip install 'torch==2.12.1+cu130' torchvision --index-url https://download.pytorch.org/whl/cu130; then
      echo "[ENV] paired install failed; installing torch from the PyTorch index and torchvision against it from the default index" >&2
      "${PYTHON}" -m pip install 'torch==2.12.1+cu130' --index-url https://download.pytorch.org/whl/cu130
      "${PYTHON}" -m pip install -c "${constraints}" torchvision
    fi
    missing="${missing//torchvision/}"
    missing="${missing//torch/}"
  fi
  if [[ -n "${missing// /}" ]]; then
    echo "[ENV] installing missing packages once: ${missing}"
    # shellcheck disable=SC2086
    "${PYTHON}" -m pip install -c "${constraints}" ${missing}
  fi
  # OpenCV comes with Ultralytics; headless nodes may lack libGL for the GUI build
  if ! "${PYTHON}" -c 'import cv2' >/dev/null 2>&1; then
    echo "[ENV] cv2 does not import; switching to opencv-python-headless"
    "${PYTHON}" -m pip uninstall -y opencv-python >/dev/null 2>&1 || true
    "${PYTHON}" -m pip install --force-reinstall --no-deps opencv-python-headless
  fi
  "${PYTHON}" -c "from ultralytics import settings; settings.update({'sync': False})" >/dev/null 2>&1 || true
  flock -u "${venv_lock_fd}"
  "${PYTHON}" - <<'PYENV'
import platform, torch, numpy, scipy, ultralytics, torchvision
print('[ENV]', platform.machine(), 'torch', torch.__version__, 'cuda', torch.version.cuda,
      'available', torch.cuda.is_available(), 'torchvision', torchvision.__version__,
      'ultralytics', ultralytics.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)
if platform.machine() != 'aarch64':
    raise SystemExit('Arrhenius Python must be aarch64')
if not torch.__version__.startswith('2.12.1'):
    raise SystemExit(f'torch 2.12.1+cu130 expected, found {torch.__version__} (a later install replaced it)')
if tuple(int(x) for x in ultralytics.__version__.split('.')[:2]) < (8, 4):
    raise SystemExit('Ultralytics >= 8.4.0 is required (YOLO26)')
try:
    import pycocotools  # noqa: F401
except Exception as e:
    raise SystemExit(f'pycocotools does not import: {e}')
print('[ENV] environment bootstrap complete')
PYENV
}

# -----------------------------------------------------------------------------
# code identity
# -----------------------------------------------------------------------------
require_code_file() {
  [[ -f "${PY_SCRIPT_SOURCE}" ]] || {
    echo "[FATAL] Missing SPLICE Python runner: ${PY_SCRIPT_SOURCE}" >&2
    echo "[FATAL] Copy it from the repository: cp ${REPO_DIR}/detector/hpc/${EXPECTED_RUNNER_BASENAME} ${CODE_DIR}/" >&2
    exit 4;
  }
  [[ "$(basename "${PY_SCRIPT_SOURCE}")" == "${EXPECTED_RUNNER_BASENAME}" ]] || {
    echo "[FATAL] Python runner basename mismatch: ${PY_SCRIPT_SOURCE}" >&2; exit 91;
  }
  [[ -f "${SELF}" ]] || {
    echo "[FATAL] Missing canonical SPLICE shell: ${SELF}" >&2
    echo "[FATAL] Copy it from the repository: cp ${REPO_DIR}/detector/hpc/${EXPECTED_SHELL_BASENAME} ${CODE_DIR}/" >&2
    exit 4;
  }
  [[ "$(basename "${SELF}")" == "${EXPECTED_SHELL_BASENAME}" ]] || {
    echo "[FATAL] Canonical shell basename mismatch: ${SELF}" >&2; exit 91;
  }
  [[ -f "${REPO_DIR}/detector/splice_det/__init__.py" && -d "${REPO_DIR}/reference/splicev_ref" && -d "${REPO_DIR}/generated/py" ]] || {
    echo "[FATAL] SPLICE-V repository not found at ${REPO_DIR}" >&2
    echo "[FATAL] cd ${CODE_DIR} && git clone https://github.com/AnonymousHERE-commits/ISCA2027.git SPLICE-V   (or set SPLICE_REPO)" >&2
    exit 4;
  }
  grep -Fqx 'VERSION = "1.0.0-arrhenius"' "${PY_SCRIPT_SOURCE}" || {
    echo "[FATAL] Python VERSION marker mismatch in ${PY_SCRIPT_SOURCE}" >&2; exit 91;
  }
  grep -Fqx 'VERSION_TAG = "SPLICE_V1_0_0"' "${PY_SCRIPT_SOURCE}" || {
    echo "[FATAL] Python VERSION_TAG marker mismatch in ${PY_SCRIPT_SOURCE}" >&2; exit 91;
  }
  grep -Fqx 'EXPECTED_RUNNER_BASENAME = "run_splicev_arrhenius.py"' "${PY_SCRIPT_SOURCE}" || {
    echo "[FATAL] Python expected-basename marker mismatch" >&2; exit 91;
  }
  grep -Fqx 'SHELL_IMPLEMENTATION_VERSION="1.0.0"' "${SELF}" || {
    echo "[FATAL] Shell version marker mismatch in ${SELF}" >&2; exit 91;
  }
  grep -Fqx 'SHELL_VERSION_TAG="SPLICE_V1_0_0"' "${SELF}" || {
    echo "[FATAL] Shell tag marker mismatch in ${SELF}" >&2; exit 91;
  }
}

verify_python_identity() {
  local runner="$1" actual identity
  [[ -f "${runner}" ]] || { echo "[FATAL] Missing runner snapshot: ${runner}" >&2; exit 91; }
  actual="$("${PYTHON}" "${runner}" --print_version | tail -n 1)"
  [[ "${actual}" == "${EXPECTED_VERSION}" ]] || {
    echo "[FATAL] Expected Python version ${EXPECTED_VERSION}, got ${actual}" >&2; exit 91;
  }
  identity="$("${PYTHON}" "${runner}" --print_identity | tail -n 1)"
  "${PYTHON}" - "${identity}" "${EXPECTED_VERSION}" "${EXPECTED_VERSION_TAG}" "${EXPECTED_RUNNER_BASENAME}" <<'PYIDENT'
import json, sys
obj = json.loads(sys.argv[1])
expected = (sys.argv[2], sys.argv[3], sys.argv[4])
actual = (obj.get('version'), obj.get('version_tag'), obj.get('expected_runner_basename'))
if actual != expected or obj.get('actual_runner_basename') != expected[2]:
    raise SystemExit(f'identity mismatch: expected={expected}, actual={obj}')
print('[IDENTITY] verified', json.dumps(obj, sort_keys=True))
PYIDENT
}

require_worker_files() {
  [[ -x "${PYTHON}" ]] || { echo "[FATAL] Missing persistent ARM64 venv: ${PYTHON}" >&2; exit 3; }
  [[ -f "${SPLICE_SUITE}/000_MANIFEST/${EXPECTED_RUNNER_BASENAME}" ]] || {
    echo "[FATAL] Missing suite runner snapshot in ${SPLICE_SUITE}/000_MANIFEST" >&2; exit 4;
  }
  [[ -f "${SPLICE_SUITE}/000_MANIFEST/code/detector/splice_det/__init__.py" ]] || {
    echo "[FATAL] Missing suite code snapshot in ${SPLICE_SUITE}/000_MANIFEST/code" >&2; exit 4;
  }
  [[ -f "${SPLICE_SUITE}/000_MANIFEST/suite_config.json" ]] || {
    echo "[FATAL] Missing suite configuration in ${SPLICE_SUITE}/000_MANIFEST" >&2; exit 4;
  }
}

# -----------------------------------------------------------------------------
# suite allocation and manifests
# -----------------------------------------------------------------------------
allocate_suite() {
  local counter lock next serial suite tag
  counter="${SPLICE_CACHE}/.next_suite_number"
  lock="${SPLICE_CACHE}/.suite_number.lock"
  exec 9>"${lock}"
  flock -x 9
  if [[ -s "${counter}" ]]; then next="$(cat "${counter}")"; else next=1; fi
  [[ "${next}" =~ ^[0-9]+$ ]] || next=1
  while :; do
    serial="$(printf '%04d' "${next}")"
    suite="${OUTPUTS_DIR}/SPLICE_V1_0_0_SUITE_${serial}"
    [[ ! -e "${suite}" ]] && break
    next=$((next+1))
  done
  printf '%d\n' "$((next+1))" > "${counter}"
  flock -u 9
  tag="SPLICE_V1_0_0_S${serial}"
  mkdir -p "${suite}/000_MANIFEST" "${suite}/999_MASTER/RAW_LOGS" "${suite}/ARTIFACTS" "${CKPT_DIR}/${tag}"
  local s
  for s in "${ALL_STAGES[@]}"; do mkdir -p "${suite}/$(stage_dir_of "${s}")/runs" "${suite}/$(stage_dir_of "${s}")/GATE"; done
  printf '%s\n' "${serial}" > "${suite}/000_MANIFEST/suite_number.txt"
  printf '%s\n' "${tag}" > "${suite}/000_MANIFEST/suite_tag.txt"
  printf '%s\n' "${tag}" > "${LIVE_SUITE_FILE}"
  printf '%s\n' "${LIVE_OUT}" > "${suite}/000_MANIFEST/LIVE_LOG_OUT.txt"
  printf '%s\n' "${LIVE_ERR}" > "${suite}/000_MANIFEST/LIVE_LOG_ERR.txt"
  printf '%s\n' "${LOGS_DIR}" > "${suite}/000_MANIFEST/LOG_DIRECTORY.txt"
  printf '%s\n' "${suite}" > "${suite}/000_MANIFEST/OUTPUT_DIRECTORY.txt"
  printf '%s\n' "${CKPT_DIR}/${tag}" > "${suite}/000_MANIFEST/CHECKPOINT_DIRECTORY.txt"
  printf '%s\n' "${suite}" > "${SPLICE_CACHE}/latest_submitted_output.txt"
  ln -sfn "${suite}" "${OUTPUTS_DIR}/SPLICE_V1_0_0_LATEST_OUTPUT"
  printf '%s\n' "${suite}"
}

# The suite runs from its own copy of the code: the runner, this shell and the
# parts of the repository the detector needs.
snapshot_code() {
  local suite="$1" dst="${suite}/000_MANIFEST/code" d
  rm -rf "${dst}"
  mkdir -p "${dst}"
  local parts=(detector/splice_det reference generated/py contracts tools tests)
  [[ -d "${REPO_DIR}/planning" ]] && parts+=(planning)
  (cd "${REPO_DIR}" && tar --exclude='__pycache__' --exclude='*.pyc' -cf - "${parts[@]}") | tar -xf - -C "${dst}"
  for d in "${parts[@]}"; do [[ -e "${dst}/${d}" ]] || { echo "[FATAL] code snapshot incomplete: ${d}" >&2; return 4; }; done
  (cd "${dst}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "${suite}/000_MANIFEST/SHA256SUMS_CODE.txt"
  {
    echo "repository=${REPO_DIR}"
    echo "commit=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty_files=$(git -C "${REPO_DIR}" status --porcelain 2>/dev/null | wc -l || echo unknown)"
    echo "tree_sha256=$(sha256sum "${suite}/000_MANIFEST/SHA256SUMS_CODE.txt" | awk '{print $1}')"
  } > "${suite}/000_MANIFEST/code_identity.txt"
}

snapshot_manifest() {
  local suite="$1" mode="$2"
  cp -f "${PY_SCRIPT_SOURCE}" "${suite}/000_MANIFEST/${EXPECTED_RUNNER_BASENAME}"
  cp -f "${SELF}" "${suite}/000_MANIFEST/${EXPECTED_SHELL_BASENAME}"
  sha256sum "${PY_SCRIPT_SOURCE}" "${SELF}" > "${suite}/000_MANIFEST/SHA256SUMS_SOURCE.txt"
  snapshot_code "${suite}"
  {
    echo "suite=${suite}"
    echo "suite_tag=$(cat "${suite}/000_MANIFEST/suite_tag.txt")"
    echo "release=${SPLICE_RELEASE}"
    echo "log_directory=${LOGS_DIR}"
    echo "checkpoint_directory=${CKPT_DIR}/$(cat "${suite}/000_MANIFEST/suite_tag.txt")"
    echo "mode=${mode}"
    echo "scale=${SCALE}"
    echo "seed=${SEED}"
    echo "parent_job=${SLURM_JOB_ID:-manual}"
    echo "created=$(date --iso-8601=seconds)"
    echo "datasets=${DATASETS}"
    echo "coco_root=${COCO_ROOT}"
    echo "crowdhuman_root=${CROWDHUMAN_ROOT}"
    echo "weights=${WEIGHTS}"
    echo "seeds=${SEEDS}"
    echo "arms=${ARMS}"
    echo "descriptors=${DESCRIPTORS}"
    echo "epochs=${EPOCHS}"
    echo "qat_epochs=${QAT_EPOCHS}"
    echo "batch=${BATCH}"
    echo "fraction=${FRACTION}"
    echo "val_limit=${VAL_LIMIT}"
    echo "native_coco=${NATIVE_COCO}"
    echo "venv=${VENV_DIR}"
    echo "python=${PYTHON}"
    echo "max_concurrent=${MAX_CONCURRENT}"
    echo "worker_cpus=${WORKER_CPUS}"
    echo "stage_retries=${STAGE_RETRIES}"
    echo "max_infra_requeues=${MAX_INFRA_REQUEUES}"
    echo "max_continuations=${MAX_CONTINUATIONS}"
    echo "gpu_health_retries=${GPU_HEALTH_RETRIES}"
    echo "dry_run=${DRY_RUN}"
    local s
    for s in "${ALL_STAGES[@]}"; do echo "resources_${s}=$(stage_time "${s}"),$(stage_mem "${s}"),$(stage_cpus "${s}")cpus"; done
    echo "expected_version=${EXPECTED_VERSION}"
    echo "expected_version_tag=${EXPECTED_VERSION_TAG}"
    echo "expected_runner_basename=${EXPECTED_RUNNER_BASENAME}"
    echo "expected_shell_basename=${EXPECTED_SHELL_BASENAME}"
    echo "runtime_shell=${RUNTIME_SHELL}"
    echo "canonical_shell=${SELF}"
    echo "canonical_shell_sha256=$(sha256sum "${SELF}" | awk '{print $1}')"
    cat "${suite}/000_MANIFEST/code_identity.txt"
  } > "${suite}/000_MANIFEST/suite.env"
}

append_registry() {
  local status="$1" run_dir="$2" registry="${SPLICE_SUITE}/000_MANIFEST/run_registry.tsv"
  {
    flock -x 9
    if [[ ! -s "${registry}" ]]; then
      printf 'timestamp\tjob_id\tarray_task\tstage\tordinal\ttask\tstatus\trun_dir\n' >&9
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-manual}" "${SLURM_ARRAY_TASK_ID:-NA}" \
      "${SPLICE_STAGE:-system}" "${ORDINAL:-0000}" "${TASK:-system}" "${status}" "${run_dir}" >&9
  } 9>>"${registry}"
}

record_stage_job() {
  local stage="$1" job="$2" map="${SPLICE_SUITE}/000_MANIFEST/job_ids.tsv"
  {
    flock -x 5
    [[ -s "${map}" ]] || printf 'stage\tjob_id\n' >&5
    printf '%s\t%s\n' "${stage}" "${job}" >&5
  } 5>>"${map}"
}

manifest_rows() { local f="$1"; [[ -f "${f}" ]] || { echo 0; return; }; echo $(( $(wc -l < "${f}") - 1 )); }

# -----------------------------------------------------------------------------
# submission helpers
# -----------------------------------------------------------------------------
submit_prepare() {
  local suite="$1" tag="$2" jid
  jid="$(sbatch --parsable --job-name=splice_V1_0_0_prep --time=04:00:00 --cpus-per-task=16 --mem=64G --gpus="${ADMIN_GPUS}" \
    --output=/dev/null --error=/dev/null \
    --export="ALL,SPLICE_CANONICAL_SHELL=${SELF},SPLICE_ROLE=prepare,SPLICE_SUITE=${suite},SPLICE_SUITE_TAG=${tag},SPLICE_SCALE=${SCALE},SPLICE_SEED=${SEED}" "${SELF}")"
  strip_jobid "${jid}"
}

# submit_array_stage SUITE STAGE MANIFEST DEPENDENCY REQUIRED_PASS [DEP_KIND]
submit_array_stage() {
  local suite="$1" stage="$2" manifest="$3" dependency="$4" required_pass="${5:-}" dep_kind="${6:-afterok}" n jid
  n="$(manifest_rows "${manifest}")"
  (( n > 0 )) || { echo "[FATAL] empty task manifest: ${manifest}" >&2; return 2; }
  local dep=() concurrency="${MAX_CONCURRENT}"
  [[ -n "${dependency}" ]] && dep=(--dependency="${dep_kind}:${dependency}")
  jid="$(sbatch --parsable "${dep[@]}" --array="0-$((n-1))%${concurrency}" --job-name="splice_V1_0_0_${stage}" \
    --time="$(stage_time "${stage}")" --cpus-per-task="$(stage_cpus "${stage}")" --mem="$(stage_mem "${stage}")" --gpus=1 --requeue \
    --output=/dev/null --error=/dev/null \
    --export="ALL,SPLICE_CANONICAL_SHELL=${SELF},SPLICE_ROLE=worker,SPLICE_SUITE=${suite},SPLICE_STAGE=${stage},SPLICE_MANIFEST=${manifest},SPLICE_REQUIRED_PASS=${required_pass},SPLICE_SCALE=${SCALE}" \
    "${SELF}")"
  strip_jobid "${jid}"
}

submit_gate() {
  local suite="$1" stage="$2" dependency="$3" jid dep=()
  [[ -n "${dependency}" ]] && dep=(--dependency="afterany:${dependency}")
  jid="$(sbatch --parsable "${dep[@]}" --job-name="splice_V1_0_0_gate_${stage}" \
    --time=01:00:00 --cpus-per-task=4 --mem=24G --gpus="${ADMIN_GPUS}" --output=/dev/null --error=/dev/null \
    --export="ALL,SPLICE_CANONICAL_SHELL=${SELF},SPLICE_ROLE=gate,SPLICE_SUITE=${suite},SPLICE_STAGE=${stage},SPLICE_SCALE=${SCALE}" "${SELF}")"
  strip_jobid "${jid}"
}

submit_finalizer() {
  local suite="$1" dependency="$2" jid dep=()
  [[ -n "${dependency}" ]] && dep=(--dependency="afterany:${dependency}")
  jid="$(sbatch --parsable "${dep[@]}" --job-name=splice_V1_0_0_pack \
    --time=04:00:00 --cpus-per-task=8 --mem=64G --gpus="${ADMIN_GPUS}" --output=/dev/null --error=/dev/null \
    --export="ALL,SPLICE_CANONICAL_SHELL=${SELF},SPLICE_ROLE=finalizer,SPLICE_SUITE=${suite}" "${SELF}")"
  strip_jobid "${jid}"
}

# Launch one stage: its task array (if it has tasks) and its gate.  Prints the
# gate's job id, or SKIPPED when the stage has no registered task.
launch_stage() {
  local suite="$1" stage="$2" dependency="$3" required_pass="$4" manifest="${5:-}" dep_kind="${6:-afterok}"
  local sdir aj gj n
  sdir="$(stage_dir_of "${stage}")"
  [[ -n "${manifest}" ]] || manifest="${suite}/000_MANIFEST/tasks_${stage}.tsv"
  n="$(manifest_rows "${manifest}")"
  if (( n == 0 )); then
    mkdir -p "${suite}/${sdir}/GATE"
    echo "SKIPPED" > "${suite}/${sdir}/GATE/SKIPPED"
    printf '{"stage":"%s","decision":"SKIPPED","reason":"no tasks registered for this stage"}\n' "${stage}" \
      > "${suite}/${sdir}/GATE/gate.json"
    echo "[STAGE] ${stage}: no tasks registered; marked SKIPPED" >&2
    printf '%s\n' "SKIPPED"
    return 0
  fi
  aj="$(submit_array_stage "${suite}" "${stage}" "${manifest}" "${dependency}" "${required_pass}" "${dep_kind}")"
  gj="$(submit_gate "${suite}" "${stage}" "${aj}")"
  SPLICE_SUITE="${suite}" record_stage_job "${stage}" "${aj}"
  SPLICE_SUITE="${suite}" record_stage_job "${stage}_gate" "${gj}"
  echo "[STAGE] ${stage}: tasks=${n} array=${aj} gate=${gj}" >&2
  printf '%s\n' "${gj}"
}

# Launch chain stage STAGE; a stage without tasks is skipped and the chain moves
# on (to the parallel evaluation stages after the last chain stage).
advance_chain() {
  local suite="$1" stage="$2" dependency="$3" required_pass="$4" gj
  while [[ -n "${stage}" ]]; do
    gj="$(launch_stage "${suite}" "${stage}" "${dependency}" "${required_pass}")"
    if [[ "${gj}" != SKIPPED ]]; then
      echo "[CHAIN] stage ${stage} launched; gate=${gj}"
      return 0
    fi
    stage="$(next_chain_stage "${stage}")"
  done
  echo "[CHAIN] chain complete; launching the parallel evaluation stages"
  launch_parallel_stages "${suite}" "${dependency}" "${required_pass}"
}

# The PASS file a worker of STAGE requires: the gate of the nearest earlier chain
# stage that has tasks (PREPARED for the first one).
required_pass_of() {
  local suite="$1" stage="$2" prev
  if is_parallel_stage "${stage}"; then prev="${CHAIN_STAGES[${#CHAIN_STAGES[@]}-1]}"; else prev="$(prev_chain_stage "${stage}")"; fi
  if [[ "${stage}" == reduce ]]; then printf '\n'; return 0; fi
  while [[ -n "${prev}" ]]; do
    if [[ ! -f "${suite}/$(stage_dir_of "${prev}")/GATE/SKIPPED" ]]; then
      printf '%s\n' "${suite}/$(stage_dir_of "${prev}")/GATE/PASS"; return 0
    fi
    prev="$(prev_chain_stage "${prev}")"
  done
  printf '%s\n' "${suite}/000_MANIFEST/PREPARED"
}

# -----------------------------------------------------------------------------
# site facts: allocation, partition, storage (recorded with every suite)
# -----------------------------------------------------------------------------
header_value() { awk -v flag="$1" '$1=="#SBATCH" && $2==flag {print $3; exit}' "${SELF}" 2>/dev/null || true; }
suite_account() { printf '%s\n' "${SBATCH_ACCOUNT:-$(header_value -A)}"; }
suite_partition() { printf '%s\n' "${SBATCH_PARTITION:-$(header_value -p)}"; }

record_site_facts() {
  local suite="$1" out part acct
  out="${suite}/000_MANIFEST/SITE.txt"; part="$(suite_partition)"; acct="$(suite_account)"
  {
    echo "account=${acct}"
    echo "partition=${part}"
    echo "storage_root=${BASE}"
    echo "user=${USER:-$(id -un)}  host=$(hostname -s)  utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "--- sinfo: partition, time limit, nodes, cpus per node, memory per node (MB), gres"
    sinfo -h -p "${part}" -o '%P %l %D %c %m %G' 2>&1 | head -n 5 || true
    echo "--- sacctmgr: associations of this user"
    timeout 60 sacctmgr -nP show assoc where user="${USER:-$(id -un)}" format=account,partition,qos 2>&1 | head -n 40 || true
    echo "--- projinfo"
    timeout 60 projinfo 2>&1 | head -n 60 || true
    echo "--- storagequota"
    timeout 60 storagequota 2>&1 | head -n 40 || true
    echo "--- df"
    df -h "${BASE}" 2>&1 || true
  } > "${out}" 2>&1 || true
  echo "[SITE] account=${acct} partition=${part} time_limit=$(sinfo -h -p "${part}" -o %l 2>/dev/null | head -n 1 || true) (details: ${out})"
}

# -----------------------------------------------------------------------------
# orchestrator
# -----------------------------------------------------------------------------
write_where_files() {
  local suite="$1" tag="$2" s
  {
    echo "LOG_FOLDER=${LOGS_DIR}"
    echo "OUTPUT_FOLDER=${suite}"
    echo "CHECKPOINT_FOLDER=${CKPT_DIR}/${tag}"
    echo "PARENT_OUT=${LOGS_DIR}/${tag}_0000_PARENT.out"
    echo "PREP_OUT=${LOGS_DIR}/${tag}_0000_PREP.out"
    for s in "${ALL_STAGES[@]}"; do
      echo "$(printf '%s' "${s}" | tr '[:lower:]' '[:upper:]')_GATE_OUT=${LOGS_DIR}/${tag}_$(stage_number_of "${s}")_9000_${s}_GATE.out"
    done
    echo "FINALIZER_OUT=${LOGS_DIR}/${tag}_9999_FINALIZER.out"
    echo "STABLE_LIVE_OUT=${LIVE_OUT}"
    echo "STABLE_LIVE_ERR=${LIVE_ERR}"
    echo "MEASURED_REPORT=${suite}/900_REDUCE/MEASURED_REPORT.md"
    echo "TABLE7_MEASURED=${suite}/900_REDUCE/table7_measured.csv"
    echo "DECLARED_TESTS=${suite}/900_REDUCE/tests.json"
    echo "PAPER_TARGET_COMPARISON=${suite}/900_REDUCE/paper_target_comparison.csv"
    echo "FINAL_REPORT=${suite}/ALL_RESULTS.txt"
    echo "FINAL_RESULTS_ZIP=${suite}/SPLICE_V1_0_0_RESULTS.zip"
  } > "${suite}/000_MANIFEST/WHERE_ARE_MY_FILES.txt"
}

orchestrate() {
  local mode="${1:-full}" suite tag parent_out parent_err prep
  case "${mode}" in
    full) ;;
    pilot) SCALE=pilot ;;
    smoke) SCALE=preflight ;;
    *) echo "[FATAL] mode must be full, pilot, smoke, resume or postprocess; got ${mode}" >&2; usage; exit 10 ;;
  esac
  case "${SCALE}" in full|pilot|preflight) ;; *) echo "[FATAL] SPLICE_SCALE must be full, pilot or preflight, got ${SCALE}" >&2; exit 2 ;; esac
  init_live_monitor "${mode}"
  attach_live_only "ORCHESTRATOR_BOOTSTRAP"
  echo "[MONITOR_FILES] ${LIVE_OUT} ${LIVE_ERR}"
  require_code_file
  command -v sbatch >/dev/null 2>&1 || { echo "[FATAL] sbatch unavailable" >&2; exit 10; }
  echo "[IDENTITY] shell=${EXPECTED_SHELL_BASENAME} runner=${EXPECTED_RUNNER_BASENAME} version=${EXPECTED_VERSION} release=${SPLICE_RELEASE}"
  echo "[ADDRESSES] base=${BASE} repository=${REPO_DIR} datasets=${DATA_DIR} checkpoints=${CKPT_DIR}"
  suite="$(allocate_suite)"
  tag="$(cat "${suite}/000_MANIFEST/suite_tag.txt")"
  export SPLICE_SUITE="${suite}"
  parent_out="${LOGS_DIR}/${tag}_0000_PARENT.out"
  parent_err="${LOGS_DIR}/${tag}_0000_PARENT.err"
  attach_stage_logs "${parent_out}" "${parent_err}" "ORCHESTRATOR_${tag}"
  echo "[SUITE_CREATED] ${suite}"
  echo "[SUITE_TAG] ${tag}"
  echo "[SCALE] ${SCALE}"
  record_site_facts "${suite}"
  snapshot_manifest "${suite}" "${mode}"
  write_where_files "${suite}" "${tag}"
  : > "${LOGS_DIR}/${tag}_9999_FINALIZER.out"
  : > "${LOGS_DIR}/${tag}_9999_FINALIZER.err"

  prep="$(submit_prepare "${suite}" "${tag}")"
  record_stage_job prepare "${prep}"
  # The first stage is launched by the prepare job itself (it needs the task
  # manifests that --init_suite writes); a failed prepare still gets packaged.
  local fj; fj="$(submit_finalizer "${suite}" "${prep}")"
  printf '%s\n' "${fj}" > "${suite}/000_MANIFEST/PREP_FAILURE_FINALIZER_JOB.txt"
  record_stage_job finalizer_if_prepare_fails "${fj}"
  echo "[STAGING] prepare (venv, identity, semantic preflight, dataset views, weights, API probe, task manifests) -> baseline -> train -> calibrate -> qat -> {evaluate, diagnostics, export} -> reduce -> finalizer"
  echo "[MONITOR] cd ${LOGS_DIR} && touch SPLICE_V1_0_0_LIVE.out SPLICE_V1_0_0_LIVE.err && tail -n 100 -F SPLICE_V1_0_0_LIVE.out SPLICE_V1_0_0_LIVE.err"
  echo "[OUTPUT_FOLDER] ${suite}"
  echo "[CHECKPOINT_FOLDER] ${CKPT_DIR}/${tag}"
  echo "[FINAL_ZIP] ${suite}/SPLICE_V1_0_0_RESULTS.zip"
}

# -----------------------------------------------------------------------------
# prepare role
# -----------------------------------------------------------------------------
opt_arg() { [[ -n "$2" ]] && printf '%s\n%s\n' "$1" "$2"; return 0; }

run_prepare() {
  local suite="${SPLICE_SUITE}" tag out err py_snapshot code_snapshot
  tag="${SPLICE_SUITE_TAG:-$(cat "${suite}/000_MANIFEST/suite_tag.txt")}"
  out="${LOGS_DIR}/${tag}_0000_PREP.out"; err="${LOGS_DIR}/${tag}_0000_PREP.err"
  attach_stage_logs "${out}" "${err}" "PREP_${tag}"
  echo "[PREP] suite=${suite} scale=${SCALE} seed=${SEED} datasets=${DATASETS}"
  bootstrap_environment
  py_snapshot="${suite}/000_MANIFEST/${EXPECTED_RUNNER_BASENAME}"
  code_snapshot="${suite}/000_MANIFEST/code"
  [[ -f "${py_snapshot}" ]] || { echo "[FATAL] missing runner snapshot ${py_snapshot}" >&2; exit 4; }
  verify_python_identity "${py_snapshot}"
  # Unit tests of the code snapshot and the autograd self-checks (CPU).  Pilot and full suites
  # fail closed; the smoke scale records a failure and goes on, so that one pass over every
  # stage shows every problem of a new installation.
  local advisory=()
  [[ "${SCALE}" == preflight ]] && advisory=(--advisory)
  "${PYTHON}" -u "${py_snapshot}" --semantic_preflight_only --cpu --code_root "${code_snapshot}" \
    --out_dir "${suite}/000_MANIFEST/SEMANTIC_PREFLIGHT" "${advisory[@]}"
  local extra=() line
  while IFS= read -r line; do extra+=("${line}"); done < <(
    opt_arg --seeds "${SEEDS// /,}"; opt_arg --arms "${ARMS// /,}"; opt_arg --descriptors "${DESCRIPTORS}"
    opt_arg --epochs "${EPOCHS}"; opt_arg --qat_epochs "${QAT_EPOCHS}"; opt_arg --fraction "${FRACTION}"
    opt_arg --val_limit "${VAL_LIMIT}"; opt_arg --native_coco "${NATIVE_COCO}")
  "${PYTHON}" -u "${py_snapshot}" --init_suite --suite "${suite}" --scale "${SCALE}" --seed "${SEED}" \
    --code_root "${code_snapshot}" --datasets "${DATASETS// /,}" --batch "${BATCH}" \
    --ckpt_root "${CKPT_DIR}/${tag}" --weights "${WEIGHTS}" "${extra[@]}"
  # Dataset views (train_fit / calib / val), pretrained weights, one training and one
  # inference step of the model on a random batch (interface check of the installed stack).
  "${PYTHON}" -u "${py_snapshot}" --prepare_dataset_only --suite "${suite}" --code_root "${code_snapshot}" \
    --datasets "${DATASETS// /,}" --data_root "${DATA_DIR}" --coco_root "${COCO_ROOT}" --crowdhuman_root "${CROWDHUMAN_ROOT}" \
    --weights "${WEIGHTS}" --out_dir "${suite}/000_MANIFEST/DATASET_PREP" "${advisory[@]}"
  sha256sum "${suite}/000_MANIFEST/datasets.json" > "${suite}/000_MANIFEST/SHA256SUMS_DATASET.txt"
  local s
  for s in "${ALL_STAGES[@]}"; do
    [[ -f "${suite}/000_MANIFEST/tasks_${s}.tsv" ]] || { echo "[FATAL] --init_suite did not write tasks_${s}.tsv" >&2; exit 74; }
    echo "[PREP] tasks_${s}.tsv: $(manifest_rows "${suite}/000_MANIFEST/tasks_${s}.tsv") tasks"
  done
  touch "${suite}/000_MANIFEST/PREPARED"
  # Hand the chain to the first stage.  The finalizer submitted by the
  # orchestrator (after this job) is cancelled because the chain now owns
  # packaging.
  local fj
  fj="$(cat "${suite}/000_MANIFEST/PREP_FAILURE_FINALIZER_JOB.txt" 2>/dev/null || true)"
  [[ -n "${fj}" ]] && { scancel "${fj}" >/dev/null 2>&1 || true; }
  advance_chain "${suite}" baseline "${SLURM_JOB_ID:-}" "${suite}/000_MANIFEST/PREPARED"
  echo "[PREP] completed"
}

# -----------------------------------------------------------------------------
# worker role
# -----------------------------------------------------------------------------
read_task() {
  local task="${SLURM_ARRAY_TASK_ID:-0}" line stage_col
  line="$(awk -F '\t' -v n="$((task+2))" 'NR==n{print; exit}' "${SPLICE_MANIFEST}")"
  [[ -n "${line}" ]] || { echo "[FATAL] no manifest row for task ${task} in ${SPLICE_MANIFEST}" >&2; exit 30; }
  IFS=$'\t' read -r ORDINAL TASK stage_col <<< "${line}"
  [[ -z "${stage_col}" || "${stage_col}" == "${SPLICE_STAGE}" ]] || {
    echo "[FATAL] manifest row stage '${stage_col}' does not match job stage '${SPLICE_STAGE}'" >&2; exit 30;
  }
  export ORDINAL TASK
}

run_worker() {
  read_task
  local tag sdir snum log_out log_err run_parent run_dir py_snapshot shell_snapshot code_snapshot rc health_rc
  tag="$(cat "${SPLICE_SUITE}/000_MANIFEST/suite_tag.txt")"
  sdir="$(stage_dir_of "${SPLICE_STAGE}")"; snum="$(stage_number_of "${SPLICE_STAGE}")"
  log_out="${LOGS_DIR}/${tag}_${snum}_${ORDINAL}_${TASK}.out"
  log_err="${LOGS_DIR}/${tag}_${snum}_${ORDINAL}_${TASK}.err"
  attach_stage_logs "${log_out}" "${log_err}" "WORKER_${sdir}_${ORDINAL}_${TASK}"
  run_parent="${SPLICE_SUITE}/${sdir}/runs/${ORDINAL}_${TASK}"
  run_dir="${run_parent}/RESULT"
  CURRENT_RUN_PARENT="${run_parent}"
  CURRENT_RUN_DIR="${run_dir}"
  CURRENT_LOG_ERR="${log_err}"
  local archive_rc=0
  archive_previous_attempt "${run_parent}" || archive_rc=$?
  if [[ ${archive_rc} -eq 2 ]]; then
    echo "[RESUME] valid completed result already exists: ${run_dir}"
    append_registry ALREADY_COMPLETED "${run_dir}"
    exit 0
  fi
  mkdir -p "${run_dir}" "${run_dir}/mplconfig"
  echo "[LOG_OUT] ${log_out}"
  echo "[LOG_ERR] ${log_err}"
  echo "[OUTPUT] ${run_dir}"

  if [[ -n "${SPLICE_REQUIRED_PASS:-}" && ! -f "${SPLICE_REQUIRED_PASS}" ]]; then
    echo "Skipped because prerequisite gate did not pass: ${SPLICE_REQUIRED_PASS}" > "${run_parent}/SKIPPED.txt"
    append_registry SKIPPED "${run_dir}"
    exit 0
  fi
  rm -f "${run_parent}/SKIPPED.txt" "${run_parent}/CONTINUE_PENDING.json" "${run_parent}/INFRA_RETRY_PENDING.json"
  [[ -f "${SPLICE_SUITE}/000_MANIFEST/PREPARED" ]] || { echo "[FATAL] suite preparation did not complete" >&2; exit 31; }
  require_worker_files
  load_gpu_runtime
  py_snapshot="${SPLICE_SUITE}/000_MANIFEST/${EXPECTED_RUNNER_BASENAME}"
  shell_snapshot="${SPLICE_SUITE}/000_MANIFEST/${EXPECTED_SHELL_BASENAME}"
  code_snapshot="${SPLICE_SUITE}/000_MANIFEST/code"
  verify_python_identity "${py_snapshot}"
  local gpu_args=()
  if is_gpu_stage "${SPLICE_STAGE}"; then
    if gpu_health_preflight "${py_snapshot}" "${run_dir}"; then
      health_rc=0
    else
      health_rc=$?
    fi
    if [[ ${health_rc} -ne 0 ]]; then
      append_registry INFRASTRUCTURE_HEALTH_FAILURE "${run_dir}"
      requeue_current_task "${run_parent}" "GPU health preflight failed with rc=${health_rc}" || exit "${health_rc}"
    fi
    [[ "${DRY_RUN}" == 1 ]] || gpu_args=(--require_cuda)
  fi
  export MPLCONFIGDIR="${run_dir}/mplconfig"
  export SPLICE_SHELL_SNAPSHOT="${shell_snapshot}"

  module list > "${run_dir}/modules.txt" 2>&1 || true
  "${PYTHON}" -m pip freeze > "${run_dir}/pip_freeze.txt" 2>/dev/null || true
  nvidia-smi > "${run_dir}/nvidia_smi.txt" 2>&1 || true
  uname -a > "${run_dir}/uname.txt" || true
  local deadline; deadline="$(job_deadline_epoch)"
  {
    echo "job_id=${SLURM_JOB_ID:-manual}"; echo "array_job_id=${SLURM_ARRAY_JOB_ID:-}"; echo "array_task=${SLURM_ARRAY_TASK_ID:-NA}"
    echo "restart_count=${SLURM_RESTART_COUNT:-0}"; echo "deadline_epoch=${deadline}"
    echo "stage=${SPLICE_STAGE}"; echo "stage_dir=${sdir}"; echo "ordinal=${ORDINAL}"; echo "task=${TASK}"
    echo "manifest=${SPLICE_MANIFEST}"; echo "node=${SLURMD_NODENAME:-$(hostname -s)}"
    echo "code_sha256=$(sha256sum "${py_snapshot}" | awk '{print $1}')"
    echo "shell_sha256=$(sha256sum "${shell_snapshot}" | awk '{print $1}')"
    echo "code_tree_sha256=$(awk -F= '$1=="tree_sha256"{print $2}' "${SPLICE_SUITE}/000_MANIFEST/code_identity.txt" 2>/dev/null || echo NA)"
    echo "datasets_sha256=$(awk '{print $1}' "${SPLICE_SUITE}/000_MANIFEST/SHA256SUMS_DATASET.txt" 2>/dev/null || echo NA)"
  } > "${run_dir}/JOB_METADATA.txt"
  append_registry RUNNING "${run_dir}"
  CURRENT_REGISTRY_ACTIVE=1

  local cmd=(
    "${PYTHON}" -u "${py_snapshot}"
    --stage "${SPLICE_STAGE}" --task "${TASK}" --ordinal "${ORDINAL}" --suite "${SPLICE_SUITE}" --run_dir "${run_dir}"
    --code_root "${code_snapshot}" --threads "${SLURM_CPUS_PER_TASK:-$(stage_cpus "${SPLICE_STAGE}")}"
    --deadline_epoch "${deadline}" "${gpu_args[@]}"
  )
  echo "================================================================================"
  echo "[TASK] stage=${SPLICE_STAGE} ordinal=${ORDINAL} task=${TASK} restart=${SLURM_RESTART_COUNT:-0}"
  echo "[RUN_DIR] ${run_dir}"
  printf '[COMMAND]'; printf ' %q' "${cmd[@]}"; printf '\n'
  echo "================================================================================"

  # A step inherits the job's CPUs per task only through SRUN_CPUS_PER_TASK (Slurm 22.05 and
  # later); without it the runner would be confined to one core.
  [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
  if srun "${cmd[@]}"; then
    rc=0
  else
    rc=$?
  fi
  if [[ ${rc} -eq ${CONTINUE_EXIT_CODE} ]]; then
    append_registry CONTINUE_REQUESTED "${run_dir}"
    CURRENT_REGISTRY_ACTIVE=0
    continue_current_task "${run_parent}" || {
      write_shell_failure_artifact "${CONTINUE_EXIT_CODE}" "CONTINUATION_BUDGET" \
        "continuation budget exhausted (SPLICE_MAX_CONTINUATIONS=${MAX_CONTINUATIONS})" ORCHESTRATION false || true
      SHELL_ERROR_HANDLER_ACTIVE=1
      exit "${CONTINUE_EXIT_CODE}"
    }
  fi
  if [[ ${rc} -ne 0 ]] && infrastructure_failure_detected "${run_dir}" "${log_err}" "${rc}"; then
    append_registry INFRASTRUCTURE_FAILURE "${run_dir}"
    requeue_current_task "${run_parent}" "worker command failed with retryable infrastructure error rc=${rc}" || exit "${rc}"
  fi
  if [[ ${rc} -eq 0 && ! -f "${run_dir}/summary.json" ]]; then
    echo "[FATAL] worker returned rc=0 but summary.json is missing" >&2
    write_shell_failure_artifact "${ARTIFACT_MISSING_EXIT_CODE}" "POST_RUN_ARTIFACT_CHECK" \
      "required summary.json missing after successful worker return" ARTIFACT_MISSING false || true
    rc="${ARTIFACT_MISSING_EXIT_CODE}"
  fi
  if [[ ${rc} -eq 0 ]]; then
    append_registry COMPLETED "${run_dir}"
    CURRENT_REGISTRY_ACTIVE=0
  else
    local failure_status="FAILED"
    if [[ -f "${run_dir}/failure_summary.json" ]]; then
      failure_status="FAILED_$("${PYTHON}" - "${run_dir}/failure_summary.json" <<'PYCLASS'
import json, sys
try:
    print((json.load(open(sys.argv[1])).get('failure_class') or 'SCIENTIFIC').upper())
except Exception:
    print('UNPARSEABLE')
PYCLASS
)"
    fi
    append_registry "${failure_status}" "${run_dir}"
    CURRENT_REGISTRY_ACTIVE=0
    # the Python runner already wrote its own failure_summary.json: no second shell record
    [[ -f "${run_dir}/failure_summary.json" ]] && SHELL_ERROR_HANDLER_ACTIVE=1
  fi
  exit "${rc}"
}

# -----------------------------------------------------------------------------
# gate role: PASS -> next stage(s); RETRY -> resubmit missing/retryable tasks
# (bounded by SPLICE_STAGE_RETRIES); FAIL -> downstream not run, suite packaged
# -----------------------------------------------------------------------------
submit_reduce_once() {
  local suite="$1" dependency="$2" lock="${suite}/000_MANIFEST/.reduce_submit.lock" marker="${suite}/000_MANIFEST/REDUCE_SUBMITTED"
  local reduce_lock_fd
  exec {reduce_lock_fd}>"${lock}"; flock -x "${reduce_lock_fd}"
  if [[ -f "${marker}" ]]; then
    flock -u "${reduce_lock_fd}"
    echo "[JOIN] reduce already submitted ($(cat "${marker}"))"
    return 0
  fi
  local gj
  gj="$(launch_stage "${suite}" reduce "${dependency}" "" "" afterany)"
  printf '%s\n' "${gj}" > "${marker}"
  flock -u "${reduce_lock_fd}"
  echo "[JOIN] reduce submitted; reduce gate=${gj}"
}

parallel_terminal() {
  local suite="$1" s g
  for s in "${PARALLEL_STAGES[@]}"; do
    g="${suite}/$(stage_dir_of "${s}")/GATE"
    [[ -f "${g}/PASS" || -f "${g}/FAIL" || -f "${g}/SKIPPED" ]] || return 1
  done
  return 0
}

try_join() {
  local suite="$1" lock="${suite}/000_MANIFEST/.join.lock"
  local join_lock_fd
  exec {join_lock_fd}>"${lock}"; flock -x "${join_lock_fd}"
  if parallel_terminal "${suite}"; then
    flock -u "${join_lock_fd}"
    submit_reduce_once "${suite}" "${SLURM_JOB_ID:-}"
  else
    flock -u "${join_lock_fd}"
    echo "[JOIN] waiting for the remaining evaluation stages"
  fi
}

launch_parallel_stages() {
  local suite="$1" dependency="$2" required_pass="$3" s gj any=0
  for s in "${PARALLEL_STAGES[@]}"; do
    gj="$(launch_stage "${suite}" "${s}" "${dependency}" "${required_pass}")"
    [[ "${gj}" == SKIPPED ]] || any=1
  done
  (( any == 1 )) || submit_reduce_once "${suite}" "${dependency}"
}

run_gate() {
  local stage="${SPLICE_STAGE}" suite="${SPLICE_SUITE}" sdir gdir tag snum log_out log_err py_snapshot rc retry=0 retry_file n
  sdir="$(stage_dir_of "${stage}")"; snum="$(stage_number_of "${stage}")"
  gdir="${suite}/${sdir}/GATE"; mkdir -p "${gdir}"
  tag="$(cat "${suite}/000_MANIFEST/suite_tag.txt")"
  log_out="${LOGS_DIR}/${tag}_${snum}_9000_${stage}_GATE.out"; log_err="${LOGS_DIR}/${tag}_${snum}_9000_${stage}_GATE.err"
  attach_stage_logs "${log_out}" "${log_err}" "GATE_${stage}_${snum}"
  load_common_runtime
  py_snapshot="${suite}/000_MANIFEST/${EXPECTED_RUNNER_BASENAME}"
  verify_python_identity "${py_snapshot}"
  if "${PYTHON}" -u "${py_snapshot}" --stage_gate "${stage}" --suite "${suite}" --stage_retries "${STAGE_RETRIES}"; then rc=0; else rc=$?; fi
  retry_file="${gdir}/RETRY_COUNT.txt"
  [[ -s "${retry_file}" ]] && retry="$(cat "${retry_file}")"; [[ "${retry}" =~ ^[0-9]+$ ]] || retry=0

  if [[ ${rc} -eq ${GATE_RETRY} ]]; then
    n="$(manifest_rows "${gdir}/missing_tasks.tsv")"
    if (( retry >= STAGE_RETRIES )); then
      echo FAIL > "${gdir}/FAIL"; rm -f "${gdir}/PASS" "${gdir}/RETRY"
      printf '{"stage":"%s","decision":"FAIL","reason":"RETRY_BUDGET_EXHAUSTED","retry_count":%s,"still_missing":%s}\n' \
        "${stage}" "${retry}" "${n}" > "${gdir}/gate_retry_exhausted.json"
      echo "[GATE ${stage}] retry budget exhausted (${retry}/${STAGE_RETRIES}); ${n} task(s) still missing/retryable" >&2
      rc=${GATE_FAIL}
    else
      retry=$((retry+1)); printf '%s\n' "${retry}" > "${retry_file}"
      local retry_manifest="${gdir}/retry_${retry}_tasks.tsv"
      cp -f "${gdir}/missing_tasks.tsv" "${retry_manifest}"
      local required_prev gj
      required_prev="$(required_pass_of "${suite}" "${stage}")"
      gj="$(launch_stage "${suite}" "${stage}" "${SLURM_JOB_ID:-}" "${required_prev}" "${retry_manifest}" afterok)"
      echo "[GATE ${stage}] retry ${retry}/${STAGE_RETRIES} submitted for ${n} task(s); next gate=${gj}"
      return 0
    fi
  fi

  if [[ ${rc} -eq ${GATE_CONTINUE} ]]; then
    # Only trainings that stopped before their time limit are open: resubmit them.  Each
    # task's own continuation budget (SPLICE_MAX_CONTINUATIONS) bounds this loop.
    local rounds_file="${gdir}/CONTINUE_ROUNDS.txt" rounds=0 cont_manifest required_prev gj
    [[ -s "${rounds_file}" ]] && rounds="$(cat "${rounds_file}")"; [[ "${rounds}" =~ ^[0-9]+$ ]] || rounds=0
    rounds=$((rounds+1)); printf '%s\n' "${rounds}" > "${rounds_file}"
    cont_manifest="${gdir}/continue_${rounds}_tasks.tsv"
    cp -f "${gdir}/missing_tasks.tsv" "${cont_manifest}"
    required_prev="$(required_pass_of "${suite}" "${stage}")"
    gj="$(launch_stage "${suite}" "${stage}" "${SLURM_JOB_ID:-}" "${required_prev}" "${cont_manifest}" afterok)"
    echo "[GATE ${stage}] continuation round ${rounds}: $(manifest_rows "${cont_manifest}") task(s) resubmitted to continue from their checkpoints; next gate=${gj}"
    return 0
  fi

  if [[ ${rc} -eq ${GATE_PASS} ]]; then
    echo "[GATE ${stage}] PASS"
    if [[ "${stage}" == reduce ]]; then
      local fj; fj="$(submit_finalizer "${suite}" "${SLURM_JOB_ID:-}")"; record_stage_job finalizer "${fj}"
      echo "[GATE reduce] finalizer=${fj}"
      return 0
    fi
    if is_chain_stage "${stage}"; then
      advance_chain "${suite}" "$(next_chain_stage "${stage}")" "${SLURM_JOB_ID:-}" "${gdir}/PASS"
      return 0
    fi
    if is_parallel_stage "${stage}"; then
      try_join "${suite}"
      return 0
    fi
    return 0
  fi

  # FAIL (fatal task failures, or retry budget exhausted)
  echo FAIL > "${gdir}/FAIL"; rm -f "${gdir}/PASS"
  echo "[GATE ${stage}] FAIL (rc=${rc}); see ${gdir}/gate.txt" >&2
  if [[ "${stage}" == reduce ]]; then
    local fj; fj="$(submit_finalizer "${suite}" "${SLURM_JOB_ID:-}")"; record_stage_job finalizer "${fj}"
    return 0
  fi
  if is_parallel_stage "${stage}"; then
    try_join "${suite}"
    return 0
  fi
  # a chain stage failed: downstream stages are not run; the reduction still
  # runs so that every missing piece of evidence is reported as NOT_RUN
  echo "[GATE ${stage}] downstream chain not submitted; reducing what exists"
  submit_reduce_once "${suite}" "${SLURM_JOB_ID:-}"
  return 0
}

# -----------------------------------------------------------------------------
# finalizer: status, ordered logs, report, ZIPs, LATEST links
# -----------------------------------------------------------------------------
finalize_suite() {
  local suite="${SPLICE_SUITE}" tag log_out log_err master
  tag="$(cat "${suite}/000_MANIFEST/suite_tag.txt")"
  log_out="${LOGS_DIR}/${tag}_9999_FINALIZER.out"; log_err="${LOGS_DIR}/${tag}_9999_FINALIZER.err"
  attach_stage_logs "${log_out}" "${log_err}" "FINALIZER_${tag}"
  master="${suite}/999_MASTER"
  mkdir -p "${master}/RAW_LOGS"
  if [[ "$(cat "${LIVE_SUITE_FILE}" 2>/dev/null || true)" == "${tag}" ]]; then
    cp -f "${LIVE_OUT}" "${master}/SPLICE_V1_0_0_LIVE.out" 2>/dev/null || true
    cp -f "${LIVE_ERR}" "${master}/SPLICE_V1_0_0_LIVE.err" 2>/dev/null || true
  fi
  load_common_runtime
  [[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"
  local p n=0
  rm -f "${master}"/RAW_LOGS/* 2>/dev/null || true
  while IFS= read -r p; do
    n=$((n+1)); cp -Lf "${p}" "${master}/RAW_LOGS/$(printf '%05d' "${n}")_$(basename "${p}")" 2>/dev/null || true
  done < <(find "${LOGS_DIR}" -maxdepth 1 -type f \( -name "${tag}_*.out" -o -name "${tag}_*.err" \) | sort -V)
  cat "${master}"/RAW_LOGS/* > "${master}/ALL_RAW_LOGS.log" 2>/dev/null || true
  find "${LOGS_DIR}" -maxdepth 1 -type f \( -name "${tag}_*.out" -o -name "${tag}_*.err" \) -printf '%f\n' | sort -V > "${master}/ORDERED_LOG_INDEX.txt"
  if command -v sacct >/dev/null 2>&1 && [[ -s "${suite}/000_MANIFEST/job_ids.tsv" ]]; then
    local job_ids
    job_ids="$(awk -F '\t' 'NR>1 && $2!="" && $2!="SKIPPED"{print $2}' "${suite}/000_MANIFEST/job_ids.tsv" | sort -u | paste -sd, -)"
    if [[ -n "${job_ids}" ]]; then
      sacct -X -j "${job_ids}" -o JobID,JobName,State,ExitCode,Elapsed,Timelimit,NodeList,MaxRSS,MaxVMSize -P > "${master}/SACCT_JOBS.tsv" 2>&1 || true
    fi
  fi

  SPLICE_SUITE_DIR="${suite}" SPLICE_MASTER="${master}" SPLICE_TAG="${tag}" "${PYTHON}" - <<'PYFINAL'
import json, os, re, zipfile
from pathlib import Path

suite = Path(os.environ['SPLICE_SUITE_DIR'])
master = Path(os.environ['SPLICE_MASTER'])
tag = os.environ['SPLICE_TAG']
STAGES = [('baseline', '010_BASELINE'), ('train', '020_TRAIN'), ('calibrate', '030_CALIBRATION'), ('qat', '040_QAT'),
          ('evaluate', '070_EVALUATE'), ('diagnostics', '080_DIAGNOSTICS'), ('export', '085_EXPORT'), ('reduce', '900_REDUCE')]
CHAIN = ['baseline', 'train', 'calibrate', 'qat']
PARALLEL = ['evaluate', 'diagnostics', 'export']
GOOD = ('PASS', 'PASS_WITH_OPTIONAL_FAILURES', 'SKIPPED')


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


gates = {}
for st, d in STAGES:
    g = suite / d / 'GATE'
    gj = load(g / 'gate.json') or {}
    if (g / 'SKIPPED').exists():
        dec = 'SKIPPED'
    elif (g / 'PASS').exists():
        dec = gj.get('decision', 'PASS')
    elif (g / 'FAIL').exists():
        dec = 'FAIL'
    elif (g / 'RETRY').exists():
        dec = 'RETRY_PENDING_OR_ABANDONED'
    else:
        dec = 'NOT_RUN'
    gates[st] = {'decision': dec, 'fatal': gj.get('fatal', []), 'optional_failed': gj.get('optional_failed', []),
                 'retry': gj.get('retry', []), 'completed': gj.get('completed'), 'total': gj.get('total')}

summaries, failures = [], []
for p in sorted(suite.glob('*/runs/*/RESULT/summary.json')):
    s = load(p)
    if isinstance(s, dict):
        s['_path'] = str(p.parent)
        summaries.append(s)
seen = set()
for pattern in ('**/failure_summary.json', '**/shell_failure_*.json'):
    for p in sorted(suite.glob(pattern)):
        if 'ATTEMPTS' in p.parts:
            continue
        x = load(p)
        if not isinstance(x, dict):
            continue
        key = (x.get('job_id'), x.get('array_task_id'), x.get('status'), x.get('message'), str(p.parent))
        if key in seen:
            continue
        seen.add(key)
        x['_path'] = str(p.parent)
        failures.append(x)
with (master / 'ALL_SUMMARIES.jsonl').open('w') as f:
    for s in summaries:
        f.write(json.dumps(s, sort_keys=True, default=str) + '\n')
with (master / 'ALL_FAILURES.jsonl').open('w') as f:
    for x in failures:
        f.write(json.dumps(x, sort_keys=True, default=str) + '\n')

config = load(suite / '000_MANIFEST' / 'suite_config.json') or {}
reduce_summary = load(suite / '900_REDUCE' / 'summary.json') or {}
release = (reduce_summary.get('release') or {}).get('decision')
first_bad = next((st for st in CHAIN if gates[st]['decision'] not in GOOD), None)
parallel_bad = [st for st in PARALLEL if gates[st]['decision'] not in GOOD]
infra = [x for x in failures if str(x.get('failure_class', '')).upper() == 'INFRASTRUCTURE' or bool(x.get('retriable'))]
if config.get('dry_run'):
    scientific_status = 'DRY_RUN_NOT_A_MEASUREMENT'
elif not (suite / '000_MANIFEST' / 'PREPARED').exists():
    scientific_status = 'PREPARE_INCOMPLETE'
elif first_bad is not None:
    scientific_status = f'{first_bad.upper()}_STAGE_INCOMPLETE'
elif gates['reduce']['decision'] not in GOOD:
    scientific_status = 'REDUCTION_INCOMPLETE'
elif parallel_bad:
    scientific_status = 'EVALUATION_PARTIAL_' + '_'.join(s.upper() for s in parallel_bad) + '_INCOMPLETE'
elif release is not None:
    scientific_status = f'EVALUATION_COMPLETE_{str(release).upper()}'
else:
    scientific_status = 'EVALUATION_COMPLETE_DECISION_NOT_DETERMINED'
if config.get('scale') not in (None, 'full') and not config.get('dry_run'):
    scientific_status += f"_SCALE_{str(config.get('scale')).upper()}_NOT_QUALIFYING"
if infra and first_bad is not None:
    scientific_status += '_INFRASTRUCTURE'
preflight = load(suite / '000_MANIFEST' / 'SEMANTIC_PREFLIGHT' / 'PREFLIGHT_REPORT.json')
probe = load(suite / '000_MANIFEST' / 'DATASET_PREP' / 'API_PROBE.json')
preflight_status = 'NOT_RUN' if not isinstance(preflight, dict) else ('PASS' if preflight.get('passed') else 'FAIL')
probe_status = 'NOT_RUN' if not isinstance(probe, dict) else str(probe.get('status', 'PASS'))
if 'FAIL' in (preflight_status, probe_status.replace('FAILED', 'FAIL')) and not config.get('dry_run'):
    scientific_status += '_PREFLIGHT_FAILED'
status = {'version': '1.0.0-arrhenius', 'suite': str(suite), 'scientific_status': scientific_status,
          'train_channel_decision': release, 'scale': config.get('scale'),
          'semantic_preflight': preflight_status, 'api_probe': probe_status,
          'gates': {k: v['decision'] for k, v in gates.items()},
          'completed_summaries': len(summaries), 'failure_summaries': len(failures),
          'infrastructure_failure_summaries': len(infra),
          'note': 'Stage completion is not claim qualification; see 900_REDUCE/tests.json and claim_ledger.json. '
                  'Manuscript values are targets only (paper_target_comparison.csv), never measured results.'}
(master / 'SUITE_STATUS.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
(suite / 'SUITE_STATUS.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')

lines = [f'SPLICE-V V1.0.0 detector suite {tag}', f'suite: {suite}', f'scale: {config.get("scale")}',
         f'scientific_status: {scientific_status}', f'train-channel decision (tests 2, 3, 9, 10, 12): {release}', '',
         '===== GATES =====']
for st, _ in STAGES:
    g = gates[st]
    done = '' if g['completed'] is None else f"  ({g['completed']}/{g['total']} tasks)"
    lines.append(f'{st:12s} {g["decision"]}{done}')
    for x in g['fatal'][:20]:
        lines.append(f'    fatal: {x}')
    for x in g['optional_failed'][:20]:
        lines.append(f'    optional failure: {x}')
lines += ['', '===== SEMANTIC PREFLIGHT (000_MANIFEST/SEMANTIC_PREFLIGHT) =====', f'status: {preflight_status}']
for name, chk in ((preflight or {}).get('checks') or {}).items():
    brief = {k: v for k, v in chk.items() if k != 'traceback'} if isinstance(chk, dict) else chk
    lines.append(f'  {name}: {json.dumps(brief, default=str)[:600]}')
    log_file = suite / '000_MANIFEST' / 'SEMANTIC_PREFLIGHT' / f'{name}.log'
    if isinstance(chk, dict) and chk.get('returncode') not in (0, None) and log_file.exists():
        lines += ['    ' + x for x in log_file.read_text(errors='replace').splitlines()[-60:]]
    if isinstance(chk, dict) and chk.get('traceback'):
        lines += ['    ' + x for x in str(chk['traceback']).splitlines()[-40:]]
lines += ['', '===== API PROBE (000_MANIFEST/DATASET_PREP/API_PROBE.json) =====', f'status: {probe_status}']
for name, chk in (probe or {}).items():
    if isinstance(chk, dict):
        lines.append(f"  {name}: {json.dumps({k: v for k, v in chk.items() if k != 'traceback'}, default=str)[:600]}")
        if chk.get('traceback'):
            lines += ['    ' + x for x in str(chk['traceback']).splitlines()[-40:]]
prep = load(suite / '000_MANIFEST' / 'DATASET_PREP' / 'DATASET_REPORT.json')
if isinstance(prep, dict):
    prep = {k: v for k, v in prep.items() if k != 'api_probe'}
lines += ['', '===== DATASETS (000_MANIFEST/DATASET_PREP/DATASET_REPORT.json) =====']
lines += json.dumps(prep, indent=1, sort_keys=True, default=str).splitlines()[:120] if prep else ['NOT_RUN']
lines += ['', '===== MEASURED REPORT (900_REDUCE/MEASURED_REPORT.md) =====']
rep = suite / '900_REDUCE' / 'MEASURED_REPORT.md'
lines += rep.read_text().splitlines() if rep.exists() else ['NOT_RUN: the reduction did not produce a report.']
lines += ['', '===== MANUSCRIPT TARGETS vs MEASURED (900_REDUCE/paper_target_comparison.csv) =====',
          'Manuscript values are targets from the paper, listed only for comparison.']
cmp_csv = suite / '900_REDUCE' / 'paper_target_comparison.csv'
lines += cmp_csv.read_text().splitlines() if cmp_csv.exists() else ['NOT_RUN']
lines += ['', '===== FAILURES =====']
if failures:
    for x in failures[:200]:
        lines.append(f"{x.get('status')} class={x.get('failure_class')} stage={x.get('stage')} task={x.get('task')} "
                     f"msg={str(x.get('message', x.get('command', '')))[:300]} path={x.get('_path')}")
else:
    lines.append('No failure summaries.')
problems = []
pat = re.compile(r'(Traceback|\[FATAL\]|Error:|CUDA error|out of memory|Killed|DUE TO TIME LIMIT)', re.I)
for p in sorted((master / 'RAW_LOGS').glob('*')):
    try:
        for i, line in enumerate(p.read_text(errors='replace').splitlines()):
            if pat.search(line):
                problems.append(f'{p.name}:{i + 1}: {line[:300]}')
    except Exception:
        pass
(master / 'FAILURES_AND_WARNINGS.txt').write_text('\n'.join(problems) + '\n')
lines += ['', '===== FATAL PATTERNS IN LOGS ====='] + (problems[:500] if problems else ['No fatal patterns found.'])
text = '\n'.join(lines) + '\n'
(master / 'ALL_RESULTS.txt').write_text(text)
(suite / 'ALL_RESULTS.txt').write_text(text)

SKIP_SUFFIX = ('.pt', '.pth', '.npz', '.ckpt', '.svb')
compact = suite / 'SPLICE_V1_0_0_RESULTS.zip'
full = suite / 'SPLICE_V1_0_0_FULL_RESULTS_NO_CHECKPOINTS.zip'
added = set()


def add(z, p):
    name = str(p.relative_to(suite))
    if name not in added:
        added.add(name)
        z.write(p, name)


with zipfile.ZipFile(compact, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for p in [suite / 'ALL_RESULTS.txt', suite / 'SUITE_STATUS.json', master / 'FAILURES_AND_WARNINGS.txt',
              master / 'ORDERED_LOG_INDEX.txt', master / 'SACCT_JOBS.tsv', master / 'ALL_FAILURES.jsonl',
              master / 'ALL_SUMMARIES.jsonl']:
        if p.exists():
            add(z, p)
    for p in sorted((suite / '000_MANIFEST').rglob('*')):
        rel = p.relative_to(suite / '000_MANIFEST').parts
        if p.is_file() and not p.name.endswith(SKIP_SUFFIX) and p.stat().st_size < 20 * 1024 * 1024 \
                and rel[0] not in ('code', 'SEMANTIC_PREFLIGHT'):
            add(z, p)
    for p in sorted((suite / '000_MANIFEST' / 'SEMANTIC_PREFLIGHT').glob('*')):
        if p.is_file() and p.suffix in ('.json', '.log') and p.stat().st_size < 2 * 1024 * 1024:
            add(z, p)
    for p in sorted((suite / '900_REDUCE').rglob('*')) if (suite / '900_REDUCE').exists() else []:
        if p.is_file() and not p.name.endswith(SKIP_SUFFIX) and p.stat().st_size < 50 * 1024 * 1024:
            add(z, p)
    for st, d in STAGES:
        for p in sorted((suite / d / 'GATE').glob('*')) if (suite / d / 'GATE').exists() else []:
            if p.is_file():
                add(z, p)
        for pattern in ('*/RESULT/*.json', '*/RESULT/*.csv', '*/RESULT/*.txt', '*/RESULT/export/*.json'):
            for p in sorted((suite / d / 'runs').glob(pattern)) if (suite / d / 'runs').exists() else []:
                if p.name not in ('frames.json', 'pip_freeze.txt') and p.stat().st_size < 5 * 1024 * 1024:
                    add(z, p)
        for p in sorted((suite / d / 'runs').glob('*/RESULT/task_log.jsonl')) if (suite / d / 'runs').exists() else []:
            rows = p.read_text(errors='replace').splitlines()
            keep = rows if len(rows) <= 400 else rows[:100] + ['... (%d lines omitted) ...' % (len(rows) - 400)] + rows[-300:]
            if str(p.relative_to(suite)) not in added:
                added.add(str(p.relative_to(suite)))
                z.writestr(str(p.relative_to(suite)), '\n'.join(keep) + '\n')
    art = suite / 'ARTIFACTS'
    for p in sorted(art.glob('*/*.json')) if art.exists() else []:
        if p.stat().st_size < 5 * 1024 * 1024:
            add(z, p)
with zipfile.ZipFile(full, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for p in sorted(suite.rglob('*')):
        if not p.is_file() or p.resolve() in {compact.resolve(), full.resolve()}:
            continue
        if p.name.endswith(('.pt', '.pth', '.ckpt')):
            continue
        if p.stat().st_size < 300 * 1024 * 1024:
            z.write(p, p.relative_to(suite))
print('[MASTER] status:', scientific_status)
print('[MASTER] report:', suite / 'ALL_RESULTS.txt')
print('[MASTER] compact ZIP:', compact)
print('[MASTER] full archive:', full)
PYFINAL
  rm -f "${OUTPUTS_DIR}/LATEST_RESULTS.zip" "${OUTPUTS_DIR}/SPLICE_V1_0_0_LATEST_RESULTS.zip"
  ln -s "${suite}/SPLICE_V1_0_0_RESULTS.zip" "${OUTPUTS_DIR}/LATEST_RESULTS.zip"
  ln -s "${suite}/SPLICE_V1_0_0_RESULTS.zip" "${OUTPUTS_DIR}/SPLICE_V1_0_0_LATEST_RESULTS.zip"
  printf '%s\n' "${suite}" > "${SPLICE_CACHE}/latest_completed_output.txt"
  echo "[LOG_FOLDER] ${LOGS_DIR}"
  echo "[LIVE_LOG_OUT] ${LIVE_OUT}"
  echo "[LIVE_LOG_ERR] ${LIVE_ERR}"
  echo "[OUTPUT_FOLDER] ${suite}"
  echo "[CHECKPOINT_FOLDER] $(cat "${suite}/000_MANIFEST/CHECKPOINT_DIRECTORY.txt" 2>/dev/null || echo NA)"
  touch "${suite}/RESULTS_ZIP_READY"
  echo "[RESULTS_ZIP] ${suite}/SPLICE_V1_0_0_RESULTS.zip"
}

# -----------------------------------------------------------------------------
# resume / postprocess an existing suite
# -----------------------------------------------------------------------------
archive_failed_results() {
  local suite="$1" stage="$2" rd stamp n=0
  stamp="$(date +%Y%m%d_%H%M%S)"
  while IFS= read -r rd; do
    [[ -f "${rd}/summary.json" && ! -f "${rd}/failure_summary.json" ]] && continue
    mkdir -p "$(dirname "${rd}")/ATTEMPTS"
    mv "${rd}" "$(dirname "${rd}")/ATTEMPTS/RESULT_resume_${stamp}"
    n=$((n+1))
  done < <(find "${suite}/$(stage_dir_of "${stage}")/runs" -mindepth 2 -maxdepth 2 -type d -name RESULT 2>/dev/null | sort)
  # budgets and markers of the earlier attempts start afresh
  find "${suite}/$(stage_dir_of "${stage}")/runs" -mindepth 2 -maxdepth 2 \( -name SKIPPED.txt -o -name CONTINUE_EXHAUSTED.json \
    -o -name CONTINUATION_COUNT.txt -o -name CONTINUE_PENDING.json -o -name INFRA_RETRY_EXHAUSTED.json \
    -o -name INFRA_REQUEUE_COUNT.txt -o -name INFRA_RETRY_PENDING.json \) -delete 2>/dev/null || true
  echo "[RESUME] ${stage}: archived ${n} incomplete/failed task result(s)"
}

# Jobs of this suite that are still queued or running (ids from job_ids.tsv).
suite_jobs_alive() {
  local suite="$1" ids
  command -v squeue >/dev/null 2>&1 || return 1
  ids="$(awk -F '\t' 'NR>1 && $2 ~ /^[0-9]+$/ {print $2}' "${suite}/000_MANIFEST/job_ids.tsv" 2>/dev/null | sort -un | paste -sd, -)"
  [[ -n "${ids}" ]] || return 1
  squeue -h -j "${ids}" -o %i 2>/dev/null | grep -q .
}

require_quiet_suite() {
  local suite="$1"
  [[ "${SPLICE_FORCE_RESUME:-0}" == 1 ]] && return 0
  if suite_jobs_alive "${suite}"; then
    echo "[FATAL] jobs of this suite are still queued or running; a second chain would race with them." >&2
    echo "[FATAL] Wait for them, or cancel them (job ids: ${suite}/000_MANIFEST/job_ids.tsv), then run this again." >&2
    echo "[FATAL] SPLICE_FORCE_RESUME=1 skips this check." >&2
    exit 12
  fi
}

# The reduction and the packaging are redone: the previous reduction is kept under ATTEMPTS/.
reset_reduce() {
  local suite="$1" rd stamp r
  rd="${suite}/$(stage_dir_of reduce)"
  rm -f "${suite}/000_MANIFEST/REDUCE_SUBMITTED" "${suite}/RESULTS_ZIP_READY" \
    "${rd}/GATE/PASS" "${rd}/GATE/FAIL" "${rd}/GATE/RETRY" "${rd}/GATE/RETRY_COUNT.txt" "${rd}/GATE/CONTINUE_ROUNDS.txt"
  stamp="$(date +%Y%m%d_%H%M%S)"
  while IFS= read -r r; do
    mkdir -p "$(dirname "${r}")/ATTEMPTS"
    mv "${r}" "$(dirname "${r}")/ATTEMPTS/RESULT_previous_${stamp}"
  done < <(find "${rd}/runs" -mindepth 2 -maxdepth 2 -type d -name RESULT 2>/dev/null | sort)
}

reopen_gate() {
  local g="$1"
  rm -f "${g}/FAIL" "${g}/RETRY" "${g}/RETRY_COUNT.txt" "${g}/CONTINUE_ROUNDS.txt" "${g}/gate_retry_exhausted.json"
}

resume_suite() {
  local suite; suite="$(readlink -f "${1:?Provide existing output folder}")"
  [[ -d "${suite}/000_MANIFEST" ]] || { echo "[FATAL] not a SPLICE suite: ${suite}" >&2; exit 10; }
  export SPLICE_SUITE="${suite}"
  local tag; tag="$(cat "${suite}/000_MANIFEST/suite_tag.txt")"
  attach_live_only "RESUME_${tag}"
  command -v sbatch >/dev/null 2>&1 || { echo "[FATAL] sbatch unavailable" >&2; exit 10; }
  require_quiet_suite "${suite}"
  if [[ -f "${suite}/000_MANIFEST/suite.env" ]]; then
    SCALE="$(awk -F= '$1=="scale"{print $2}' "${suite}/000_MANIFEST/suite.env")"; SCALE="${SCALE:-full}"
  fi
  if [[ ! -f "${suite}/000_MANIFEST/PREPARED" ]]; then
    echo "[RESUME] suite was never prepared; resubmitting the prepare job"
    local prep fj; prep="$(submit_prepare "${suite}" "${tag}")"; record_stage_job prepare_resume "${prep}"
    fj="$(submit_finalizer "${suite}" "${prep}")"
    printf '%s\n' "${fj}" > "${suite}/000_MANIFEST/PREP_FAILURE_FINALIZER_JOB.txt"
    record_stage_job finalizer_if_prepare_fails "${fj}"
    return 0
  fi
  reset_reduce "${suite}"
  local s g gj
  for s in "${CHAIN_STAGES[@]}"; do
    g="${suite}/$(stage_dir_of "${s}")/GATE"
    if [[ ! -f "${g}/PASS" && ! -f "${g}/SKIPPED" ]]; then
      reopen_gate "${g}"
      archive_failed_results "${suite}" "${s}"
      gj="$(submit_gate "${suite}" "${s}" "")"; record_stage_job "${s}_gate_resume" "${gj}"
      echo "[RESUME] first incomplete chain stage: ${s}; gate=${gj} (it resubmits the open tasks; the later stages follow)"
      return 0
    fi
  done
  local any=0
  for s in "${PARALLEL_STAGES[@]}"; do
    g="${suite}/$(stage_dir_of "${s}")/GATE"
    if [[ ! -f "${g}/PASS" && ! -f "${g}/SKIPPED" ]]; then
      reopen_gate "${g}"
      archive_failed_results "${suite}" "${s}"
      gj="$(submit_gate "${suite}" "${s}" "")"; record_stage_job "${s}_gate_resume" "${gj}"
      echo "[RESUME] evaluation stage ${s}: gate=${gj}"
      any=1
    fi
  done
  if (( any == 0 )); then
    echo "[RESUME] every stage is complete; redoing the reduction and the packaging"
    submit_reduce_once "${suite}" ""
  fi
}

postprocess_existing() {
  local suite; suite="$(readlink -f "${1:?Provide existing output folder}")"
  [[ -d "${suite}/000_MANIFEST" ]] || { echo "[FATAL] not a SPLICE suite: ${suite}" >&2; exit 10; }
  export SPLICE_SUITE="${suite}"
  attach_live_only "POSTPROCESS_$(cat "${suite}/000_MANIFEST/suite_tag.txt")"
  command -v sbatch >/dev/null 2>&1 || { echo "[FATAL] sbatch unavailable" >&2; exit 10; }
  require_quiet_suite "${suite}"
  if [[ -f "${suite}/000_MANIFEST/suite.env" ]]; then
    SCALE="$(awk -F= '$1=="scale"{print $2}' "${suite}/000_MANIFEST/suite.env")"; SCALE="${SCALE:-full}"
  fi
  reset_reduce "${suite}"
  submit_reduce_once "${suite}" ""
  echo "[POSTPROCESS] reduction and packaging resubmitted for ${suite}"
}

# -----------------------------------------------------------------------------
# check mode: storage, directories, code, allocation, datasets (login node; submits nothing)
# -----------------------------------------------------------------------------
run_check() {
  trap - ERR EXIT
  set +eE +o pipefail
  local fails=0 warns=0 d f m acct part limit assoc out py s
  ok() { printf '[ OK ] %s\n' "$*"; }
  warn() { printf '[WARN] %s\n' "$*"; warns=$((warns+1)); }
  bad() { printf '[FAIL] %s\n' "$*"; fails=$((fails+1)); }
  echo "SPLICE-V ${EXPECTED_VERSION} launcher check (utc $(date -u +%Y-%m-%dT%H:%M:%SZ), host $(hostname -s), user ${USER:-$(id -un)})"

  echo; echo "== Storage and directories"
  echo "     storage project folder: ${BASE}"
  if [[ -d "${BASE}" ]]; then ok "project folder exists"; else bad "project folder missing: ${BASE} (create it or set SPLICE_BASE)"; fi
  for d in code datasets cache logs outputs checkpoints; do
    if [[ -d "${BASE}/${d}" && -w "${BASE}/${d}" ]]; then ok "${d}/ exists and is writable"
    elif [[ -d "${BASE}/${d}" ]]; then bad "${d}/ exists but is not writable for ${USER:-you}"
    else bad "${d}/ missing under ${BASE}"; fi
  done
  if command -v storagequota >/dev/null 2>&1; then
    echo "     storagequota (bytes and file counts of your storage projects):"
    timeout 60 storagequota 2>&1 | head -n 30 | sed 's/^/       /'
  else
    warn "storagequota not found on this host: check the quota of the storage project by hand"
  fi
  df -h "${BASE}" 2>/dev/null | sed 's/^/       /'
  echo "     expected use: datasets about 35 GB and 0.3 million files (raw images and the label files of the"
  echo "     views), venv about 8 GB, checkpoints of a full suite about 20 GB, outputs a few GB"

  echo; echo "== Code"
  if [[ -f "${SELF}" ]]; then ok "launcher: ${SELF}"; else bad "launcher missing in the code folder: cp ${REPO_DIR}/detector/hpc/${EXPECTED_SHELL_BASENAME} ${CODE_DIR}/"; fi
  if [[ -f "${PY_SCRIPT_SOURCE}" ]]; then ok "runner: ${PY_SCRIPT_SOURCE}"; else bad "runner missing in the code folder: cp ${REPO_DIR}/detector/hpc/${EXPECTED_RUNNER_BASENAME} ${CODE_DIR}/"; fi
  if [[ -f "${REPO_DIR}/detector/splice_det/__init__.py" && -d "${REPO_DIR}/reference/splicev_ref" && -d "${REPO_DIR}/generated/py" ]]; then
    ok "repository: ${REPO_DIR} (commit $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown))"
    for f in "${EXPECTED_SHELL_BASENAME}" "${EXPECTED_RUNNER_BASENAME}"; do
      if [[ -f "${CODE_DIR}/${f}" && -f "${REPO_DIR}/detector/hpc/${f}" ]] && ! cmp -s "${CODE_DIR}/${f}" "${REPO_DIR}/detector/hpc/${f}"; then
        warn "code/${f} differs from the repository copy; the suite runs the copy in code/ (cp ${REPO_DIR}/detector/hpc/${f} ${CODE_DIR}/)"
      fi
    done
  else
    bad "repository not found at ${REPO_DIR}: cd ${CODE_DIR} && git clone https://github.com/AnonymousHERE-commits/ISCA2027.git SPLICE-V   (or set SPLICE_REPO)"
  fi
  if [[ -f "${PY_SCRIPT_SOURCE}" ]]; then
    if grep -Fqx 'VERSION = "1.0.0-arrhenius"' "${PY_SCRIPT_SOURCE}" && grep -Fqx 'VERSION_TAG = "SPLICE_V1_0_0"' "${PY_SCRIPT_SOURCE}"; then
      ok "runner version markers match ${EXPECTED_VERSION}"
    else
      bad "runner version markers do not match ${EXPECTED_VERSION}: ${PY_SCRIPT_SOURCE}"
    fi
  fi

  echo; echo "== Allocation"
  acct="$(suite_account)"; part="$(suite_partition)"
  echo "     account (compute project): ${acct}     partition: ${part}"
  if [[ "${acct}" == *-gpu ]]; then ok "the account name ends with -gpu (required for the GPU nodes)"; else bad "the account must end with -gpu to reach the GPU nodes: '${acct}'"; fi
  if [[ -f "${SELF}" ]] && grep -Eq '^#SBATCH +(-N|--nodes)' "${SELF}"; then
    warn "the launcher header names -N/--nodes: on Arrhenius this allocates whole nodes (four GH200) to every job"
  fi
  if command -v sbatch >/dev/null 2>&1; then
    assoc="$(timeout 60 sacctmgr -nP show assoc where user="${USER:-$(id -un)}" format=account 2>/dev/null | sort -u)"
    if [[ -n "${assoc}" ]] && grep -Fxq "${acct}" <<< "${assoc}"; then ok "account ${acct} is among your Slurm associations"
    elif [[ -n "${assoc}" ]]; then bad "account ${acct} is not among your Slurm associations: $(paste -sd' ' <<< "${assoc}")"
    else warn "could not list your Slurm associations (sacctmgr); compare with projinfo below"; fi
    limit="$(sinfo -h -p "${part}" -o %l 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    if [[ -n "${limit}" ]]; then ok "partition ${part} exists; wall-time limit ${limit}"; else bad "partition ${part} not found by sinfo"; fi
    echo "     requests per task (time, memory, CPUs, one GPU each; times above the partition limit are lowered to it):"
    for s in "${ALL_STAGES[@]}"; do printf '       %-12s %s  %s  %s CPUs\n' "${s}" "$(stage_time "${s}")" "$(stage_mem "${s}")" "$(stage_cpus "${s}")"; done
    if [[ -f "${SELF}" ]]; then
      if out="$(sbatch --test-only "${SELF}" smoke 2>&1)"; then ok "sbatch --test-only accepts the parent job: ${out}"
      else bad "sbatch --test-only rejects the parent job: ${out}"; fi
      if out="$(sbatch --test-only --time="$(stage_time train)" --cpus-per-task="$(stage_cpus train)" --mem="$(stage_mem train)" --gpus=1 "${SELF}" smoke 2>&1)"; then
        ok "sbatch --test-only accepts a training task: ${out}"
      else
        bad "sbatch --test-only rejects a training task (lower SPLICE_TIME_TRAIN / SPLICE_MEM_TRAIN / SPLICE_WORKER_CPUS): ${out}"
      fi
    fi
    if command -v projinfo >/dev/null 2>&1; then
      echo "     projinfo:"; timeout 60 projinfo 2>&1 | head -n 40 | sed 's/^/       /'
    fi
  else
    warn "sbatch not found: run this check on an Arrhenius login node to verify the account and the partition"
  fi

  echo; echo "== Datasets (raw downloads; the suite builds its views under ${DATA_DIR}/splice_views)"
  py="$(command -v python3 || true)"
  if [[ -z "${py}" ]]; then
    warn "python3 not found on this host: dataset folders not inspected"
  elif [[ ! -f "${REPO_DIR}/detector/splice_det/data.py" ]]; then
    warn "repository missing: dataset folders not inspected"
  else
    out="$("${py}" - "${REPO_DIR}/detector/splice_det/data.py" "${COCO_ROOT}" "${CROWDHUMAN_ROOT}" "${DATASETS}" <<'PYCHECKDATA'
import importlib.util, json, sys
from pathlib import Path

data_py, coco_root, ch_root, datasets = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split()
spec = importlib.util.spec_from_file_location("splice_data_check", data_py)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
FULL = {"train2017": 118287, "val2017": 5000, "train": 15000, "val": 4370}


def line(kind, msg):
    print(f"[{kind}] {msg}")


if "coco" in datasets:
    try:
        c = m.discover_coco(coco_root)
        for split in ("train2017", "val2017"):
            n = len(m._images_in(c[split]))
            line(" OK " if n == FULL[split] else "WARN", f"COCO {split}: {n} images in {c[split]}"
                 + ("" if n == FULL[split] else f" (the full split has {FULL[split]})"))
        line(" OK ", f"COCO val annotations: {c['ann_val2017']}")
        if "ann_train2017" in c:
            line(" OK ", f"COCO train annotations: {c['ann_train2017']}")
        else:
            line(" OK ", f"COCO train labels: existing YOLO labels in {c['labels_train2017']}")
    except m.DataError as e:
        line("FAIL", str(e))
if "crowdhuman" in datasets:
    try:
        c = m.discover_crowdhuman(ch_root)
        for split in ("train", "val"):
            ids = [json.loads(x)["ID"] for x in Path(c[split]).read_text().splitlines() if x.strip()]
            have = sum(1 for i in ids if i in c["index"])
            kind = " OK " if have == len(ids) == FULL[split] else ("WARN" if have else "FAIL")
            line(kind, f"CrowdHuman {split}: {have} of {len(ids)} annotated images on disk ({c[split]})"
                 + ("" if kind == " OK " else f" (the full split has {FULL[split]})"))
    except m.DataError as e:
        line("FAIL", str(e))
PYCHECKDATA
)"
    if [[ -n "${out}" ]]; then printf '%s\n' "${out}"; fi
    fails=$((fails + $(grep -c '^\[FAIL\]' <<< "${out}")))
    warns=$((warns + $(grep -c '^\[WARN\]' <<< "${out}")))
    if grep -q 'Traceback' <<< "${out}"; then bad "the dataset inspection crashed (see above)"; fi
  fi

  echo; echo "== Runtime"
  if [[ -f "${WEIGHTS}" ]]; then ok "pretrained weights: ${WEIGHTS}"
  else warn "pretrained weights not found: ${WEIGHTS} (the prepare job downloads yolo26s.pt; without outbound network on the GPU nodes, place the file there)"; fi
  if [[ -x "${PYTHON}" ]]; then ok "python environment: ${PYTHON}"
  else warn "python environment not built yet: the first prepare job creates ${VENV_DIR} (torch 2.12.1+cu130, torchvision, ultralytics >= 8.4, pycocotools)"; fi
  if type module >/dev/null 2>&1; then
    for m in GPU/buildenv-nvhpc/25.9-cu13.0 GPU/Python/3.13.5-bare-gcc-2025b-eb; do
      if module is-avail "${m}" >/dev/null 2>&1; then ok "module ${m}"; else warn "module ${m} is not visible from this host (the jobs load it on the GPU nodes)"; fi
    done
  else
    warn "no module command in this shell: modules not inspected"
  fi

  echo
  echo "== Result: ${fails} failure(s), ${warns} warning(s)"
  if (( fails > 0 )); then echo "   Fix the [FAIL] lines before submitting."; exit 1; fi
  echo "   Next: sbatch ${SELF} smoke     (a few iterations of every stage; then pilot, then the full suite)"
  exit 0
}

# ---- Main dispatch -------------------------------------------------------------
case "${SPLICE_ROLE:-orchestrator}" in
  orchestrator)
    case "${1:-full}" in
      help|-h|--help) usage ;;
      check) run_check ;;
      postprocess) postprocess_existing "${2:?Usage: sbatch SCRIPT postprocess /absolute/output/folder}" ;;
      resume) resume_suite "${2:?Usage: sbatch SCRIPT resume /absolute/output/folder}" ;;
      full|pilot|smoke) orchestrate "${1:-full}" ;;
      *) echo "[FATAL] unknown mode=${1:-}" >&2; usage; exit 10 ;;
    esac
    ;;
  prepare) run_prepare ;;
  worker) run_worker ;;
  gate) run_gate ;;
  finalizer) finalize_suite ;;
  *) echo "[FATAL] unknown SPLICE_ROLE=${SPLICE_ROLE}" >&2; exit 99 ;;
esac
