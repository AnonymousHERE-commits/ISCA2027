# The SPLICE detector suite on Arrhenius

`run_splicev.sh` trains, calibrates, quantizes, evaluates and exports the SPLICE
detector of SPLICE-V on COCO 2017 and CrowdHuman and reduces the results of the paper's
`train` channel: Table 7 (YOLO26-S, the seven readout arms, the gate, the manifest's sites, the
six forecast-descriptor variants), the declared tests 2, 3, 9, 10 and 12 of Table 6 with the
arm (f) comparison, and the `train` rows of Table 5.

| file | role |
|---|---|
| `run_splicev.sh` | the launcher: one Slurm job chain per suite (prepare, stages, gates, reduction, packaging) |
| `run_splicev_arrhenius.py` | the runner every job calls: one task of one stage, the gates, the reduction |
| `download_datasets.sh` | COCO 2017, CrowdHuman and the YOLO26-S weights into the project folder |

## Allocation and storage

| item | value | where it comes from |
|---|---|---|
| compute project (Slurm account) | `naiss2025-5-643-gpu` | header of the launcher; GPU nodes need the suffix `-gpu` |
| partition | `gpu` | nodes with four GH200 (72 Arm cores, 96 GB GPU memory and 128 GB main memory per GPU) |
| storage project folder | `/nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V` | `BASE` in the launcher (`SPLICE_BASE` overrides it) |

Every job of a suite is one task on one GPU.  The launcher header has no `-N` line: on
Arrhenius, naming `-N` in a job script allocates the whole node (four GPUs) to the job.  The
header's `#SBATCH` lines are also the defaults of every job the suite submits; a worker
overrides time, CPUs (32) and memory (110 GB), which stay inside one GPU's share of a node.
`SBATCH_ACCOUNT` and `SBATCH_PARTITION` in the environment override account and partition
for a whole suite.

`bash run_splicev.sh check` verifies all of this on the login node without submitting
or writing anything: the folders, the code, the account among your Slurm associations, the
partition and its wall-time limit, `sbatch --test-only` for the parent job and for a training
task, the storage quota (`storagequota`), the datasets, the weights and the modules.

Layout under the project folder (the six folders exist already):

```
code/         SPLICE-V/                      git clone of github.com/AnonymousHERE-commits/ISCA2027
              run_splicev.sh         copies of the two launcher files (the suite runs these)
              run_splicev_arrhenius.py
              venv_arrhenius_gh200/          built by the first prepare job
datasets/     coco/                          train2017/ val2017/ annotations/
              crowdhuman/                    annotation_{train,val}.odgt Images/
              splice_views/                  built by the suite: train_fit / calib / val per dataset
checkpoints/  pretrained/yolo26s.pt
              SPLICE_V1_0_0_S####/<run>/{last,trained,calibrated,last_qat,qat}.pt
cache/        suite counter, pip, torch and Ultralytics caches
logs/         SPLICE_V1_0_0_LIVE.out/.err and one ordered log pair per job
outputs/      SPLICE_V1_0_0_SUITE_####/  one folder per suite
```

Expected use of the storage project (estimates): about 35 GB and 0.3 million files for the
datasets (the views link the raw images with hard links, which add no file to the quota, and
write one label file per image), about 8 GB for the Python environment, roughly 20 GB of
checkpoints and a few GB of outputs for a full suite.

## Setup (once)

```bash
cd /nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V/code
git clone https://github.com/AnonymousHERE-commits/ISCA2027.git SPLICE-V
cp SPLICE-V/detector/hpc/run_splicev.sh SPLICE-V/detector/hpc/run_splicev_arrhenius.py .

# datasets and weights (login node, inside tmux or screen; COCO train2017 alone is about 19 GB)
bash SPLICE-V/detector/hpc/download_datasets.sh

bash run_splicev.sh check
```

After a `git pull`, copy the two launcher files again; `check` warns when the copies in
`code/` differ from the repository.  Datasets that already exist elsewhere can be used in
place: `SPLICE_COCO_ROOT=/path/coco SPLICE_CROWDHUMAN_ROOT=/path/crowdhuman` (the folders
`train2017`, `val2017`, `annotations` and the `odgt` files are located automatically, also
below `images/`).

## Running

```bash
sbatch run_splicev.sh smoke     # first: every stage and gate on the real data, a few iterations each
sbatch run_splicev.sh pilot     # then: 2 epochs on 5% of the training images, 500 validation images
sbatch run_splicev.sh           # the registered protocol
```

| scale | runs trained | tasks: baseline / train / calibrate / qat / evaluate / diagnostics / export | what it is for |
|---|---|---|---|
| `smoke` | 7 (arms a, c, e; native on CrowdHuman; seed 0) | 1 / 7 / 6 / 2 / 9 / 2 / 2 | modules, environment, datasets, the Ultralytics and YOLO26 interfaces, the job chain; one hour per job; its numbers mean nothing |
| `pilot` | 9 (arms a, e, f, g; seed 0) | 1 / 9 / 8 / 2 / 11 / 4 / 2 | seconds per iteration and memory on the GH200, a first look at every table; not a qualifying run |
| full | 46 | 1 / 46 / 42 / 6 / 52 / 18 / 6 | the protocol below |

The registered protocol (full): arms (a), (e), (f), (g) with seeds 0, 1, 2 on both datasets;
arms (b), (c), (d) and the six descriptor variants of arm (e) with seed 0; YOLO26-S trained by
the same loop with three seeds on CrowdHuman and one on COCO, where the published checkpoint
is also evaluated in this pipeline (the `baseline` stage); 50 epochs from the published
YOLO26-S weights (batch 16, 640 x 640), calibration on the calibration split, 12 epochs of
quantization-aware training of arm (e) per seed.  A COCO training is about 7,085 iterations
per epoch and a CrowdHuman training about 860; the pilot's `task_log.jsonl` files give the
seconds per iteration (`s_per_it`) from which the cost of the full suite follows before it is
submitted.

The smoke scale is also the first execution of the PyTorch code on real data: the prepare job
runs the unit tests of the code snapshot and the autograd self-checks on real torch and probes
the installed Ultralytics interfaces.  In the smoke scale a failure there is recorded and
reported and the suite goes on, so that one pass shows every problem; pilot and full suites
stop in the prepare job instead.

Selections (space-separated lists, exported before `sbatch`):

```bash
SPLICE_DATASETS="crowdhuman" SPLICE_ARMS="a e f g" SPLICE_SEEDS="0 1 2" SPLICE_DESCRIPTORS=0 sbatch run_splicev.sh
```

| variable | default | meaning |
|---|---|---|
| `SPLICE_DATASETS` | `coco crowdhuman` | datasets of the suite |
| `SPLICE_ARMS` | scale's protocol | readout arms to train (`a` and `e` are needed by the declared tests) |
| `SPLICE_SEEDS` | `0 1 2` (full) | seeds; the declared tests need three |
| `SPLICE_DESCRIPTORS` | 1 (full) | the six forecast-descriptor variants of arm (e) |
| `SPLICE_NATIVE_COCO` | 1 (full) | number of seeds of YOLO26-S trained on COCO by this loop |
| `SPLICE_EPOCHS`, `SPLICE_QAT_EPOCHS` | 50, 12 | training and QAT epochs |
| `SPLICE_BATCH`, `SPLICE_FRACTION`, `SPLICE_VAL_LIMIT` | 16, 1.0, all | batch size, share of the training images, validation images |
| `SPLICE_MAX_CONCURRENT` | 6 | tasks of a stage that run at the same time |
| `SPLICE_WORKER_CPUS`, `SPLICE_CPUS_<STAGE>` | 32 | CPUs per task (data loading and the neck's CPU steps) |
| `SPLICE_TIME_<STAGE>`, `SPLICE_MEM_<STAGE>` | see `check` | wall time and memory per task of a stage |
| `SPLICE_STAGE_RETRIES` | 2 | resubmissions of failed or missing tasks per stage |
| `SPLICE_MAX_INFRA_REQUEUES` | 3 | requeues of a task after a GPU or node fault |
| `SPLICE_MAX_CONTINUATIONS` | 8 | continuations of a training that reaches its time limit |
| `SPLICE_COCO_ROOT`, `SPLICE_CROWDHUMAN_ROOT`, `SPLICE_WEIGHTS`, `SPLICE_REPO` | under the project folder | other locations |

## Following a suite

```bash
cd /nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V/logs
tail -n 100 -F SPLICE_V1_0_0_LIVE.out SPLICE_V1_0_0_LIVE.err
squeue -u $USER
```

Every job mirrors its own log pair into the two LIVE files; the per-job logs are
`SPLICE_V1_0_0_S####_<stage number>_<ordinal>_<task>.out/.err`.  The suite folder
(`outputs/SPLICE_V1_0_0_SUITE_####`, also `outputs/SPLICE_V1_0_0_LATEST_OUTPUT`):

| path | content |
|---|---|
| `000_MANIFEST/` | the suite's code snapshot and its hashes, `suite_config.json`, the task lists, `declared_tests.json`, `SITE.txt` (account, partition, limits, quota at submission), `WHERE_ARE_MY_FILES.txt`, the preflight and dataset reports, `run_registry.tsv`, `job_ids.tsv` |
| `010_BASELINE` ... `085_EXPORT` | `runs/<ordinal>_<task>/RESULT/` (summary, metrics, per-image matches, logs; earlier attempts under `ATTEMPTS/`) and `GATE/` (decision and reasons); the integer export of a QAT model is `085_EXPORT/runs/<ordinal>_<run>/RESULT/export/` |
| `ARTIFACTS/<run>/` | `train.json`, `calibration.json`, `qat.json` (state and hashes of the run's checkpoints) |
| `900_REDUCE/` | `MEASURED_REPORT.md`, `table7_measured.csv`, `tests.json`, `train_channel_parameters.csv`, `paper_target_comparison.csv`, `claim_ledger.json`, `measured_gain.json` |
| `SUITE_STATUS.json`, `ALL_RESULTS.txt` | status of every gate, the report, the failures and the fatal lines of all logs in one text |
| `SPLICE_V1_0_0_RESULTS.zip` | the compact results archive (no checkpoints, no per-image arrays); `outputs/LATEST_RESULTS.zip` points to the newest |
| `SPLICE_V1_0_0_FULL_RESULTS_NO_CHECKPOINTS.zip` | everything except the checkpoints |

Reading the result: `SUITE_STATUS.json` says how far the suite came; `900_REDUCE/tests.json`
holds one decision per declared test.  A test is `PASS` or `FAIL` only with its full evidence
(three seeds, both variants, the full validation set); with less it is `NOT_QUALIFIED` or
`NOT_EVALUATED`.  The manuscript's values appear only in `paper_target_comparison.csv`, next
to the measured ones, as targets.

## The job chain

```
parent (20 min) -> prepare -> baseline -> train -> calibrate -> qat -> { evaluate, diagnostics, export } -> reduce -> finalizer
```

- **prepare** builds or checks the environment (torch 2.12.1+cu130 and torchvision from the
  PyTorch index, Ultralytics >= 8.4, pycocotools), verifies the runner, runs the semantic
  preflight, writes the suite configuration and the task lists, builds the dataset views,
  fetches the weights if they are missing and probes the installed interfaces.
- Each **stage** is a job array (at most `SPLICE_MAX_CONCURRENT` tasks at a time) followed by
  a **gate** job.  The gate passes the chain on, resubmits failed or missing tasks (at most
  `SPLICE_STAGE_RETRIES` times), or stops the chain; a failure of an optional run (arms b, c,
  d, f, g and the descriptor variants) is recorded and does not stop it, a failure of
  YOLO26-S or of arms (a) and (e) does.  When the chain stops, what exists is still reduced
  and packaged.
- A task whose GPU fails its health check, or that dies of a CUDA or node fault, is requeued
  on another node (`scontrol requeue`); where a requeue is not possible the gate resubmits it.
  A job that Slurm ends (time limit, preemption, `scancel`) records that and is resubmitted
  by its gate within the retry budget.
- A **training** checkpoints every epoch.  It stops before the job's time limit and continues
  from the checkpoint in a new job (requeue, or resubmission by the gate), at most
  `SPLICE_MAX_CONTINUATIONS` times.  Stage times above the partition's limit are lowered to
  it, so the default 24-hour chunks can be raised or lowered freely.
- **reduce** runs when the three evaluation stages have ended (the bootstrap's replicates run
  in parallel on the task's CPUs); the **finalizer** collects the logs, writes the status and
  the archives.

```bash
sbatch run_splicev.sh resume /nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V/outputs/SPLICE_V1_0_0_SUITE_0001
sbatch run_splicev.sh postprocess <suite folder>     # reduction and packaging only
```

`resume` continues from the first stage that did not pass: completed tasks are kept, failed
ones are archived and rerun, trainings continue from their checkpoints, and the reduction is
redone.  Both refuse to start while jobs of the suite are still queued or running.

Exit codes of a task: 0 done; 33 an input or an upstream result is missing; 74 a result file
is missing after the run; 75 infrastructure (requeued); 76 stopped before the time limit
(continues); anything above 128 is a kill by signal (retried by the gate); other codes are
failures of the code or of a check and are not retried.

## Protocol notes

- **Roles of the data.**  `train_fit` is the training split without the calibration split;
  `calib` (1/24 of COCO train2017, 1/12 of CrowdHuman train, chosen by a hash of the file
  name) is used for the band exponents, the conformal thresholds, the partner prior, the
  activation scales and the sensitivity; `val` is COCO val2017 (5,000 images) and CrowdHuman
  val (4,370).  The validation sets are never used for calibration or selection.
- **COCO** AP and AP75 come from pycocotools on the original annotation file.
- **CrowdHuman**: full-body boxes of `person` entries; entries with another tag or
  `ignore = 1` are ignore regions (left out of training, handled by the official matching
  rule in the evaluation).  Training labels are clipped to the image; the evaluation matches
  against the boxes clipped to the image and reports MR^-2 on all persons and on the heavily
  occluded ones (visible ratio below 0.65).
- **Gate.**  Every SPLICE model is evaluated with and without the objectness gate from the
  same forward pass; test 12 compares the recall on heavily occluded persons.
- **Manifest's sites.**  The `manifest` variant is the QAT model with every rounding site of
  the manifest active.  It is the trainable stand-in for the integer model M; the bit-exact
  model M and the device belong to the hardware track of this repository.
- **Skip rule.**  The skipped share of class-head blocks at theta_det = 0.25 is reported for
  blocks of 1 x 1, 2 x 2 and 4 x 4 cells, because the paper does not fix the block size.
- **Baseline.**  In a full suite the published YOLO26-S checkpoint must reach at least 46.0 AP
  on COCO val2017 in this pipeline (the paper reports 47.6), otherwise the baseline stage
  fails and the chain stops.

## What has been run

The launcher, the runner's orchestration (task lists, gates, retries, requeues,
continuations, resume, reduction, packaging), the dataset views, the two evaluators, the
bootstrap and the download script are covered by the repository's tests, which drive the
real launcher through a mock Slurm (`tests/test_hpc_suite.py`, `tests/mock_slurm`) in its dry
mode (`SPLICE_DRY_RUN=1`: synthetic task results through the real file formats, no GPU).
The PyTorch model, training and inference code is covered there in forward mode only and has
not been executed on a GPU or on the real datasets: the smoke scale on Arrhenius is its first
real run.

## If something fails

| symptom | where to look |
|---|---|
| `sbatch` rejects the job | `bash run_splicev.sh check`: account, partition, time limit, memory |
| the suite stops in the prepare job | `logs/..._0000_PREP.err`; `000_MANIFEST/SEMANTIC_PREFLIGHT/*.log`; `000_MANIFEST/DATASET_PREP/DATASET_REPORT.json` and `API_PROBE.json` |
| a stage gate fails | `<stage folder>/GATE/gate.txt` names the tasks and their failure class; the task's log pair is in `logs/` |
| a training keeps continuing | `SPLICE_TIME_TRAIN` (longer chunks) or `SPLICE_MAX_CONTINUATIONS` |
| out of memory on the GPU | `SPLICE_BATCH=8` (a new suite) |
| quota exceeded | `storagequota`; old suites under `outputs/` and their checkpoints under `checkpoints/` can be deleted |

`ALL_RESULTS.txt` of the suite collects the gate decisions, the failures and the
fatal lines of every log.
