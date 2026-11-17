"""The Arrhenius launcher (detector/hpc/run_splicev.sh) and its runner, end to end
under a mock Slurm (tests/mock_slurm): the job chain with dependencies, gates, bounded
retries, requeue after infrastructure faults, continuation of time-limited trainings,
optional and fatal task failures, skipped stages, resume, the reduction and the packaging,
and the read-only check mode (directories, account, partition, datasets).

The suite runs in its dry mode (SPLICE_DRY_RUN=1): tasks produce synthetic results through
the real evaluators and file formats, so no GPU, torch or dataset is needed.  What a real
task computes is covered by the detector tests and, on the cluster, by the smoke scale.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MOCK = REPO / "tests" / "mock_slurm"
HPC = REPO / "detector" / "hpc"
SHELL, RUNNER = "run_splicev.sh", "run_splicev_arrhenius.py"

sys.path.insert(0, str(MOCK))
sys.path.insert(0, str(REPO / "tests"))
import mock_slurm  # noqa: E402


@unittest.skipUnless(shutil.which("bash") and shutil.which("flock") and shutil.which("tar"), "needs bash, flock and tar")
class Launcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.base = root / "SPLICE-V"
        for d in ("code", "datasets", "cache", "logs", "outputs", "checkpoints"):
            (self.base / d).mkdir(parents=True)
        shutil.copy(HPC / SHELL, self.base / "code" / SHELL)
        shutil.copy(HPC / RUNNER, self.base / "code" / RUNNER)
        os.symlink(REPO, self.base / "code" / "SPLICE-V")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("SPLICE_", "SLURM_"))}
        self.env.update(PATH=f"{MOCK}{os.pathsep}{os.environ['PATH']}", SPLICE_BASE=str(self.base),
                        MOCK_SLURM_DIR=str(root / "slurm"), SPLICE_DRY_RUN="1", SPLICE_PYTHON=sys.executable,
                        HOME=str(root))

    def tearDown(self):
        self.tmp.cleanup()

    def launch(self, *args, **extra):
        env = dict(self.env, **extra)
        r = subprocess.run(["sbatch", str(self.base / "code" / SHELL), *args], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        old = os.environ.get("MOCK_SLURM_DIR")
        os.environ["MOCK_SLURM_DIR"] = env["MOCK_SLURM_DIR"]
        try:
            return mock_slurm.run_queue()
        finally:
            if old is None:
                os.environ.pop("MOCK_SLURM_DIR", None)
            else:
                os.environ["MOCK_SLURM_DIR"] = old

    def suite(self, n=1) -> Path:
        return self.base / "outputs" / f"SPLICE_V1_0_0_SUITE_{n:04d}"

    def status(self, n=1):
        return json.loads((self.suite(n) / "SUITE_STATUS.json").read_text())

    def gate(self, stage_dir, n=1):
        return json.loads((self.suite(n) / stage_dir / "GATE" / "gate.json").read_text())

    def run_parent(self, stage_dir, task, n=1) -> Path:
        (d,) = (self.suite(n) / stage_dir / "runs").glob(f"*_{task}")
        return d

    def registry(self, n=1):
        rows = (self.suite(n) / "000_MANIFEST" / "run_registry.tsv").read_text().splitlines()[1:]
        return [r.split("\t") for r in rows]

    # ------------------------------------------------------------------ scenarios
    def test_smoke_chain(self):
        rec = self.launch("smoke")
        states = [r["state"] for r in rec.values()]
        self.assertEqual(states.count("CANCELLED"), 1)                      # the finalizer for a failed prepare
        self.assertTrue(all(s in ("COMPLETED", "CANCELLED") for s in states), rec)
        st = self.status()
        self.assertEqual(set(st["gates"].values()), {"PASS"})
        self.assertEqual(st["scientific_status"], "DRY_RUN_NOT_A_MEASUREMENT")
        s = self.suite()
        cfg = json.loads((s / "000_MANIFEST" / "suite_config.json").read_text())
        self.assertEqual(cfg["scale"], "preflight")
        self.assertEqual(cfg["ckpt_root"], str(self.base / "checkpoints" / "SPLICE_V1_0_0_S0001"))
        for f in ("900_REDUCE/MEASURED_REPORT.md", "900_REDUCE/tests.json", "900_REDUCE/table7_measured.csv",
                  "900_REDUCE/paper_target_comparison.csv", "ALL_RESULTS.txt", "RESULTS_ZIP_READY",
                  "000_MANIFEST/code/detector/splice_det/neck.py", "000_MANIFEST/code_identity.txt",
                  "000_MANIFEST/SEMANTIC_PREFLIGHT/PREFLIGHT_REPORT.json", "000_MANIFEST/WHERE_ARE_MY_FILES.txt"):
            self.assertTrue((s / f).exists(), f)
        z = s / "SPLICE_V1_0_0_RESULTS.zip"
        names = zipfile.ZipFile(z).namelist()
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("900_REDUCE/MEASURED_REPORT.md", names)
        self.assertFalse(any(n.endswith((".pt", ".npz")) or n.startswith("000_MANIFEST/code/") for n in names))
        self.assertEqual(os.readlink(self.base / "outputs" / "LATEST_RESULTS.zip"), str(z))
        tests = json.loads((s / "900_REDUCE" / "tests.json").read_text())
        self.assertEqual(tests["9"]["decision"], "NOT_QUALIFIED")           # one seed is not the declared three
        self.assertIn("DRY RUN", (s / "900_REDUCE" / "MEASURED_REPORT.md").read_text())
        done = [r for r in self.registry() if r[6] == "COMPLETED"]
        self.assertEqual(len(done), st["completed_summaries"])
        live = (self.base / "logs" / "SPLICE_V1_0_0_LIVE.out").read_text()
        self.assertIn("[RESULTS_ZIP]", live)

    def test_faults_requeue_continue_retry_and_optional_failure(self):
        faults = ";".join(["train:coco__arm-e__s0:infra",              # GPU fault -> requeue of the task
                           "train:crowdhuman__arm-e__s0:continue:2",   # stops before the time limit twice
                           "train:crowdhuman__arm-a__s0:crash",        # dies without a summary -> gate retry
                           "train:coco__arm-c__s0:fatal"])             # optional run fails for good
        self.launch("smoke", SPLICE_DRY_FAULTS=faults)
        st = self.status()
        self.assertEqual(st["gates"]["train"], "PASS_WITH_OPTIONAL_FAILURES")
        self.assertEqual(st["gates"]["reduce"], "PASS")
        g = self.gate("020_TRAIN")
        self.assertEqual([r["task"] for r in g["optional_failed"]], ["coco__arm-c__s0"])
        self.assertEqual(g["retry_count"], 1)
        self.assertTrue((self.run_parent("020_TRAIN", "coco__arm-e__s0") / "INFRA_REQUEUE_1.json").exists())
        run = self.run_parent("020_TRAIN", "crowdhuman__arm-e__s0")
        self.assertEqual((run / "CONTINUATION_COUNT.txt").read_text().strip(), "2")
        self.assertTrue((run / "RESULT" / "summary.json").exists())
        self.assertTrue((self.suite() / "020_TRAIN" / "GATE" / "retry_1_tasks.tsv").exists())
        status = {r[5]: r[6] for r in self.registry() if r[3] == "train"}
        self.assertEqual(status["coco__arm-c__s0"], "FAILED_SCIENTIFIC")
        # downstream tasks of the failed optional run are prerequisite failures, not retries
        cal = self.gate("030_CALIBRATION")
        self.assertEqual(cal["decision"], "PASS_WITH_OPTIONAL_FAILURES")
        self.assertEqual(cal["optional_failed"][0]["failure_class"], "PREREQUISITE")

    def test_required_failure_then_resume(self):
        self.launch("smoke", SPLICE_DRY_FAULTS="train:crowdhuman__arm-e__s0:fatal")
        st = self.status()
        self.assertEqual(st["gates"]["train"], "FAIL")
        self.assertEqual(st["gates"]["calibrate"], "NOT_RUN")
        self.assertEqual(st["gates"]["reduce"], "PASS")                     # what exists is reduced and packaged
        self.assertEqual(st["scientific_status"], "DRY_RUN_NOT_A_MEASUREMENT")
        tests = json.loads((self.suite() / "900_REDUCE" / "tests.json").read_text())
        self.assertEqual(tests["9"]["decision"], "NOT_EVALUATED")           # missing evidence is not a pass
        self.launch("resume", str(self.suite()))
        st = self.status()
        self.assertEqual(st["gates"]["train"], "PASS")
        self.assertEqual(st["gates"]["evaluate"], "PASS")
        self.assertEqual(st["gates"]["reduce"], "PASS")
        attempts = list((self.run_parent("020_TRAIN", "crowdhuman__arm-e__s0") / "ATTEMPTS").iterdir())
        self.assertEqual(len(attempts), 1)                                  # the failed attempt is kept, not deleted

    def test_single_dataset_skips_the_baseline_stage_and_times_are_clamped(self):
        rec = self.launch("smoke", SPLICE_DATASETS="crowdhuman", SPLICE_ARMS="e", MOCK_PARTITION_LIMIT="00:30:00",
                          SPLICE_TIME_QAT="00:10:00")
        st = self.status()
        self.assertEqual(st["gates"]["baseline"], "SKIPPED")
        self.assertEqual({k: v for k, v in st["gates"].items() if k != "baseline"},
                         {k: "PASS" for k in st["gates"] if k != "baseline"})
        tasks = (self.suite() / "000_MANIFEST" / "tasks_train.tsv").read_text().splitlines()[1:]
        self.assertEqual([t.split("\t")[1] for t in tasks], ["crowdhuman__native__s0", "crowdhuman__arm-e__s0"])
        times = {r["name"]: r["time"] for r in rec.values()}
        self.assertEqual(times["splice_V1_0_0_train"], "0-00:30:00")          # 01:00:00 lowered to the partition's limit
        self.assertEqual(times["splice_V1_0_0_qat"], "00:10:00")              # a request below the limit is kept
        self.assertEqual((self.suite() / "000_MANIFEST" / "partition_time_limit.txt").read_text().strip(), "00:30:00")
        site = (self.suite() / "000_MANIFEST" / "SITE.txt").read_text()
        self.assertIn("account=naiss2025-5-643-gpu", site)
        self.assertIn("partition=gpu", site)

    def test_continuation_and_infrastructure_retry_when_requeue_is_denied(self):
        faults = "train:crowdhuman__arm-e__s0:continue:2;train:crowdhuman__native__s0:infra"
        self.launch("smoke", SPLICE_DATASETS="crowdhuman", SPLICE_ARMS="e", SPLICE_DRY_FAULTS=faults, MOCK_REQUEUE_DENIED="1")
        st = self.status()
        self.assertEqual(st["gates"]["train"], "PASS")
        self.assertEqual(st["gates"]["reduce"], "PASS")
        g = self.suite() / "020_TRAIN" / "GATE"
        self.assertEqual((g / "RETRY_COUNT.txt").read_text().strip(), "1")     # the infrastructure fault spent one retry
        self.assertEqual((g / "CONTINUE_ROUNDS.txt").read_text().strip(), "1")  # the second continuation spent none
        self.assertEqual(len((g / "continue_1_tasks.tsv").read_text().splitlines()), 2)
        run = self.run_parent("020_TRAIN", "crowdhuman__arm-e__s0")
        self.assertEqual((run / "CONTINUATION_COUNT.txt").read_text().strip(), "2")
        self.assertFalse((run / "CONTINUE_PENDING.json").exists())
        self.assertEqual(len(list((run / "ATTEMPTS").iterdir())), 2)            # both stopped chunks are kept

    def test_job_ended_by_sigterm_is_recorded_and_retried(self):
        self.launch("smoke", SPLICE_DATASETS="crowdhuman", SPLICE_ARMS="e", SPLICE_DRY_FAULTS="train:crowdhuman__arm-e__s0:term")
        st = self.status()
        self.assertEqual(st["gates"]["train"], "PASS")
        self.assertEqual((self.suite() / "020_TRAIN" / "GATE" / "RETRY_COUNT.txt").read_text().strip(), "1")
        (old,) = (self.run_parent("020_TRAIN", "crowdhuman__arm-e__s0") / "ATTEMPTS").iterdir()
        rec = json.loads((old / "failure_summary.json").read_text())
        self.assertEqual((rec["failure_class"], rec["retriable"], rec["exit_code"]), ("KILLED_BY_SIGNAL", True, 143))
        self.assertIn("SIGTERM", rec["command"])
        self.assertIn("KILLED_BY_SIGNAL", [r[6] for r in self.registry() if r[5] == "crowdhuman__arm-e__s0"])

    def test_continuation_budget(self):
        self.launch("smoke", SPLICE_DATASETS="crowdhuman", SPLICE_ARMS="e", SPLICE_MAX_CONTINUATIONS="1",
                    SPLICE_DRY_FAULTS="train:crowdhuman__arm-e__s0:continue:3")
        st = self.status()
        self.assertEqual(st["gates"]["train"], "FAIL")
        g = self.gate("020_TRAIN")
        self.assertEqual([r["task"] for r in g["fatal"]], ["crowdhuman__arm-e__s0"])
        self.assertIn("continuation budget exhausted", g["fatal"][0]["message"])
        self.assertTrue((self.run_parent("020_TRAIN", "crowdhuman__arm-e__s0") / "CONTINUE_EXHAUSTED.json").exists())

    def test_postprocess_and_second_suite(self):
        self.launch("smoke", SPLICE_DATASETS="crowdhuman", SPLICE_ARMS="e")
        report = self.suite() / "900_REDUCE" / "MEASURED_REPORT.md"
        report.unlink()
        self.launch("postprocess", str(self.suite()))
        self.assertTrue(report.exists())
        kept = list((self.run_parent("900_REDUCE", "reduce") / "ATTEMPTS").iterdir())
        self.assertEqual(len(kept), 1)                                        # the earlier reduction is kept
        self.assertTrue((self.suite() / "RESULTS_ZIP_READY").exists())
        self.launch("resume", str(self.suite()))                              # nothing is open: reduce and package again
        self.assertEqual(self.status()["gates"]["reduce"], "PASS")
        self.launch("smoke", SPLICE_DATASETS="crowdhuman", SPLICE_ARMS="e")
        self.assertTrue(self.suite(2).is_dir())                              # numbered suites
        self.assertEqual(os.readlink(self.base / "outputs" / "SPLICE_V1_0_0_LATEST_OUTPUT"), str(self.suite(2)))

    def test_failed_preflight_is_advisory_in_smoke_and_stops_a_pilot(self):
        self.launch("smoke", SPLICE_DATASETS="crowdhuman", SPLICE_ARMS="e", SPLICE_DRY_FAULTS="preflight:fail")
        st = self.status()
        self.assertEqual(st["semantic_preflight"], "FAIL")
        self.assertEqual(st["gates"]["reduce"], "PASS")                       # every stage still ran
        self.assertIn("status: FAIL", (self.suite() / "ALL_RESULTS.txt").read_text())
        rec = self.launch("pilot", SPLICE_DATASETS="crowdhuman", SPLICE_ARMS="e", SPLICE_DRY_FAULTS="preflight:fail")
        self.assertEqual(sorted(r["name"] for r in rec.values())[1:],         # the parent job, then only these two
                         ["splice_V1_0_0_pack", "splice_V1_0_0_prep"])
        st = self.status(2)
        self.assertEqual(st["semantic_preflight"], "FAIL")
        self.assertEqual(set(st["gates"].values()), {"NOT_RUN"})              # the pilot stopped in the prepare job
        self.assertFalse((self.suite(2) / "000_MANIFEST" / "PREPARED").exists())
        names = zipfile.ZipFile(self.suite(2) / "SPLICE_V1_0_0_RESULTS.zip").namelist()
        self.assertIn("000_MANIFEST/SEMANTIC_PREFLIGHT/PREFLIGHT_REPORT.json", names)
        self.assertIn("000_MANIFEST/SEMANTIC_PREFLIGHT/test_detector_core.log", names)

    def test_check_mode(self):
        from test_detector_data import Image, make_coco, make_crowdhuman
        if Image is None:
            self.skipTest("Pillow not installed")
        make_coco(self.base / "datasets" / "coco", n_train=12, n_val=2, nested=False)
        make_crowdhuman(self.base / "datasets" / "crowdhuman", n_train=6, n_val=2)
        before = sorted(str(p) for p in self.base.rglob("*"))

        def check(**extra):
            return subprocess.run(["bash", str(self.base / "code" / SHELL), "check"], env=dict(self.env, **extra),
                                  capture_output=True, text=True)
        r = check()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for text in ("[ OK ] account naiss2025-5-643-gpu is among your Slurm associations",
                     "[ OK ] partition gpu exists; wall-time limit 7-00:00:00",
                     "[ OK ] sbatch --test-only accepts the parent job", "[ OK ] sbatch --test-only accepts a training task",
                     "[WARN] COCO train2017: 12 images", "(the full split has 118287)",
                     "[WARN] CrowdHuman val: 2 of 2 annotated images on disk", "== Result: 0 failure(s)"):
            self.assertIn(text, r.stdout)
        self.assertEqual(before, sorted(str(p) for p in self.base.rglob("*")))    # the check writes nothing
        r = check(MOCK_ACCOUNTS="naiss2025-5-643-cpu other-gpu")
        self.assertEqual(r.returncode, 1)
        self.assertIn("[FAIL] account naiss2025-5-643-gpu is not among your Slurm associations", r.stdout)
        r = check(SBATCH_ACCOUNT="naiss2026-1-1", MOCK_SBATCH_REJECT="Invalid account or account/partition combination specified")
        self.assertEqual(r.returncode, 1)
        self.assertIn("[FAIL] the account must end with -gpu", r.stdout)
        self.assertIn("Invalid account or account/partition combination specified", r.stdout)
        shutil.rmtree(self.base / "datasets" / "coco")
        r = check()
        self.assertEqual(r.returncode, 1)
        self.assertIn("[FAIL] COCO 2017 incomplete", r.stdout)

    def test_missing_code_is_reported(self):
        (self.base / "code" / RUNNER).unlink()
        rec = self.launch("smoke")
        self.assertEqual([r["state"] for r in rec.values()], ["FAILED"])
        log = (self.base / "logs" / "SPLICE_V1_0_0_LIVE.err").read_text()
        self.assertIn("Missing SPLICE Python runner", log)


class Snapshot(unittest.TestCase):
    def test_preflight_tests_pass_inside_the_code_snapshot(self):
        """A suite runs from its own copy of the code.  The copy, built from the launcher's
        list of parts, must hold everything the prepare job's tests need."""
        import importlib.util
        import re
        sh = (HPC / SHELL).read_text()
        parts = re.search(r"local parts=\(([^)]*)\)", sh).group(1).split() + ["planning"]
        spec = importlib.util.spec_from_file_location("splice_runner_under_test", HPC / RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as tmp:
            for part in parts:
                self.assertTrue((REPO / part).exists(), part)
                shutil.copytree(REPO / part, Path(tmp) / part, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(tmp) / "tests"), str(Path(tmp) / "detector")]))
            for name in runner.PREFLIGHT_TESTS + runner.PREFLIGHT_TESTS_TORCH:
                r = subprocess.run([sys.executable, "-m", "unittest", f"tests.{name}"], cwd=tmp, env=env,
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, f"{name} inside the snapshot:\n{r.stderr[-1500:]}")


class Identity(unittest.TestCase):
    def test_markers_and_addresses(self):
        sh = (HPC / SHELL).read_text()
        py = (HPC / RUNNER).read_text()
        for line in ('VERSION = "1.0.0-arrhenius"', 'VERSION_TAG = "SPLICE_V1_0_0"',
                     'EXPECTED_RUNNER_BASENAME = "run_splicev_arrhenius.py"'):
            self.assertIn("\n" + line + "\n", py)
        for line in ('SHELL_IMPLEMENTATION_VERSION="1.0.0"', 'SHELL_VERSION_TAG="SPLICE_V1_0_0"',
                     "#SBATCH -A naiss2025-5-643-gpu", "#SBATCH -p gpu",
                     "#SBATCH -o /nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V/logs/SPLICE_V1_0_0_PARENT_%j.out"):
            self.assertIn("\n" + line + "\n", sh)
        self.assertIn('BASE="${SPLICE_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V}"', sh)
        import re                                       # every absolute storage path is inside the SPLICE-V project folder
        paths = {p for p in re.findall(r"/nobackup/proj/disk/[A-Za-z0-9_/-]+", sh)}
        self.assertTrue(paths)
        self.assertTrue(all(p.startswith("/nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V") for p in paths), paths)
        # on Arrhenius, naming -N in a job script allocates whole nodes: the header must not
        self.assertFalse([ln for ln in sh.splitlines() if ln.startswith("#SBATCH") and (" -N" in ln or "--nodes" in ln)])
        r = subprocess.run([sys.executable, str(HPC / RUNNER), "--print_identity"], capture_output=True, text=True)
        ident = json.loads(r.stdout)
        self.assertEqual(ident["actual_runner_basename"], ident["expected_runner_basename"])


if __name__ == "__main__":
    unittest.main()
