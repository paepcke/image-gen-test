"""
Standalone timing harness for LivePortrait on quatro.

Lives at <proj-root>/src/image_gen/benchmark_liveportrait.py, where
<proj-root> = /home/paepcke/VSCodeWorkspaces/image-gen-test. Not part of
the therapist_trainer repo. Run via `conda run -n exprtest`, e.g.:

    conda run -n exprtest python src/image_gen/benchmark_liveportrait.py \\
        --source assets/ref_client.jpg \\
        --driving assets/driving_expressions.mp4 \\
        --gpu 0
"""

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("image_gen")

PROJ_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ROOT = PROJ_ROOT / "third_party" / "LivePortrait"


class ExpressionBenchmark:
    """Times LivePortrait single-image-animation inference on one GPU.

    :param source: Path to the reference photo (face + torso, the
        identity/background to preserve across all generated frames).
    :param driving: Path to the driving video whose expression/pose
        sequence will be transferred onto ``source``.
    :param gpu_index: Which of quatro's three 2080 Tis to pin this run
        to, via ``CUDA_VISIBLE_DEVICES``.
    :param repo_root: Path to the cloned LivePortrait repo (defaults to
        ``<proj-root>/third_party/LivePortrait``, as set up by
        ``setup_env.sh``).
    """

    def __init__(self, source: Path, driving: Path, gpu_index: int,
                 repo_root: Path = DEFAULT_REPO_ROOT):
        self.source = Path(source)
        self.driving = Path(driving)
        self.gpu_index = gpu_index
        self.repo_root = Path(repo_root)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Fails fast with a clear message if inputs are missing.

        :raises FileNotFoundError: if source, driving, or repo path is
            absent.
        """
        if not self.repo_root.exists():
            raise FileNotFoundError(
                f"LivePortrait repo not found at {self.repo_root}. "
                f"Run setup_env.sh first."
            )
        for p in (self.source, self.driving):
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected input not found: {p}. Drop a reference "
                    f"photo and driving video into assets/ before running."
                )

    def run_once(self) -> float:
        """Runs one LivePortrait inference pass and returns wall-clock seconds.

        :return: Elapsed seconds for the inference subprocess only
            (model load + generation), not counting environment setup.
        """
        cmd = [
            "python", "inference.py",
            "-s", str(self.source.resolve()),
            "-d", str(self.driving.resolve()),
        ]
        run_env = dict(os.environ)
        run_env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        log.info("Running on GPU %d: %s", self.gpu_index, " ".join(cmd))

        start = time.perf_counter()
        subprocess.run(cmd, cwd=self.repo_root, env=run_env, check=True)
        elapsed = time.perf_counter() - start
        log.info("Elapsed: %.2fs", elapsed)
        return elapsed

    def run_warm_and_cold(self, n_warm_runs: int = 2) -> dict:
        """Runs one cold pass (model load included) then warm passes.

        Model-load time is a one-time cost per session in a real
        deployment, so both numbers matter: cold tells you worst-case
        first-request latency, warm tells you steady-state per-turn
        latency.

        :param n_warm_runs: Number of additional timed runs after the
            first (cold) one.
        :return: Dict with 'cold_seconds' and 'warm_seconds' (list).
        """
        cold = self.run_once()
        warm = [self.run_once() for _ in range(n_warm_runs)]
        return {"cold_seconds": cold, "warm_seconds": warm}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--driving", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--warm-runs", type=int, default=2)
    args = parser.parse_args()

    bench = ExpressionBenchmark(
        source=args.source, driving=args.driving, gpu_index=args.gpu,
    )
    results = bench.run_warm_and_cold(n_warm_runs=args.warm_runs)
    log.info("Results: %s", results)


if __name__ == "__main__":
    main()
    
