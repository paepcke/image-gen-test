 # **********************************************************
 #
 # @Author: Andreas Paepcke
 # @Date:   2026-09-08 13:26:15
 # @File:   /Users/paepcke/VSCodeWorkspaces/image-gen-test/src/image_gen/benchmark_persistent.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2026-09-08 20:08:18
 #
 # **********************************************************

"""
Benchmarks steady-state (weights-already-loaded) LivePortrait latency
using a short, Live-Photo-scale driving clip rather than a full
multi-second video.

Usage (run from <proj-root>, after trim_driving_video.sh has produced
a short clip):

    conda run -n image-gen-test python src/image_gen/benchmark_persistent.py \\
        --source assets/d19.jpg --driving assets/d19_short.mp4 \\
        --gpu 0 --n-runs 5
"""

import argparse
import logging
from pathlib import Path

from image_gen.live_portrait_service import LivePortraitService

log = logging.getLogger("image_gen")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--driving", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("animations_bench"))
    parser.add_argument("--driving-multiplier", type=float, default=1.0,
                         help="Scales expression strength; try 0.3-0.5 for "
                              "a mild version of an apex expression photo.")
    args = parser.parse_args()

    service = LivePortraitService(gpu_index=args.gpu)

    # Separate subfolder per multiplier: otherwise every run (regardless
    # of --driving-multiplier) writes the same filename and silently
    # overwrites the previous run's result, making it impossible to
    # visually compare two multiplier settings afterward.
    output_dir = args.output_dir / f"mult_{args.driving_multiplier}"

    driving = args.driving
    result = service.generate(args.source, driving, output_dir,
                               driving_multiplier=args.driving_multiplier)
    log.info("First call (post weight-load, pre-cache): %.3fs -> %s",
              result["elapsed_seconds"], result["wfp"])

    # If `driving` was a raw video, LivePortrait just cached its motion
    # template next to it. Switch to that for the remaining runs, so
    # they measure true steady-state latency (no re-extraction).
    cached_template = driving.with_suffix(".pkl")
    if cached_template.exists():
        driving = cached_template
        log.info("Using cached template for remaining runs: %s", driving)

    timings = []
    for i in range(args.n_runs):
        result = service.generate(args.source, driving, output_dir,
                                   driving_multiplier=args.driving_multiplier)
        timings.append(result["elapsed_seconds"])

    mean = sum(timings) / len(timings)
    log.info("Steady-state timings (s): %s", [f"{t:.3f}" for t in timings])
    log.info("Mean: %.3fs  Min: %.3fs  Max: %.3fs", mean, min(timings), max(timings))


if __name__ == "__main__":
    main()
