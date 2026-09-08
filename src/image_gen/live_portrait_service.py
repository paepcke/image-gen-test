"""
Persistent-process LivePortrait service.

Loads model weights exactly once at construction, then serves repeated
generate() calls against that already-loaded state — the number that
actually matters for a deployed service, unlike shelling out to
inference.py per call (which reloads all five weight files every time;
see the earlier benchmark_liveportrait.py run, where "cold" and "warm"
came back nearly identical because each was really a cold run).

Lives at <proj-root>/src/image_gen/live_portrait_service.py. Not part
of the therapist_trainer repo. Reuses LivePortrait's own ArgumentConfig,
InferenceConfig, CropConfig, LivePortraitPipeline, and partial_fields
helper directly from the cloned repo, rather than re-implementing them,
to avoid drifting out of sync with whatever version is checked out
under third_party/LivePortrait.
"""

import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("image_gen")

PROJ_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJ_ROOT / "third_party" / "LivePortrait"


class LivePortraitService:
    """Loads LivePortrait weights once; serves many generate() calls.

    Construct exactly one instance per process, and construct it before
    any other code in this process imports torch or onnxruntime:
    CUDA_VISIBLE_DEVICES is set in __init__, before those libraries are
    first imported, so the pinned GPU is the only one this process's
    CUDA context ever sees. In production this maps to one persistent
    process per concurrent trainee session, each pinned to one of
    quatro's three GPUs.

    :param gpu_index: Physical GPU index (0, 1, or 2 on quatro) this
        service instance is pinned to.
    """

    def __init__(self, gpu_index: int):
        self.gpu_index = gpu_index
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        # Imported here (not at module top-level) so CUDA_VISIBLE_DEVICES
        # above is guaranteed set before torch/onnxruntime initialize.
        import inference as lp_inference  # LivePortrait's own inference.py
        from src.config.argument_config import ArgumentConfig
        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_pipeline import LivePortraitPipeline

        self._ArgumentConfig = ArgumentConfig
        default_args = ArgumentConfig()
        inference_cfg = lp_inference.partial_fields(InferenceConfig, default_args.__dict__)
        crop_cfg = lp_inference.partial_fields(CropConfig, default_args.__dict__)

        log.info("Loading LivePortrait weights on GPU %d ...", gpu_index)
        start = time.perf_counter()
        self.pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        self.load_seconds = time.perf_counter() - start
        log.info("Weights loaded in %.2fs", self.load_seconds)

    def generate(self, source: Path, driving: Path, output_dir: Path,
                 driving_multiplier: float = 1.0) -> dict:
        """Runs one animation against already-loaded weights.

        Works identically whether `driving` is a video, a single
        driving image (e.g. a canonical Ekman-style expression photo —
        pose is NOT transferred in this case, only expression: with a
        single-frame driving input, LivePortrait's frame-0 pose is used
        as its own baseline, so the source's original head pose is
        preserved exactly), or a cached .pkl motion template.

        :param source: Reference photo (torso + face) to animate.
        :param driving: A driving video, a single driving image, or a
            cached .pkl motion template from a prior call against the
            same driving input — pass the .pkl on every call after the
            first to skip re-extracting motion (LivePortrait caches it
            automatically next to the driving file after its first use,
            for images as well as videos).
        :param output_dir: Directory LivePortrait writes results into.
        :param driving_multiplier: Scales the transferred expression
            delta (LivePortrait's own `driving_multiplier` argument,
            default 1.0 = full strength). Canonical expression photos
            (e.g. Ekman-style apex expressions) are usually far more
            intense than what "subtly reflecting a statement" calls
            for — try 0.3-0.5 as a starting point for a much milder
            version of the same expression.
        :return: Dict with 'elapsed_seconds', 'wfp' (animated output),
            and 'wfp_concat' (side-by-side comparison video).
        """
        args = self._ArgumentConfig(
            source=str(source), driving=str(driving), output_dir=str(output_dir),
            driving_multiplier=driving_multiplier,
        )
        start = time.perf_counter()
        wfp, wfp_concat = self.pipeline.execute(args)
        elapsed = time.perf_counter() - start
        log.info("generate() elapsed: %.2fs", elapsed)
        return {"elapsed_seconds": elapsed, "wfp": wfp, "wfp_concat": wfp_concat}
