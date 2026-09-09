#!/usr/bin/env bash
# Standalone LivePortrait POC setup — run on quatro, paepcke account, OUTSIDE the therapist repo.
# <proj-root> = /home/paepcke/VSCodeWorkspaces/image-gen-test
# Usage: bash src/image_gen/setup_env.sh
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="image-gen-test"
DEPS_DIR="${PROJ_ROOT}/third_party"

mkdir -p "${DEPS_DIR}"

# --- conda env ---------------------------------------------------------
# `pip` is listed explicitly: conda's python package does not reliably
# pull in pip as a dependency on its own, and when it's missing from the
# env's bin/, PATH lookups for a bare `pip` fall through to whatever
# pip is next on PATH (e.g. ~/.local/bin/pip) — which is typically
# wired to the *system* Python, triggering Debian's PEP 668
# "externally-managed-environment" error. Listing it here guarantees
# envs/${ENV_NAME}/bin/pip exists.
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

# IMPORTANT: use `conda run -n <env>` for every step below, NOT
# `conda activate` + bare commands (activate's shell-function behavior
# isn't reliable in non-interactive scripts). And within that, install
# via `python -m pip`, NOT bare `pip` — `python -m pip` always resolves
# relative to whichever `python` conda run selected, so it can't
# silently fall through to a shadowing pip elsewhere on PATH the way a
# bare `pip` lookup can.

echo "Sanity check — this MUST show a path under the ${ENV_NAME} env, not /usr/bin:"
conda run -n "${ENV_NAME}" which python
conda run -n "${ENV_NAME}" python -m pip --version

# Torch build for CUDA 12.6 (matches quatro's driver 560.35.03). Turing
# (2080 Ti, compute capability 7.5) runs this fine; the only thing
# Turing lacks is FlashAttention-2, which LivePortrait doesn't require.
conda run -n "${ENV_NAME}" python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# --- LivePortrait --------------------------------------------------------
cd "${DEPS_DIR}"
if [ ! -d LivePortrait ]; then
    git clone https://github.com/KwaiVGI/LivePortrait.git
fi
cd LivePortrait
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt

# LivePortrait's inference.py shells out to ffmpeg/ffprobe directly;
# installed into the env (not system-wide) to avoid needing sudo on
# quatro and to keep this POC self-contained.
conda install -n "${ENV_NAME}" -y -c conda-forge ffmpeg

# Pretrained weights
conda run -n "${ENV_NAME}" python -m huggingface_hub.commands.huggingface_cli \
    download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*"

echo "Setup done. Env: ${ENV_NAME}. Repo: ${DEPS_DIR}/LivePortrait"

# Editable install of our own packages (common, cloud_img_procurement,
# image_gen) per pyproject.toml -- lets scripts import them by package
# name (e.g. `from cloud_img_procurement.bucket_enums import Race`)
# instead of each script hand-inserting <proj-root>/src into sys.path.
conda run -n "${ENV_NAME}" python -m pip install -e "${PROJ_ROOT}"

echo "Next: drop a reference photo (torso+face) and a driving video into"
echo "${PROJ_ROOT}/assets/, then run src/image_gen/benchmark_liveportrait.py"
