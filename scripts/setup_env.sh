#!/usr/bin/env bash
# Create/repair the conda environment for this pipeline.
#
#   bash scripts/setup_env.sh [env_name]
#
# Three version constraints are load-bearing and are the reason this script
# exists rather than a plain `pip install -r requirements.txt`:
#
#   1. mujoco must be >=3.3.0 (robosuite 1.5.2 requires it) but NOT >=3.4:
#      mujoco 3.12 makes robosuite's get_joint_qpos_addr assert on the Panda
#      model. 3.3.7 is the tested version.
#
#   2. numpy must stay on 1.x. robosuite and robomimic both have code paths that
#      assume numpy 1 semantics.
#
#   3. robomimic and lerobot must be installed with --no-deps. Their dependency
#      pins would otherwise downgrade torch and pull numpy 2.x, breaking (1) and
#      (2). Their remaining runtime imports are installed explicitly below.
set -euo pipefail

ENV_NAME="${1:-dagger_learned_helplessness}"
echo "==> setting up conda env: $ENV_NAME"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
if ! python -c "import sys; assert sys.version_info[:2]==(3,10)" 2>/dev/null; then
  conda install -y -n "$ENV_NAME" python=3.10 pip
fi

PY="$(conda run -n "$ENV_NAME" which python)"
echo "==> python: $PY"

"$PY" -m pip install --upgrade pip
# cmake is needed to build egl_probe (a robomimic dependency) from source.
"$PY" -m pip install cmake

echo "==> core"
"$PY" -m pip install "torch==2.14.0" "torchvision==0.29.0" "numpy==1.26.4" \
  "h5py==3.16.0" "scipy==1.15.3" "pygame==2.6.1" "matplotlib==3.10.9" \
  "imageio==2.37.4" "imageio-ffmpeg==0.6.0" "einops==0.8.2" \
  pyyaml tqdm termcolor

echo "==> robosuite (pin mujoco after, robosuite pulls a newer one)"
"$PY" -m pip install "robosuite==1.5.2"
"$PY" -m pip install "mujoco==3.3.7"

echo "==> robomimic (reference BC-RNN)"
"$PY" -m pip install --no-deps "robomimic==0.3.0"
"$PY" -m robomimic.scripts.setup_macros || true

echo "==> lerobot (reference ACT) + its runtime imports, without its pins"
"$PY" -m pip install --no-deps "lerobot==0.4.4"
"$PY" -m pip install --no-deps draccus mergedeep typing_inspect mypy_extensions \
  huggingface_hub safetensors datasets multiprocess pandas pytz tzdata pyarrow \
  dill xxhash httpx httpcore h11 idna requests urllib3 certifi \
  charset-normalizer hf-xet accelerate diffusers deepdiff orderly_set \
  cachebox av future iso8601 pyserial

echo "==> verifying"
"$PY" - <<'PYCHECK'
import numpy, torch, robosuite, robomimic, mujoco, pygame, h5py
from lerobot.policies.act.modeling_act import ACT
from robomimic.models.policy_nets import RNNGMMActorNetwork
print(f"  numpy      {numpy.__version__}   (must be 1.x)")
print(f"  torch      {torch.__version__}  mps={torch.backends.mps.is_available()}")
print(f"  robosuite  {robosuite.__version__}")
print(f"  robomimic  {robomimic.__version__}")
print(f"  mujoco     {mujoco.__version__}   (must be 3.3.x)")
print(f"  pygame     {pygame.version.ver}")
assert numpy.__version__.startswith("1."), "numpy 2.x will break robosuite/robomimic"
assert mujoco.__version__.startswith("3.3"), "mujoco must be 3.3.x"
print("  reference nets import OK (robomimic BC-RNN, lerobot ACT)")
PYCHECK

echo
echo "==> environment ready. Next:"
echo "    conda activate $ENV_NAME"
echo "    python scripts/download_data.sh          # official robomimic datasets"
echo "    python scripts/smoke_test.py             # headless end-to-end check"
