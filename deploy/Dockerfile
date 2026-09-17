# Study server image. MuJoCo needs a GL stack even for offscreen rendering, so
# this uses EGL (headless) rather than GLFW, which is what MUJOCO_GL selects below.
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# libegl/libgl: MuJoCo offscreen rendering. cmake: builds robomimic's egl_probe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl \
        libegl1 libgl1 libglew-dev libosmesa6 libglfw3 \
        libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install in the same order and with the same pins as scripts/setup_env.sh:
# robomimic and lerobot must be --no-deps or they downgrade torch and pull numpy 2.
RUN pip install --upgrade pip && pip install cmake
RUN pip install \
      "torch==2.14.0" "torchvision==0.29.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install \
      "numpy==1.26.4" "h5py==3.16.0" "scipy==1.15.3" "matplotlib==3.10.9" \
      "imageio==2.37.4" "einops==0.8.2" "opencv-python-headless==4.11.0.86" \
      "pygame==2.6.1" pyyaml tqdm termcolor \
      "fastapi==0.141.1" "uvicorn[standard]==0.52.4" "websockets==16.1.1" \
      python-multipart
RUN pip install "robosuite==1.5.2" && pip install "mujoco==3.3.7"
RUN pip install --no-deps "robomimic==0.3.0" && \
    python -m robomimic.scripts.setup_macros || true
RUN pip install --no-deps "lerobot==0.4.4" && \
    pip install --no-deps draccus mergedeep typing_inspect mypy_extensions \
      huggingface_hub safetensors datasets multiprocess pandas pytz tzdata \
      pyarrow dill xxhash httpx httpcore h11 idna requests urllib3 certifi \
      charset-normalizer hf-xet accelerate diffusers deepdiff orderly_set \
      cachebox av future iso8601 pyserial

COPY dagger_lh/ /app/dagger_lh/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

# Fail the build rather than the study if the stack is subtly wrong.
RUN python -c "\
import numpy, torch, robosuite, mujoco; \
from lerobot.policies.act.modeling_act import ACT; \
from robomimic.models.policy_nets import RNNGMMActorNetwork; \
assert numpy.__version__.startswith('1.'), numpy.__version__; \
assert mujoco.__version__.startswith('3.3'), mujoco.__version__; \
print('stack ok', numpy.__version__, torch.__version__, robosuite.__version__)"

# Offscreen rendering must work in the container, not just import.
RUN python -c "\
import os; os.environ.setdefault('MUJOCO_GL','egl'); \
import sys; sys.path.insert(0,'/app'); \
from dagger_lh.config import EnvConfig; \
from dagger_lh.envs import make_env, silence_robosuite_logs; \
silence_robosuite_logs(); \
e=make_env(EnvConfig(horizon=5)); e.reset(); \
img=e.render_operator(height=128,width=128); \
assert img.shape==(128,128,3) and img.std()>0, 'offscreen render produced no image'; \
e.close(); print('egl offscreen render ok')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["python", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8000", \
     "--config", "configs/study_online.yaml", "--max-sessions", "8"]
