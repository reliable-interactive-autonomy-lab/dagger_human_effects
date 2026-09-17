"""robosuite environment factory and observation plumbing.

Keeps two views of the world deliberately separate:
  * the *policy* observation (small, cheap, whatever the network was trained on)
  * the *operator* view (large RGB frames rendered on demand for the pygame HUD)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EnvConfig

# MuJoCo needs a GL backend chosen before first import on macOS/Linux.
os.environ.setdefault("MUJOCO_GL", "glfw")


#: robosuite 1.5 changed the sign convention of the object-relative
#: end-effector sensors created by `ManipulationEnv._get_obj_eef_sensor`:
#: it now returns ``obj_pos - eef_pos``, whereas robosuite <=1.4.1 -- the version
#: robomimic's published datasets were generated with -- stored
#: ``eef_pos - obj_pos``. Nothing errors when this is ignored; a policy
#: pretrained on those datasets simply fails, because a few of the `object`
#: dimensions are negated. See `RobosuiteEnv._object_state`.
OBJ_EEF_REL_SUFFIXES = ("_pos",)


def _is_obj_eef_relative(name: str) -> bool:
    """True for object-modality sensors of the form `<a>_to_<b>_pos`."""
    return "_to_" in name and name.endswith(OBJ_EEF_REL_SUFFIXES)


def _lazy_robosuite():
    import robosuite as suite  # noqa: WPS433 (deferred: heavy import)
    return suite


class RobosuiteEnv:
    """Thin wrapper giving robomimic-shaped observations plus operator rendering."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        suite = _lazy_robosuite()

        cameras = list(dict.fromkeys(cfg.camera_names + [cfg.render_camera]))
        controller_configs = suite.load_composite_controller_config(
            controller=cfg.controller, robot=cfg.robots
        )
        self._env_kwargs = dict(
            env_name=cfg.env_name,
            robots=cfg.robots,
            controller_configs=controller_configs,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=cfg.use_image_obs,
            use_object_obs=True,
            camera_names=cameras,
            camera_heights=cfg.camera_height,
            camera_widths=cfg.camera_width,
            control_freq=cfg.control_freq,
            horizon=cfg.horizon,
            reward_shaping=cfg.reward_shaping,
            ignore_done=True,
            hard_reset=False,
        )
        self.env = suite.make(**self._env_kwargs)
        self._last_raw_obs: Dict[str, np.ndarray] = {}
        self._last_obs: Dict[str, np.ndarray] = {}
        self._flipped_sensors: set = set()
        self.t = 0

    # ------------------------------------------------------------------ spaces
    @property
    def action_dim(self) -> int:
        return int(self.env.action_dim)

    @property
    def obs_keys(self) -> List[str]:
        return list(self.cfg.low_dim_keys) + list(self.cfg.image_keys)

    def obs_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Shapes in robomimic convention (images as C,H,W)."""
        if not self._last_obs:
            self.reset()
        shapes: Dict[str, Tuple[int, ...]] = {}
        for k in self.cfg.low_dim_keys:
            shapes[k] = (int(np.asarray(self._last_obs[k]).size),)
        for k in self.cfg.image_keys:
            shapes[k] = (3, self.cfg.camera_height, self.cfg.camera_width)
        return shapes

    # ------------------------------------------------------------------ stepping
    def reset(self) -> Dict[str, np.ndarray]:
        raw = self.env.reset()
        self._last_raw_obs = raw
        self.t = 0
        return self._extract(raw)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        raw, reward, _, info = self.env.step(action)
        self._last_raw_obs = raw
        self.t += 1
        success = bool(self.env._check_success())
        horizon_hit = self.t >= int(self.cfg.horizon)
        info = dict(info or {})
        info.update(success=success, horizon=horizon_hit, t=self.t)
        return self._extract(raw), float(reward), bool(success or horizon_hit), info

    def _object_state(self) -> np.ndarray:
        """`object-state` rebuilt in robomimic-dataset (robosuite 1.4.1) convention.

        Concatenates the object-modality observables in robosuite's own insertion
        order -- matching how robosuite builds `object-state` -- while negating the
        object-to-end-effector position sensors whose sign flipped in robosuite 1.5.
        Set `EnvConfig.legacy_object_state = False` to keep robosuite 1.5 signs.
        """
        parts: List[np.ndarray] = []
        for name, obs in self.env._observables.items():
            if not (obs.is_enabled() and obs.is_active()):
                continue
            if obs.modality != "object":
                continue
            val = np.atleast_1d(np.asarray(obs.obs, dtype=np.float64)).copy()
            if self.cfg.legacy_object_state and _is_obj_eef_relative(name):
                val = -val
                self._flipped_sensors.add(name)
            parts.append(val)
        if not parts:
            raise KeyError(f"{self.cfg.env_name} exposes no object-modality sensors")
        return np.concatenate(parts, axis=-1)

    @property
    def flipped_sensors(self) -> List[str]:
        """Which object sensors were sign-corrected (for logging / auditing)."""
        return sorted(self._flipped_sensors)

    def _extract(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Select the configured keys. Images stay uint8 HWC (robomimic on-disk form)."""
        alias = dict(self.cfg.obs_key_aliases or {})
        out: Dict[str, np.ndarray] = {}
        for k in self.cfg.low_dim_keys:
            if alias.get(k) == "object-state" or k == "object-state":
                out[k] = self._object_state().astype(np.float32)
                continue
            src = k if k in raw else alias.get(k, k)
            if src not in raw:
                raise KeyError(
                    f"observation key '{k}' not produced by {self.cfg.env_name} "
                    f"(also tried alias '{alias.get(k)}'); available: {sorted(raw.keys())}"
                )
            out[k] = np.asarray(raw[src], dtype=np.float32).ravel()
        for k in self.cfg.image_keys:
            src = k if k in raw else alias.get(k, k)
            if src not in raw:
                raise KeyError(f"image key '{k}' not in obs; available: {sorted(raw.keys())}")
            out[k] = np.asarray(raw[src], dtype=np.uint8)
        self._last_obs = out
        return out

    # ------------------------------------------------------------------ rendering
    def render_operator(self, camera: Optional[str] = None,
                        height: Optional[int] = None,
                        width: Optional[int] = None) -> np.ndarray:
        """High-resolution frame for the human operator (HWC uint8, upright)."""
        cam = camera or self.cfg.render_camera
        h = height or self.cfg.render_height
        w = width or self.cfg.render_width
        img = self.env.sim.render(width=w, height=h, camera_name=cam)
        return np.asarray(img[::-1], dtype=np.uint8)  # MuJoCo renders bottom-up

    # ------------------------------------------------------------------ state I/O
    def get_sim_state(self) -> np.ndarray:
        return np.asarray(self.env.sim.get_state().flatten(), dtype=np.float64)

    def get_model_xml(self) -> str:
        return self.env.sim.model.get_xml()

    def reset_to_sim_state(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        self.env.reset()
        self.env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
        self.env.sim.forward()
        # The controller caches a goal derived from the pose at reset time; after
        # overwriting sim state that goal is stale and the first few actions are
        # applied relative to the wrong reference.
        for robot in getattr(self.env, "robots", []):
            for attr in ("composite_controller", "controller"):
                ctrl = getattr(robot, attr, None)
                reset_goal = getattr(ctrl, "reset_goal", None) if ctrl else None
                if callable(reset_goal):
                    try:
                        reset_goal()
                    except Exception:
                        pass
        # force_update is required after writing sim state directly: without it
        # robosuite returns observables cached from the last actual step.
        raw = self.env._get_observations(force_update=True)
        self._last_raw_obs = raw
        self.t = 0
        return self._extract(raw)

    def env_meta(self) -> Dict[str, Any]:
        """robomimic-compatible `env_args` blob written into demo HDF5 files."""
        kwargs = {k: v for k, v in self._env_kwargs.items() if k != "env_name"}
        return {
            "env_name": self.cfg.env_name,
            "type": 1,  # robomimic EnvType.ROBOSUITE_TYPE
            "env_kwargs": json.loads(json.dumps(kwargs, default=str)),
        }

    def eef_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        raw = self._last_raw_obs
        return (np.asarray(raw.get("robot0_eef_pos", np.zeros(3))),
                np.asarray(raw.get("robot0_eef_quat", np.array([0, 0, 0, 1.0]))))

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def make_env(cfg: EnvConfig) -> RobosuiteEnv:
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    return RobosuiteEnv(cfg)


def silence_robosuite_logs() -> None:
    """robosuite is chatty; the HUD is the operator's channel, not stdout."""
    os.environ.setdefault("ROBOSUITE_COLOR_LOGS", "0")
    try:
        import robosuite.utils.log_utils as lu
        lu.ROBOSUITE_DEFAULT_LOGGER.setLevel("ERROR")
    except Exception:
        pass
