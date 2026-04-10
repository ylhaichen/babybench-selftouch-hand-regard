import csv
import importlib.util
import math
import os
from collections import defaultdict, deque

os.environ.setdefault("MPLCONFIGDIR", "/tmp/babybench_mplconfig")
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import mujoco
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import babybench.eval as bb_eval
import babybench.utils as bb_utils


DEFAULT_BASE_TRAIN_FOR = 5_000_000
DEFAULT_DIFFICULT_TRAIN_FOR = 2_000_000
DEFAULT_AFTER_TRAIN_FOR = 2_000_000


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def default_save_dir(stage):
    defaults = {
        "base": "results/intrinsic_motivation_stelios_x_giannis_smooth",
        "difficult": "results/resume_random_init",
        "after": "results/intrinsic_motivation_stelios_x_giannis_after_difficult_task",
    }
    return defaults[stage]


def default_train_for(stage):
    defaults = {
        "base": DEFAULT_BASE_TRAIN_FOR,
        "difficult": DEFAULT_DIFFICULT_TRAIN_FOR,
        "after": DEFAULT_AFTER_TRAIN_FOR,
    }
    return defaults[stage]


def stage_log_std_schedule(stage, train_for):
    if stage == "base":
        return 0.5, -1.0, int(train_for)
    if stage == "difficult":
        return 0.0, -1.1, int(train_for)
    if stage == "after":
        return -0.511, -1.0, int(0.6 * 5_000_000)
    raise ValueError(f"Unsupported stage: {stage}")


def maybe_tensorboard_log_dir(save_dir):
    if importlib.util.find_spec("tensorboard") is None:
        return None
    return os.path.join(save_dir, "tb_logs")


def resolve_device(device="auto"):
    requested = str(device).strip().lower()
    if requested == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if requested == "cuda" and not th.cuda.is_available():
        raise RuntimeError("Requested device='cuda' but torch.cuda.is_available() is False")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    return requested


def get_unwrapped_env(env):
    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    return unwrapped


def find_env_attr(env, attr_name, default=None):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, attr_name):
            return getattr(current, attr_name)
        current = getattr(current, "env", None)
    return default


def flatten_touch_to_sensors(touch_array):
    arr = np.asarray(touch_array)
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return np.linalg.norm(arr, axis=1)
    return np.abs(arr).reshape(-1)


class Float32ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self._to_float32_space(env.observation_space)

    def _to_float32_space(self, space):
        if isinstance(space, spaces.Box) and np.issubdtype(space.dtype, np.floating):
            return spaces.Box(
                low=space.low.astype(np.float32),
                high=space.high.astype(np.float32),
                shape=space.shape,
                dtype=np.float32,
            )
        if isinstance(space, spaces.Dict):
            return spaces.Dict({k: self._to_float32_space(v) for k, v in space.spaces.items()})
        return space

    def observation(self, obs):
        if isinstance(obs, dict):
            return {
                k: (
                    np.asarray(v, dtype=np.float32)
                    if hasattr(v, "dtype") and np.issubdtype(np.asarray(v).dtype, np.number)
                    else v
                )
                for k, v in obs.items()
            }
        array = np.asarray(obs)
        return array.astype(np.float32) if np.issubdtype(array.dtype, np.number) else obs


class BodyMapBuilder:
    def __init__(self, env):
        self.env = env
        self.unwrapped = get_unwrapped_env(env)
        self.model = getattr(self.unwrapped, "model", None)
        self.data = getattr(self.unwrapped, "data", None)

        self.sensor_to_body_part = {}
        self.body_part_names = {}
        self.sensor_xyz = {}

        self.left_hand_bodies = set()
        self.right_hand_bodies = set()
        self.body_bodies = set()

        self.left_hand_geoms = set()
        self.right_hand_geoms = set()
        self.body_geoms = set()

        self._build_sensor_map()
        self._build_geom_sets()

    def _build_sensor_map(self):
        touch_obj = getattr(self.unwrapped, "touch", None)
        mapped = False
        if touch_obj is not None:
            sensor_positions = getattr(touch_obj, "sensor_positions", None)
            if isinstance(sensor_positions, dict) and self.model is not None:
                idx = 0
                for body_id, positions in sensor_positions.items():
                    n_positions = len(positions)
                    for i in range(n_positions):
                        self.sensor_to_body_part[idx] = int(body_id)
                        try:
                            point = np.asarray(positions[i], dtype=np.float32)
                            if point.shape == (3,):
                                self.sensor_xyz[idx] = point
                        except Exception:
                            pass
                        idx += 1
                    try:
                        name = self.model.body(body_id).name
                    except Exception:
                        name = f"body_{body_id}"
                    self.body_part_names[int(body_id)] = name
                mapped = idx > 0

        if mapped:
            return

        touch_dim = 0
        try:
            if isinstance(self.env.observation_space, spaces.Dict) and "touch" in self.env.observation_space.spaces:
                touch_dim = int(np.prod(self.env.observation_space.spaces["touch"].shape))
        except Exception:
            touch_dim = 0

        if touch_dim <= 0:
            return

        n_sensors = touch_dim // 3 if touch_dim % 3 == 0 else touch_dim
        sensors_per_part = max(1, math.ceil(n_sensors / 68))
        for i in range(n_sensors):
            body_part_id = min(i // sensors_per_part, 67)
            self.sensor_to_body_part[i] = body_part_id
            self.body_part_names[body_part_id] = f"part_{body_part_id}"

    def _is_left_hand_name(self, name):
        name = (name or "").lower()
        return any(
            token in name
            for token in (
                "left_hand",
                "left_fingers",
                "left_ff",
                "left_mf",
                "left_rf",
                "left_lf",
                "left_th",
                "left_thumb",
                "left_palm",
                "leftfinger",
                "left_finger",
                "lhand",
                "lfinger",
                "lthumb",
                "lpalm",
            )
        )

    def _is_right_hand_name(self, name):
        name = (name or "").lower()
        return any(
            token in name
            for token in (
                "right_hand",
                "right_fingers",
                "right_ff",
                "right_mf",
                "right_rf",
                "right_lf",
                "right_th",
                "right_thumb",
                "right_palm",
                "rightfinger",
                "right_finger",
                "rhand",
                "rfinger",
                "rthumb",
                "rpalm",
            )
        )

    def _is_ignored_geom(self, name):
        name = (name or "").lower()
        return any(token in name for token in ("floor", "ground", "plane", "table", "crib", "support", "wall"))

    def _build_geom_sets(self):
        if self.model is None:
            return

        for body_id in range(self.model.nbody):
            try:
                body_name = self.model.body(body_id).name
            except Exception:
                body_name = f"body_{body_id}"

            if self._is_left_hand_name(body_name):
                self.left_hand_bodies.add(body_id)
            elif self._is_right_hand_name(body_name):
                self.right_hand_bodies.add(body_id)
            else:
                self.body_bodies.add(body_id)

        for geom_id in range(self.model.ngeom):
            try:
                geom_name = self.model.geom(geom_id).name
                body_id = int(self.model.geom_bodyid[geom_id])
            except Exception:
                geom_name = f"geom_{geom_id}"
                body_id = -1

            if self._is_ignored_geom(geom_name):
                continue

            if body_id in self.left_hand_bodies:
                self.left_hand_geoms.add(geom_id)
            elif body_id in self.right_hand_bodies:
                self.right_hand_geoms.add(geom_id)
            elif body_id in self.body_bodies:
                self.body_geoms.add(geom_id)

    def sensors_to_body_parts(self, active_sensor_ids):
        parts = set()
        for sensor_id in active_sensor_ids:
            body_part = self.sensor_to_body_part.get(int(sensor_id))
            if body_part is not None:
                parts.add(body_part)
        return parts


class IntrinsicBodyExploration:
    def __init__(self, body_map):
        self.map = body_map

        self.global_parts = set()
        self.global_left_geoms = set()
        self.global_right_geoms = set()

        self.r_new_global_part = 0.06
        self.r_new_episode_part = 0.03
        self.r_diminish_base = 0.02
        self.r_new_global_geom = 0.05
        self.r_new_episode_geom = 0.03
        self.r_balance_bonus = 0.05
        self.balance_tolerance = 2
        self.milestones = [5, 10, 15, 20, 25, 30]
        self.milestone_bonus = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]

        self.touch_threshold = 1e-6
        self.max_step_reward = 1.0

        self.voxel_size = 0.02
        self.r_new_voxel = 0.002
        self.r_new_episode_voxel = 0.001
        self.global_voxels = set()

        self.reset_episode()

    def reset_episode(self):
        self.ep_parts = set()
        self.ep_part_counts = defaultdict(int)
        self.ep_left_geoms = set()
        self.ep_right_geoms = set()
        self.reached_milestones = set()
        self.balance_awarded = False
        self.ep_voxels = set()

    def _active_sensors(self, touch_obs):
        mags = flatten_touch_to_sensors(touch_obs)
        active = np.where(mags > self.touch_threshold)[0]
        return active.tolist()

    def _contacts_by_hand(self):
        left_hit = set()
        right_hit = set()

        data = getattr(self.map, "data", None)
        model = getattr(self.map, "model", None)
        if data is None or model is None:
            return left_hit, right_hit

        n_contacts = int(getattr(data, "ncon", 0))
        for i in range(n_contacts):
            contact = data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if geom1 in self.map.left_hand_geoms and geom2 in self.map.body_geoms:
                left_hit.add(geom2)
            elif geom2 in self.map.left_hand_geoms and geom1 in self.map.body_geoms:
                left_hit.add(geom1)

            if geom1 in self.map.right_hand_geoms and geom2 in self.map.body_geoms:
                right_hit.add(geom2)
            elif geom2 in self.map.right_hand_geoms and geom1 in self.map.body_geoms:
                right_hit.add(geom1)

        return left_hit, right_hit

    def step_reward(self, obs_dict):
        reward = 0.0

        touch_obs = obs_dict.get("touch")
        if touch_obs is not None:
            active_sensors = self._active_sensors(touch_obs)
            active_parts = self.map.sensors_to_body_parts(active_sensors)

            for part in active_parts:
                if part not in self.global_parts:
                    self.global_parts.add(part)
                    reward += self.r_new_global_part

                if part not in self.ep_parts:
                    self.ep_parts.add(part)
                    reward += self.r_new_episode_part
                else:
                    self.ep_part_counts[part] += 1
                    count = self.ep_part_counts[part]
                    if count <= 8:
                        reward += self.r_diminish_base / float(count)

            for sensor_id in active_sensors:
                point = self.map.sensor_xyz.get(int(sensor_id))
                if point is None:
                    continue
                ix, iy, iz = (point / self.voxel_size).astype(np.int32).tolist()
                key = (ix, iy, iz)
                if key not in self.global_voxels:
                    self.global_voxels.add(key)
                    reward += self.r_new_voxel
                if key not in self.ep_voxels:
                    self.ep_voxels.add(key)
                    reward += self.r_new_episode_voxel

        left_hit, right_hit = self._contacts_by_hand()

        for geom_id in left_hit:
            if geom_id not in self.global_left_geoms:
                self.global_left_geoms.add(geom_id)
                reward += self.r_new_global_geom
            if geom_id not in self.ep_left_geoms:
                self.ep_left_geoms.add(geom_id)
                reward += self.r_new_episode_geom

        for geom_id in right_hit:
            if geom_id not in self.global_right_geoms:
                self.global_right_geoms.add(geom_id)
                reward += self.r_new_global_geom
            if geom_id not in self.ep_right_geoms:
                self.ep_right_geoms.add(geom_id)
                reward += self.r_new_episode_geom

        ep_coverage = len(self.ep_parts)
        for idx, threshold in enumerate(self.milestones):
            if ep_coverage >= threshold and idx not in self.reached_milestones:
                reward += self.milestone_bonus[idx]
                self.reached_milestones.add(idx)

        if (not self.balance_awarded) and abs(len(self.global_left_geoms) - len(self.global_right_geoms)) <= self.balance_tolerance:
            reward += self.r_balance_bonus
            self.balance_awarded = True

        return float(np.clip(reward, 0.0, self.max_step_reward))

    def info(self):
        return {
            "global_parts": len(self.global_parts),
            "ep_parts": len(self.ep_parts),
            "global_left_geoms": len(self.global_left_geoms),
            "global_right_geoms": len(self.global_right_geoms),
            "ep_left_geoms": len(self.ep_left_geoms),
            "ep_right_geoms": len(self.ep_right_geoms),
        }


class IntrinsicSelfTouchWrapper(gym.Wrapper):
    def __init__(self, env, enable_speed_penalty=False):
        super().__init__(env)
        self.body_map = BodyMapBuilder(env)
        self.intrinsic = IntrinsicBodyExploration(self.body_map)
        self.enable_speed_penalty = bool(enable_speed_penalty)

        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_idx = 0
        self.recent_intrinsic = deque(maxlen=100)
        self.last_proprio = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.intrinsic.reset_episode()
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_idx += 1
        self.last_proprio = obs.get("observation")
        return obs, info

    def _speed_penalty(self, obs):
        if not self.enable_speed_penalty:
            self.last_proprio = obs.get("observation")
            return 0.0

        current_proprio = obs.get("observation")
        if self.last_proprio is None or current_proprio is None:
            self.last_proprio = current_proprio
            return 0.0

        delta = np.abs(np.asarray(current_proprio) - np.asarray(self.last_proprio))
        speed = float(np.mean(delta))
        self.last_proprio = current_proprio
        speed_threshold = 0.08
        if speed <= speed_threshold:
            return 0.0
        return -min(0.5, (speed - speed_threshold) ** 2 * 5.0)

    def _randinit_metrics(self):
        prob = find_env_attr(self.env, "curr_prob")
        noise = find_env_attr(self.env, "curr_noise")
        return prob, noise

    def step(self, action):
        obs, extrinsic, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        intrinsic_reward = self.intrinsic.step_reward(obs)
        speed_penalty = self._speed_penalty(obs)
        total_reward = float(extrinsic + intrinsic_reward + speed_penalty)

        self.episode_reward += total_reward
        self.episode_steps += 1
        self.recent_intrinsic.append(intrinsic_reward)

        info.update(self.intrinsic.info())
        randinit_prob, randinit_noise = self._randinit_metrics()
        info["extrinsic_reward"] = float(extrinsic)
        info["intrinsic_reward"] = intrinsic_reward
        info["speed_penalty"] = speed_penalty
        info["total_reward"] = total_reward
        info["episode_steps"] = int(self.episode_steps)
        info["episode_idx"] = int(self.episode_idx)
        info["randinit_prob"] = float(randinit_prob) if randinit_prob is not None else np.nan
        info["randinit_noise"] = float(randinit_noise) if randinit_noise is not None else np.nan
        return obs, total_reward, terminated, truncated, info


class ProgressCallback(BaseCallback):
    def __init__(self, save_dir, save_freq=25_000, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = int(save_freq)
        self.last_save = 0

    def _on_step(self):
        if self.n_calls - self.last_save >= self.save_freq:
            path = os.path.join(self.save_dir, f"stelios_x_giannis_ppo_model_{self.n_calls}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"Saved model: {path}")
            self.last_save = self.n_calls
        return True


class StdDecayCallback(BaseCallback):
    def __init__(self, total_timesteps, init_log_std=0.3, final_log_std=-1.1, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.init_log_std = float(init_log_std)
        self.final_log_std = float(final_log_std)

    def _on_step(self):
        if not hasattr(self.model, "policy") or not hasattr(self.model.policy, "log_std"):
            return True
        fraction = min(1.0, self.model.num_timesteps / self.total_timesteps)
        current = self.init_log_std + fraction * (self.final_log_std - self.init_log_std)
        with th.no_grad():
            log_std_param = self.model.policy.log_std
            if log_std_param is not None:
                log_std_param.data.fill_(current)
        return True


class LiveMetricsPlotCallback(BaseCallback):
    def __init__(self, save_dir, plot_freq=2048, rolling_window=200, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.plot_freq = max(1, int(plot_freq))
        self.rolling_window = max(10, int(rolling_window))

        self.csv_path = os.path.join(save_dir, "live_metrics.csv")
        self.plot_path = os.path.join(save_dir, "live_metrics.png")

        self.step_metric_keys = (
            "normalized_reward",
            "extrinsic_reward",
            "intrinsic_reward",
            "speed_penalty",
            "total_reward",
        )
        self.latest_metric_keys = (
            "global_parts",
            "ep_parts",
            "global_left_geoms",
            "global_right_geoms",
            "ep_left_geoms",
            "ep_right_geoms",
            "randinit_prob",
            "randinit_noise",
            "episode_idx",
            "episode_steps",
        )

        self.step_windows = {key: deque(maxlen=self.rolling_window) for key in self.step_metric_keys}
        self.latest_values = {key: np.nan for key in self.latest_metric_keys}
        self.episode_returns = deque(maxlen=self.rolling_window)
        self.episode_lengths = deque(maxlen=self.rolling_window)
        self.history = defaultdict(list)
        self.completed_episodes = 0
        self.last_plot_timestep = 0

        self.csv_file = None
        self.csv_writer = None
        self.csv_fields = (
            "timesteps",
            "episodes_completed",
            "normalized_reward_mean",
            "extrinsic_reward_mean",
            "intrinsic_reward_mean",
            "speed_penalty_mean",
            "total_reward_mean",
            "episode_return_mean",
            "episode_length_mean",
            "global_parts",
            "ep_parts",
            "global_left_geoms",
            "global_right_geoms",
            "ep_left_geoms",
            "ep_right_geoms",
            "log_std_mean",
            "action_std_mean",
            "randinit_prob",
            "randinit_noise",
            "episode_idx",
            "episode_steps",
        )

    def _on_training_start(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields)
        self.csv_writer.writeheader()
        self.csv_file.flush()
        if self.verbose:
            print(f"Live plot will update at: {self.plot_path}")
            print(f"Live metrics CSV: {self.csv_path}")
        self._emit_snapshot(force=True)

    def _safe_float(self, value):
        if value is None:
            return np.nan
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return np.nan
            return float(arr.reshape(-1)[0])
        except Exception:
            return np.nan

    def _mean_or_nan(self, values):
        return float(np.mean(values)) if values else np.nan

    def _current_log_std(self):
        if not hasattr(self.model, "policy") or not hasattr(self.model.policy, "log_std"):
            return np.nan
        log_std_param = getattr(self.model.policy, "log_std", None)
        if log_std_param is None:
            return np.nan
        with th.no_grad():
            return float(log_std_param.detach().mean().cpu().item())

    def _update_from_infos(self):
        rewards = self.locals.get("rewards")
        if rewards is not None:
            reward_value = self._safe_float(rewards)
            if not np.isnan(reward_value):
                self.step_windows["normalized_reward"].append(reward_value)

        infos = self.locals.get("infos") or []
        info = infos[0] if infos else {}

        for key in ("extrinsic_reward", "intrinsic_reward", "speed_penalty", "total_reward"):
            value = self._safe_float(info.get(key))
            if not np.isnan(value):
                self.step_windows[key].append(value)

        for key in self.latest_metric_keys:
            value = self._safe_float(info.get(key))
            if not np.isnan(value):
                self.latest_values[key] = value

        episode_info = info.get("episode")
        if isinstance(episode_info, dict):
            ep_return = self._safe_float(episode_info.get("r"))
            ep_length = self._safe_float(episode_info.get("l"))
            if not np.isnan(ep_return):
                self.episode_returns.append(ep_return)
            if not np.isnan(ep_length):
                self.episode_lengths.append(ep_length)
            self.completed_episodes += 1

    def _snapshot_row(self):
        log_std = self._current_log_std()
        action_std = float(np.exp(log_std)) if not np.isnan(log_std) else np.nan
        return {
            "timesteps": int(self.model.num_timesteps),
            "episodes_completed": int(self.completed_episodes),
            "normalized_reward_mean": self._mean_or_nan(self.step_windows["normalized_reward"]),
            "extrinsic_reward_mean": self._mean_or_nan(self.step_windows["extrinsic_reward"]),
            "intrinsic_reward_mean": self._mean_or_nan(self.step_windows["intrinsic_reward"]),
            "speed_penalty_mean": self._mean_or_nan(self.step_windows["speed_penalty"]),
            "total_reward_mean": self._mean_or_nan(self.step_windows["total_reward"]),
            "episode_return_mean": self._mean_or_nan(self.episode_returns),
            "episode_length_mean": self._mean_or_nan(self.episode_lengths),
            "global_parts": self.latest_values["global_parts"],
            "ep_parts": self.latest_values["ep_parts"],
            "global_left_geoms": self.latest_values["global_left_geoms"],
            "global_right_geoms": self.latest_values["global_right_geoms"],
            "ep_left_geoms": self.latest_values["ep_left_geoms"],
            "ep_right_geoms": self.latest_values["ep_right_geoms"],
            "log_std_mean": log_std,
            "action_std_mean": action_std,
            "randinit_prob": self.latest_values["randinit_prob"],
            "randinit_noise": self.latest_values["randinit_noise"],
            "episode_idx": self.latest_values["episode_idx"],
            "episode_steps": self.latest_values["episode_steps"],
        }

    def _record_tensorboard_scalars(self, row):
        self.logger.record("selftouch_live/reward_normalized_mean", row["normalized_reward_mean"])
        self.logger.record("selftouch_live/reward_extrinsic_mean", row["extrinsic_reward_mean"])
        self.logger.record("selftouch_live/reward_intrinsic_mean", row["intrinsic_reward_mean"])
        self.logger.record("selftouch_live/reward_speed_penalty_mean", row["speed_penalty_mean"])
        self.logger.record("selftouch_live/reward_total_mean", row["total_reward_mean"])
        self.logger.record("selftouch_live/episode_return_mean", row["episode_return_mean"])
        self.logger.record("selftouch_live/episode_length_mean", row["episode_length_mean"])
        self.logger.record("selftouch_live/global_parts", row["global_parts"])
        self.logger.record("selftouch_live/global_left_geoms", row["global_left_geoms"])
        self.logger.record("selftouch_live/global_right_geoms", row["global_right_geoms"])
        self.logger.record("selftouch_live/log_std_mean", row["log_std_mean"])
        self.logger.record("selftouch_live/action_std_mean", row["action_std_mean"])
        self.logger.record("selftouch_live/randinit_prob", row["randinit_prob"])
        self.logger.record("selftouch_live/randinit_noise", row["randinit_noise"])

    def _write_plot(self):
        if not self.history["timesteps"]:
            return

        timesteps = self.history["timesteps"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=140)
        ax_rewards, ax_coverage, ax_episode, ax_schedule = axes.flatten()

        reward_series = (
            ("normalized_reward_mean", "normalized"),
            ("extrinsic_reward_mean", "extrinsic"),
            ("intrinsic_reward_mean", "intrinsic"),
            ("speed_penalty_mean", "speed_penalty"),
            ("total_reward_mean", "total"),
        )
        for key, label in reward_series:
            ax_rewards.plot(timesteps, self.history[key], label=label, linewidth=1.8)
        ax_rewards.set_title("Reward")
        ax_rewards.set_xlabel("timesteps")
        ax_rewards.set_ylabel("rolling mean")
        ax_rewards.grid(alpha=0.3)
        ax_rewards.legend(loc="best")

        coverage_series = (
            ("global_parts", "global_parts"),
            ("ep_parts", "ep_parts"),
            ("global_left_geoms", "global_left_geoms"),
            ("global_right_geoms", "global_right_geoms"),
        )
        for key, label in coverage_series:
            ax_coverage.plot(timesteps, self.history[key], label=label, linewidth=1.8)
        ax_coverage.set_title("Coverage")
        ax_coverage.set_xlabel("timesteps")
        ax_coverage.set_ylabel("count")
        ax_coverage.grid(alpha=0.3)
        ax_coverage.legend(loc="best")

        ax_episode.plot(timesteps, self.history["episode_return_mean"], label="episode_return_mean", linewidth=1.8)
        ax_episode.set_title("Episode Summary")
        ax_episode.set_xlabel("timesteps")
        ax_episode.set_ylabel("return")
        ax_episode.grid(alpha=0.3)
        episode_len_ax = ax_episode.twinx()
        episode_len_ax.plot(
            timesteps,
            self.history["episode_length_mean"],
            label="episode_length_mean",
            color="tab:orange",
            linewidth=1.8,
        )
        episode_len_ax.set_ylabel("length")
        episode_handles, episode_labels = ax_episode.get_legend_handles_labels()
        length_handles, length_labels = episode_len_ax.get_legend_handles_labels()
        ax_episode.legend(episode_handles + length_handles, episode_labels + length_labels, loc="best")

        schedule_series = (
            ("log_std_mean", "log_std"),
            ("action_std_mean", "action_std"),
            ("randinit_prob", "randinit_prob"),
            ("randinit_noise", "randinit_noise"),
        )
        for key, label in schedule_series:
            ax_schedule.plot(timesteps, self.history[key], label=label, linewidth=1.8)
        ax_schedule.set_title("Schedule")
        ax_schedule.set_xlabel("timesteps")
        ax_schedule.set_ylabel("value")
        ax_schedule.grid(alpha=0.3)
        ax_schedule.legend(loc="best")

        latest_text = (
            f"episodes={int(self.history['episodes_completed'][-1])}\n"
            f"global_parts={self.history['global_parts'][-1]:.0f}\n"
            f"left_geoms={self.history['global_left_geoms'][-1]:.0f}\n"
            f"right_geoms={self.history['global_right_geoms'][-1]:.0f}"
        )
        fig.text(0.985, 0.5, latest_text, va="center", ha="right", fontsize=10)
        fig.tight_layout(rect=(0, 0, 0.96, 1))
        fig.savefig(self.plot_path, bbox_inches="tight")
        plt.close(fig)

    def _emit_snapshot(self, force=False):
        current_timestep = int(getattr(self.model, "num_timesteps", 0))
        if not force and current_timestep - self.last_plot_timestep < self.plot_freq:
            return

        row = self._snapshot_row()
        for field in self.csv_fields:
            self.history[field].append(row[field])

        if self.csv_writer is not None:
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        self._record_tensorboard_scalars(row)
        self._write_plot()
        self.last_plot_timestep = current_timestep

    def _on_step(self):
        self._update_from_infos()
        self._emit_snapshot(force=False)
        return True

    def _on_training_end(self):
        self._emit_snapshot(force=True)
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None


class TouchBodyCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, groups=68, touch_hidden=256, obs_hidden=128, fused_dim=512):
        super().__init__(observation_space, features_dim=fused_dim)

        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("TouchBodyCombinedExtractor requires Dict observation space")

        self.has_obs = "observation" in observation_space.spaces
        self.has_touch = "touch" in observation_space.spaces
        self.groups = int(groups)
        self.touch_dim = int(np.prod(observation_space.spaces["touch"].shape)) if self.has_touch else 0
        self.obs_dim = int(np.prod(observation_space.spaces["observation"].shape)) if self.has_obs else 0

        if self.has_obs and self.obs_dim > 0:
            self.obs_net = nn.Sequential(
                nn.LayerNorm(self.obs_dim),
                nn.Linear(self.obs_dim, obs_hidden),
                nn.SiLU(),
                nn.Linear(obs_hidden, obs_hidden),
                nn.SiLU(),
            )
        else:
            self.obs_net = None
            obs_hidden = 0

        if self.has_touch and self.touch_dim > 0:
            self.touch_ln = nn.LayerNorm(self.groups)
            self.touch_net = nn.Sequential(
                nn.Linear(self.groups, touch_hidden),
                nn.SiLU(),
                nn.Linear(touch_hidden, touch_hidden),
                nn.SiLU(),
            )
        else:
            self.touch_ln = None
            self.touch_net = None
            touch_hidden = 0

        self.fuse = nn.Sequential(
            nn.Linear(obs_hidden + touch_hidden, fused_dim),
            nn.SiLU(),
        )

    def _reduce_touch_batch(self, touch):
        if touch.dim() == 1:
            touch = touch.unsqueeze(0)
        batch_size, touch_dim = touch.shape
        split_size = int(math.ceil(touch_dim / self.groups))
        chunks = th.split(touch, split_size, dim=1)
        means = [chunk.mean(dim=1, keepdim=True) for chunk in chunks[: self.groups]]
        if len(means) < self.groups:
            means.extend([touch.new_zeros((batch_size, 1)) for _ in range(self.groups - len(means))])
        return th.cat(means, dim=1)

    def forward(self, observations):
        features = []

        if self.obs_net is not None:
            obs = observations["observation"]
            if not isinstance(obs, th.Tensor):
                obs = th.as_tensor(obs)
            obs = obs.float()
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            else:
                obs = obs.view(obs.shape[0], -1)
            features.append(self.obs_net(obs))

        if self.touch_net is not None:
            touch = observations["touch"]
            if not isinstance(touch, th.Tensor):
                touch = th.as_tensor(touch)
            touch = touch.float()
            if touch.dim() == 1:
                touch = touch.unsqueeze(0)
            else:
                touch = touch.view(touch.shape[0], -1)
            pooled = self._reduce_touch_batch(touch)
            pooled = self.touch_ln(pooled)
            features.append(self.touch_net(pooled))

        fused = features[0] if len(features) == 1 else th.cat(features, dim=1)
        return self.fuse(fused)


class RandomInitJointsEpisodeRamp(gym.Wrapper):
    def __init__(
        self,
        env,
        start_prob=0.0,
        end_prob=1.0,
        start_noise=0.1,
        end_noise=1.0,
        ramp_episodes=1000,
        start_after_episodes=0,
        seed=None,
    ):
        super().__init__(env)
        self.start_prob = float(start_prob)
        self.end_prob = float(end_prob)
        self.start_noise = float(start_noise)
        self.end_noise = float(end_noise)
        self.ramp_episodes = max(1, int(ramp_episodes))
        self.start_after_episodes = max(0, int(start_after_episodes))
        self.rng = np.random.default_rng(seed)

        self.episode_count = 0
        self.curr_prob = self.start_prob
        self.curr_noise = self.start_noise

    def _update_schedule(self):
        fraction = min(1.0, self.episode_count / float(self.ramp_episodes))
        prob = (1.0 - fraction) * self.start_prob + fraction * self.end_prob
        noise = (1.0 - fraction) * self.start_noise + fraction * self.end_noise
        self.curr_prob = 0.0 if self.episode_count < self.start_after_episodes else prob
        self.curr_noise = noise

    def _randomize_angles(self):
        unwrapped = self.env.unwrapped
        model = getattr(unwrapped, "model", None)
        data = getattr(unwrapped, "data", None)
        if model is None or data is None:
            return

        for joint_id in range(model.njnt):
            joint_type = int(model.jnt_type[joint_id])
            qpos_addr = int(model.jnt_qposadr[joint_id])
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue

            limited = int(model.jnt_limited[joint_id]) == 1
            if limited:
                low = float(model.jnt_range[joint_id, 0])
                high = float(model.jnt_range[joint_id, 1])
                sample = self.rng.uniform(low, high)
            else:
                base = float(data.qpos[qpos_addr])
                width = self.curr_noise * (np.pi / 4.0 if joint_type == mujoco.mjtJoint.mjJNT_HINGE else 0.05)
                sample = base + self.rng.uniform(-width, width)

            data.qpos[qpos_addr] = sample

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def _refresh_observation(self, obs):
        for name in ("_get_obs", "observe", "get_observation", "get_obs"):
            fn = getattr(self.env.unwrapped, name, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_schedule()
        if self.rng.random() < self.curr_prob:
            self._randomize_angles()
            obs = self._refresh_observation(obs)

        self.episode_count += 1
        info = dict(info or {})
        info["randinit_prob"] = self.curr_prob
        info["randinit_noise"] = self.curr_noise
        info["episode_idx"] = self.episode_count
        return obs, info


def make_stage_env(
    config,
    stage,
    training=True,
    enable_speed_penalty=None,
    rand_start_prob=0.0,
    rand_end_prob=1.0,
    rand_start_noise=0.1,
    rand_end_noise=1.0,
    rand_ramp_episodes=1000,
    rand_start_after_episodes=0,
):
    env = bb_utils.make_env(config, training=training)
    if training and stage == "difficult":
        env = RandomInitJointsEpisodeRamp(
            env,
            start_prob=rand_start_prob,
            end_prob=rand_end_prob,
            start_noise=rand_start_noise,
            end_noise=rand_end_noise,
            ramp_episodes=rand_ramp_episodes,
            start_after_episodes=rand_start_after_episodes,
        )
    env = Float32ObsWrapper(env)
    if training:
        if enable_speed_penalty is None:
            enable_speed_penalty = stage == "base"
        env = IntrinsicSelfTouchWrapper(env, enable_speed_penalty=enable_speed_penalty)
    return env


def make_monitored_stage_env(
    config,
    stage,
    training=True,
    enable_speed_penalty=None,
    rand_start_prob=0.0,
    rand_end_prob=1.0,
    rand_start_noise=0.1,
    rand_end_noise=1.0,
    rand_ramp_episodes=1000,
    rand_start_after_episodes=0,
):
    env = make_stage_env(
        config=config,
        stage=stage,
        training=training,
        enable_speed_penalty=enable_speed_penalty,
        rand_start_prob=rand_start_prob,
        rand_end_prob=rand_end_prob,
        rand_start_noise=rand_start_noise,
        rand_end_noise=rand_end_noise,
        rand_ramp_episodes=rand_ramp_episodes,
        rand_start_after_episodes=rand_start_after_episodes,
    )
    return Monitor(
        env,
        info_keywords=(
            "extrinsic_reward",
            "intrinsic_reward",
            "speed_penalty",
            "total_reward",
            "global_parts",
            "ep_parts",
            "global_left_geoms",
            "global_right_geoms",
            "ep_left_geoms",
            "ep_right_geoms",
            "episode_steps",
            "episode_idx",
            "randinit_prob",
            "randinit_noise",
        ),
    )


def make_vec_env_from_single_env(env):
    return DummyVecEnv([lambda: env])


def make_vecnormalize(venv, training=True):
    vec = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=5.0, gamma=0.99)
    vec.training = bool(training)
    if not training:
        vec.norm_reward = False
    return vec


def policy_kwargs():
    return dict(
        features_extractor_class=TouchBodyCombinedExtractor,
        features_extractor_kwargs=dict(groups=68, touch_hidden=256, obs_hidden=128, fused_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.SiLU,
        ortho_init=True,
    )


def build_base_ppo(venv, tensorboard_log, device="auto"):
    kwargs = policy_kwargs()
    kwargs["log_std_init"] = 0.5
    return PPO(
        "MultiInputPolicy",
        venv,
        verbose=1,
        learning_rate=7.5e-5,
        n_steps=2048,
        batch_size=512,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.10,
        ent_coef=0.001,
        vf_coef=0.7,
        clip_range_vf=0.2,
        max_grad_norm=0.5,
        target_kl=None,
        use_sde=False,
        policy_kwargs=kwargs,
        tensorboard_log=tensorboard_log,
        device=device,
    )


def build_difficult_ppo(venv, tensorboard_log, device="auto"):
    kwargs = policy_kwargs()
    kwargs["log_std_init"] = 0.0
    return PPO(
        "MultiInputPolicy",
        venv,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.06,
        use_sde=False,
        policy_kwargs=kwargs,
        tensorboard_log=tensorboard_log,
        device=device,
    )


def load_or_create_stage_model(stage, venv, save_dir, resume_model=None, device="auto"):
    resolved_device = resolve_device(device)
    tensorboard_log = maybe_tensorboard_log_dir(save_dir)
    if resume_model:
        model = PPO.load(resume_model, device=resolved_device)
        model.set_env(venv)
        model.tensorboard_log = tensorboard_log
        return model
    if stage == "base":
        return build_base_ppo(venv, tensorboard_log, device=resolved_device)
    if stage == "difficult":
        return build_difficult_ppo(venv, tensorboard_log, device=resolved_device)
    if stage == "after":
        raise ValueError("The 'after' stage requires --resume-model")
    raise ValueError(f"Unsupported stage: {stage}")


def load_vecnormalize_for_eval(raw_env, vecnormalize_path):
    venv = make_vec_env_from_single_env(raw_env)
    venv = VecNormalize.load(vecnormalize_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def ensure_eval_dirs(save_dir):
    os.makedirs(os.path.join(save_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)


def manual_contact_stats(env):
    unwrapped = get_unwrapped_env(env)
    left_hand_geoms = set(getattr(unwrapped, "left_hand_geoms", []))
    right_hand_geoms = set(getattr(unwrapped, "right_hand_geoms", []))
    body_geoms = set(getattr(unwrapped, "mimo_geoms", []))

    left_hit = set()
    right_hit = set()

    data = getattr(unwrapped, "data", None)
    if data is None:
        return left_hit, right_hit

    n_contacts = int(getattr(data, "ncon", 0))
    for i in range(n_contacts):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        if geom1 in left_hand_geoms and geom2 in body_geoms:
            left_hit.add(geom2)
        elif geom2 in left_hand_geoms and geom1 in body_geoms:
            left_hit.add(geom1)

        if geom1 in right_hand_geoms and geom2 in body_geoms:
            right_hit.add(geom2)
        elif geom2 in right_hand_geoms and geom1 in body_geoms:
            right_hit.add(geom1)

    return left_hit, right_hit


def make_eval_runner(env, duration, render, save_dir):
    return bb_eval.EVALS["self_touch"](
        env=env,
        duration=duration,
        render=render,
        save_dir=save_dir,
    )
