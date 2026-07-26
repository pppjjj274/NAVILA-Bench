# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import torch
import numpy as np

from rsl_rl.env import VecEnv
from safe_vln.safety import (
    BlockedDetector,
    blocked_progress_risk,
    combined_step_risk,
    front_obstacle_distance,
    inverse_distance_risk,
    linear_risk,
    orientation_angle as compute_orientation_angle,
    smoothness_risk,
    unsafe_contact_diagnostics,
)
from safe_vln.objective import default_cost_profile, validate_cost_profile

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from .measures import add_measurement


def get_proprio_obs_dim(env: ManagerBasedRLEnv) -> int:
    """Returns the dimension of the proprioceptive observations."""
    return env.unwrapped.observation_manager.compute_group("proprio").shape[1]


class RslRlVecEnvHistoryWrapper(RslRlVecEnvWrapper):
    """Wraps around Isaac Lab environment for RSL-RL to add history buffer to the proprioception observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv, history_length: int = 1):
        """Initializes the wrapper."""
        super().__init__(env)

        self.history_length = history_length
        self.proprio_obs_dim = get_proprio_obs_dim(env)
        self.proprio_obs_buf = torch.zeros(self.num_envs, self.history_length, self.proprio_obs_dim,
                                                    dtype=torch.float, device=self.unwrapped.device)

        self.clip_actions = 20.0

    """
    Properties
    """
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        self.proprio_obs_buf = torch.cat([proprio_obs.unsqueeze(1)] * self.history_length, dim=1)
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        obs_dict["policy"] = curr_obs

        return curr_obs, {"observations": obs_dict}

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Resets the environment."""
        obs_dict, infos = self.env.reset()
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        self.proprio_obs_buf = torch.stack([torch.zeros_like(proprio_obs)] * self.history_length, dim=1)
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        infos["observations"] = obs_dict

        return curr_obs, infos

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip the actions (for testing only)
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        # print("============== Height Map ==============")
        # print(obs_dict["test_height_map"])
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # update obsservation history buffer & reset the history buffer for done environments
        self.proprio_obs_buf = torch.where(
            (self.episode_length_buf < 1)[:, None, None],
            torch.stack([torch.zeros_like(proprio_obs)] * self.history_length, dim=1),
            torch.cat([
                self.proprio_obs_buf[:, 1:],
                proprio_obs.unsqueeze(1)
            ], dim=1)
        )
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        extras["observations"]["policy"] = curr_obs

        # return the step information
        return curr_obs, rew, dones, extras

    def update_command(self, command: torch.Tensor) -> None:
        """Updates the command for the environment."""
        self.proprio_obs_buf[:, -1, 6:9] = command

    def close(self):  # noqa: D102
        return self.env.close()


class VLNEnvWrapper:
    """Wrapper to configure an :class:`ManagerBasedRLEnv` instance to VLN environment."""

    def __init__(self, env: ManagerBasedRLEnv,
                 low_level_policy, task_name,
                 episode, max_length=10000, high_level_obs_key: str | None = "camera_obs",
                 safe_vln=False, contact_threshold=1.0, orientation_limit=0.8,
                 blocked_seconds=2.0, blocked_distance=0.1,
                 cost_profile=None, calibration_file=None,
                 measure_names=["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
        ):
        self.env = env
        self.task_name = task_name
        self.episode = episode
        self.measure_names = measure_names
        if safe_vln and task_name != "go2_matterport_vision":
            raise ValueError("Safe-VLN currently supports only go2_matterport_vision")
        self.safe_vln = safe_vln
        if cost_profile is None:
            profile = deepcopy(default_cost_profile())
            profile.pop("fingerprint", None)
            profile["hard_thresholds"].update(
                {
                    "contact_force_n": float(contact_threshold),
                    "orientation_rad": float(orientation_limit),
                    "blocked_seconds": float(blocked_seconds),
                    "blocked_distance_m": float(blocked_distance),
                }
            )
            self.cost_profile = validate_cost_profile(profile)
        else:
            self.cost_profile = validate_cost_profile(cost_profile)
        hard = self.cost_profile["hard_thresholds"]
        self.contact_threshold = float(hard["contact_force_n"])
        self.orientation_limit = float(hard["orientation_rad"])
        self.blocked_seconds = float(hard["blocked_seconds"])
        if self.blocked_seconds <= 0:
            raise ValueError("blocked_seconds must be positive")
        step_dt = float(self.unwrapped.cfg.sim.dt * self.unwrapped.cfg.decimation)
        self.blocked_window_steps = max(1, round(self.blocked_seconds / step_dt))
        self.blocked_detector = BlockedDetector(
            window_steps=self.blocked_window_steps,
            min_displacement=float(hard["blocked_distance_m"]),
            forward_threshold=float(hard["forward_command_mps"]),
        )
        self._previous_macro_command = None
        self._active_smoothness_risk = 0.0
        self._calibration_file = None
        if calibration_file:
            calibration_path = Path(calibration_file).expanduser()
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            self._calibration_file = calibration_path.open("a", encoding="utf-8")
        self.last_safety = self._empty_safety()

        self.env_step = 0
        self.max_length = max_length

        self.high_level_obs_key = high_level_obs_key
        if high_level_obs_key is not None:
            assert high_level_obs_key in self.env.observation_space.spaces.keys() # CHECK this

        self.low_level_policy = low_level_policy
        self.low_level_action = None

        self.curr_pos, self.prev_pos = None, None
        self.is_stop_called = False

    @staticmethod
    def _empty_safety():
        return {
            "unsafe_contact": False,
            "fall": False,
            "blocked": False,
            "cost": 0.0,
            "hard_cost": 0.0,
            "dense_risk": 0.0,
            "hard_violation": False,
            "risk_components": {
                "contact": 0.0,
                "tilt": 0.0,
                "near_obstacle": 0.0,
                "blocked": 0.0,
                "speed_near": 0.0,
                "smoothness": 0.0,
            },
            "termination_reason": None,
            "contact_bodies": [],
            "contact_body_forces": {},
            "max_unsafe_contact_force": 0.0,
            "orientation_angle": 0.0,
            "blocked_steps": 0,
            "blocked_displacement": 0.0,
            "front_obstacle_distance_m": None,
            "planar_speed_mps": 0.0,
        }

    def _compute_go2_safety(self, action, previous_position, current_position):
        """Read post-physics Go2 safety state before any high-level reset."""
        if not self.safe_vln:
            return self._empty_safety()

        contact_sensor = self.unwrapped.scene.sensors["contact_forces"]
        unsafe_contact, contact_body_forces, max_contact_force = (
            unsafe_contact_diagnostics(
                contact_sensor.body_names,
                contact_sensor.data.net_forces_w_history[0],
                self.contact_threshold,
            )
        )

        projected_gravity = self.unwrapped.scene["robot"].data.projected_gravity_b
        orientation = compute_orientation_angle(projected_gravity[0])
        fall = orientation > self.orientation_limit

        if unsafe_contact or fall:
            self.blocked_detector.reset()
            blocked_status = None
        else:
            blocked_status = self.blocked_detector.update(
                action, previous_position, current_position
            )
        blocked = bool(blocked_status and blocked_status.blocked)
        soft = self.cost_profile["soft_thresholds"]
        ray_sector = self.cost_profile["ray_sector"]
        try:
            lidar = self.unwrapped.scene.sensors["lidar_sensor"]
        except KeyError as exc:
            raise RuntimeError(
                "Safe-VLN v2 requires the Go2 lidar_sensor even when cameras are disabled"
            ) from exc
        robot = self.unwrapped.scene["robot"]
        obstacle_distance = front_obstacle_distance(
            lidar.data.pos_w[0],
            lidar.data.ray_hits_w[0],
            robot.data.root_quat_w[0],
            horizontal_half_angle_deg=ray_sector["horizontal_half_angle_deg"],
            vertical_half_angle_deg=ray_sector["vertical_half_angle_deg"],
        )
        planar_speed = float(
            torch.linalg.vector_norm(robot.data.root_lin_vel_b[0, :2]).item()
        )
        near_risk = (
            0.0
            if obstacle_distance is None
            else inverse_distance_risk(
                obstacle_distance,
                soft["near_critical_m"],
                soft["near_safe_m"],
            )
        )
        blocked_risk = (
            0.0
            if blocked_status is None
            else blocked_progress_risk(
                blocked_status,
                window_steps=self.blocked_window_steps,
                min_displacement=float(
                    self.cost_profile["hard_thresholds"]["blocked_distance_m"]
                ),
            )
        )
        risk_components = {
            "contact": linear_risk(
                max_contact_force,
                soft["contact_force_n"],
                self.contact_threshold,
            ),
            "tilt": linear_risk(
                orientation, soft["orientation_rad"], self.orientation_limit
            ),
            "near_obstacle": near_risk,
            "blocked": blocked_risk,
            "speed_near": near_risk
            * min(1.0, planar_speed / soft["planar_speed_scale_mps"]),
            "smoothness": self._active_smoothness_risk,
        }
        dense_risk = combined_step_risk(
            risk_components, self.cost_profile["risk_weights"]
        )
        hard_violation = unsafe_contact or fall or blocked

        reasons = []
        if unsafe_contact:
            reasons.append("unsafe_contact")
        if fall:
            reasons.append("fall")
        if blocked:
            reasons.append("blocked")
        return {
            "unsafe_contact": unsafe_contact,
            "fall": fall,
            "blocked": blocked,
            "cost": float(hard_violation) + dense_risk,
            "hard_cost": float(hard_violation),
            "dense_risk": dense_risk,
            "hard_violation": hard_violation,
            "risk_components": risk_components,
            "termination_reason": "+".join(reasons) or None,
            "contact_bodies": sorted(contact_body_forces),
            "contact_body_forces": contact_body_forces,
            "max_unsafe_contact_force": max_contact_force,
            "orientation_angle": orientation,
            "blocked_steps": blocked_status.observed_steps if blocked_status else 0,
            "blocked_displacement": blocked_status.displacement if blocked_status else 0.0,
            "front_obstacle_distance_m": obstacle_distance,
            "planar_speed_mps": planar_speed,
        }

    def begin_macro_action(self, command) -> None:
        """Set the command-transition risk once at a high-level boundary."""
        current = torch.as_tensor(command).reshape(-1)[:3].detach().cpu()
        self._active_smoothness_risk = smoothness_risk(
            self._previous_macro_command, current
        )
        self._previous_macro_command = current.clone()

    def _write_calibration_record(self, safety) -> None:
        if self._calibration_file is None:
            return
        record = {
            "episode_id": str(self.episode.get("episode_id")),
            "env_step": self.env_step,
            "hard_violation": bool(safety["hard_violation"]),
            "unsafe_contact": bool(safety["unsafe_contact"]),
            "fall": bool(safety["fall"]),
            "blocked": bool(safety["blocked"]),
            "max_unsafe_contact_force": safety["max_unsafe_contact_force"],
            "orientation_angle": safety["orientation_angle"],
            "front_obstacle_distance_m": safety["front_obstacle_distance_m"],
            "planar_speed_mps": safety["planar_speed_mps"],
        }
        self._calibration_file.write(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self._calibration_file.flush()

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def set_measures(self):
        self.measure_manager = add_measurement(self.env, self.episode, self.measure_names)

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
        low_level_obs, infos = self.env.reset()
        self.low_level_obs = low_level_obs
        zero_cmd = torch.tensor([0., 0., 0.], device=low_level_obs.device)

        if "go2" in self.task_name:
            warmup_steps = 100
        elif "h1" or "g1" in self.task_name:
            warmup_steps = 200
        else:
            warmup_steps = 50

        for i in range(warmup_steps):
            if i % 100 == 0 or i == warmup_steps - 1:
                print(f"Warmup step {i}/{warmup_steps}...")

            self.update_command(zero_cmd)
            actions = self.low_level_policy(self.low_level_obs)
            low_level_obs, _, _, infos = self.env.step(actions)
            self.low_level_obs = low_level_obs
            self.low_level_action = actions

        self.env_step, self.same_pos_count = 0, 0
        self.blocked_detector.reset()
        self._previous_macro_command = None
        self._active_smoothness_risk = 0.0
        self.last_safety = self._empty_safety()

        self.set_measures()
        self.measure_manager.reset_measures()
        measurements = self.measure_manager.get_measurements()
        infos["measurements"] = measurements

        self.prev_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().clone()

        obs = (
            low_level_obs
            if self.high_level_obs_key is None
            else infos["observations"][self.high_level_obs_key]
        )
        return obs, infos

    def update_command(self, command) -> None:
        """Update the command for the low-level policy."""

        # make sure command is a tensor on the same device as low_level_obs
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=self.env.unwrapped.device)

        if isinstance(self.env, RslRlVecEnvHistoryWrapper):
            self.low_level_obs[:, 6:9] = command
            self.env.proprio_obs_buf[:, -1, 6:9] = command

        else:
            self.low_level_obs[:, 9:12] = command

    def step(self, action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in the environment.

        Args:
            action: The action of high-level planner, which should be velocity command to the low-level policy.

        Returns:
            obs: The observation of the high-level planner.
            reward: The reward of the environment.
            done: Whether the episode is done.
            info: Additional information of the environment.

        """

        self.update_command(action)

        low_level_action = self.low_level_policy(self.low_level_obs)
        self.low_level_action = low_level_action

        low_level_obs, reward, done, info = self.env.step(low_level_action)
        self.low_level_obs = low_level_obs
        obs = (
            low_level_obs
            if self.high_level_obs_key is None
            else info["observations"][self.high_level_obs_key]
        )
        self.env_step += 1

        current_pos = self.unwrapped.scene["robot"].data.root_pos_w[0].detach()
        if self.safe_vln:
            self.last_safety = self._compute_go2_safety(
                action, self.prev_pos, current_pos
            )
            self._write_calibration_record(self.last_safety)
            same_pos = False
            self.prev_pos = current_pos.clone()
        else:
            self.last_safety = self._empty_safety()
            same_pos = self.check_same_pos()
        if self.last_safety["hard_violation"]:
            print(
                "[SAFE-VLN] terminating "
                f"reason={self.last_safety['termination_reason']} "
                f"contact_bodies={self.last_safety['contact_bodies']} "
                f"max_force={self.last_safety['max_unsafe_contact_force']:.3f}N "
                f"orientation={self.last_safety['orientation_angle']:.3f}rad "
                f"blocked_steps={self.last_safety['blocked_steps']} "
                f"blocked_displacement={self.last_safety['blocked_displacement']:.3f}m"
            )


        self.measure_manager.update_measures()
        measurements = self.measure_manager.get_measurements()
        info["measurements"] = measurements
        info["safety"] = self.last_safety

        done = (
            done[0]
            or self.last_safety["hard_violation"]
            or same_pos
            or self.env_step >= self.max_length
            or self.is_stop_called
        )

        return obs, reward, done, info

    def check_same_pos(self) -> bool:
        curr_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach()
        robot_vel = torch.norm(self.env.unwrapped.scene["robot"].data.root_vel_w[0].detach())
        if torch.norm(curr_pos - self.prev_pos) < 0.01 and robot_vel < 0.1:
            self.same_pos_count += 1
        else:
            self.same_pos_count = 0
        self.prev_pos = curr_pos.clone()

        # Break out of the loop if the robot has stayed in the same location for 1000 steps
        if self.same_pos_count >= 1000:
            print("Robot has stayed in the same location for 1000 steps. Breaking out of the loop.")
            return True

        return False

    def set_stop_called(self, is_stop_called: bool) -> None:
        """Set the stop called flag."""
        self.env.is_stop_called = is_stop_called
        self.is_stop_called = is_stop_called

    def close(self) -> None:
        if self._calibration_file is not None:
            self._calibration_file.close()
            self._calibration_file = None
        self.env.close()
