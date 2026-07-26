"""Macro-transition recording and Safe-VLN episode summaries."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from .cmdp import compute_returns


class SafeTrajectoryRecorder:
    def __init__(
        self,
        *,
        episode_id: str,
        scene_id: str,
        instruction: str,
        gamma: float = 0.99,
        progress_scale: float = 1.0,
        step_penalty: float = -0.01,
        success_reward: float = 10.0,
        cost_limit: float = 0.0,
    ) -> None:
        self.episode_id = str(episode_id)
        self.scene_id = scene_id
        self.instruction = instruction
        self.gamma = gamma
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.cost_limit = cost_limit
        self.transitions: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None

    def begin(self, policy_output: Mapping[str, Any], distance_before: float) -> None:
        if self._active is not None:
            raise RuntimeError("cannot begin a macro action before finalizing the active action")
        self._active = {
            "index": len(self.transitions),
            "action_id": int(policy_output["action_id"]),
            "action_text": policy_output["text"],
            "velocity_command": list(policy_output["velocity_command"]),
            "requested_duration": float(policy_output["duration"]),
            "reward_value": policy_output.get("reward_value"),
            "cost_value": policy_output.get("cost_value"),
            "old_log_prob": policy_output.get("log_prob"),
            "policy_version": policy_output.get("policy_version"),
            "decision_id": policy_output.get("decision_id"),
            "action_probabilities": deepcopy(
                policy_output.get("action_probabilities")
            ),
            "invalid_action": bool(policy_output.get("invalid_action", False)),
            "distance_before": float(distance_before),
            "executed_env_steps": 0,
        }

    def count_env_step(self) -> None:
        if self._active is None:
            raise RuntimeError("no active macro action")
        self._active["executed_env_steps"] += 1

    def finish(
        self,
        *,
        distance_after: float,
        reward_override: float | None = None,
        reward_components: Mapping[str, Any] | None = None,
        success: bool = False,
        unsafe_contact: bool = False,
        fall: bool = False,
        blocked: bool = False,
        safety_diagnostics: Mapping[str, Any] | None = None,
        terminated: bool = False,
        truncated: bool = False,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        if self._active is None:
            raise RuntimeError("no active macro action")
        if not math.isfinite(distance_after):
            raise ValueError("distance_after must be finite")
        transition = self._active
        self._active = None
        progress = transition["distance_before"] - float(distance_after)
        if reward_override is None:
            reward = (
                self.progress_scale * progress
                + self.step_penalty
                + self.success_reward * float(success)
            )
            resolved_reward_components = {
                "physical_progress": self.progress_scale * progress,
                "step_penalty": self.step_penalty,
                "success": self.success_reward * float(success),
            }
        else:
            reward = float(reward_override)
            if not math.isfinite(reward):
                raise ValueError("reward_override must be finite")
            resolved_reward_components = deepcopy(dict(reward_components or {}))
        transition.update(
            {
                "distance_after": float(distance_after),
                "physical_progress": progress,
                "reward": reward,
                "reward_components": resolved_reward_components,
                "cost": float(unsafe_contact) + float(fall) + float(blocked),
                "cost_components": {
                    "unsafe_contact": float(unsafe_contact),
                    "fall": float(fall),
                    "blocked": float(blocked),
                },
                "safety_diagnostics": deepcopy(dict(safety_diagnostics or {})),
                "success": bool(success),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "done": bool(terminated or truncated),
                "termination_reason": termination_reason,
            }
        )
        self.transitions.append(transition)
        return deepcopy(transition)

    def finalize(self) -> None:
        if self._active is not None:
            raise RuntimeError("cannot finalize episode with an active macro action")
        dones = [bool(item["done"]) for item in self.transitions]
        reward_returns = compute_returns([item["reward"] for item in self.transitions], dones, self.gamma)
        cost_returns = compute_returns([item["cost"] for item in self.transitions], dones, self.gamma)
        for item, reward_return, cost_return in zip(self.transitions, reward_returns, cost_returns):
            item["reward_return"] = reward_return
            item["cost_return"] = cost_return

    def summary(self, measurements: Mapping[str, Any] | None = None) -> dict[str, Any]:
        measurements = measurements or {}
        cumulative_cost = sum(item.get("cost", 0.0) for item in self.transitions)
        success = bool(measurements.get("success", False))
        constraint_satisfied = cumulative_cost <= self.cost_limit
        last_reason = self.transitions[-1].get("termination_reason") if self.transitions else None
        spl = float(measurements.get("spl", 0.0) or 0.0)
        return {
            "total_high_level_reward": sum(item.get("reward", 0.0) for item in self.transitions),
            "cumulative_cost": cumulative_cost,
            "unsafe_contact_count": sum(item.get("cost_components", {}).get("unsafe_contact", 0.0) for item in self.transitions),
            "fall_count": sum(item.get("cost_components", {}).get("fall", 0.0) for item in self.transitions),
            "blocked_count": sum(item.get("cost_components", {}).get("blocked", 0.0) for item in self.transitions),
            "has_collision": any(item.get("cost_components", {}).get("unsafe_contact", 0.0) > 0 for item in self.transitions),
            "has_blocked": any(item.get("cost_components", {}).get("blocked", 0.0) > 0 for item in self.transitions),
            "num_macro_actions": len(self.transitions),
            "invalid_action_count": sum(bool(item.get("invalid_action")) for item in self.transitions),
            "constraint_satisfied": constraint_satisfied,
            "safe_success": success and constraint_satisfied,
            "safe_spl": spl if constraint_satisfied else 0.0,
            "termination_reason": last_reason,
        }

    def to_dict(self, measurements: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": "safe-vln-go2-v1",
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "instruction": self.instruction,
            "gamma": self.gamma,
            "summary": self.summary(measurements),
            "transitions": deepcopy(self.transitions),
        }
