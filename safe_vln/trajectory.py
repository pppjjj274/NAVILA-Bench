"""Macro-transition recording and Safe-VLN episode summaries."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from .cmdp import compute_returns
from .objective import COST_NORMALIZATION, SCHEMA_VERSION


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
        failed_stop_penalty: float = -1.0,
        missed_stop_penalty: float = -0.5,
        cost_limit: float = 0.0,
        objective_config: Mapping[str, Any] | None = None,
        schema_version: str | None = None,
    ) -> None:
        self.episode_id = str(episode_id)
        self.scene_id = scene_id
        self.instruction = instruction
        self.gamma = gamma
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.failed_stop_penalty = failed_stop_penalty
        self.missed_stop_penalty = missed_stop_penalty
        self.cost_limit = cost_limit
        self.objective_config = deepcopy(dict(objective_config or {}))
        self.schema_version = (
            str(schema_version)
            if schema_version is not None
            else (
                str(self.objective_config.get("schema_version", SCHEMA_VERSION))
                if self.objective_config
                else "safe-vln-go2-v1"
            )
        )
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
            "policy_interface": policy_output.get("policy_interface"),
            "policy_objective_fingerprint": policy_output.get(
                "objective_fingerprint"
            ),
            "decision_id": policy_output.get("decision_id"),
            "action_probabilities": deepcopy(
                policy_output.get("action_probabilities")
            ),
            "invalid_action": bool(policy_output.get("invalid_action", False)),
            "distance_before": float(distance_before),
            "executed_env_steps": 0,
            "_risk_sum": 0.0,
            "_risk_peak": 0.0,
            "_risk_component_sums": {},
            "_risk_component_peaks": {},
            "_hard_events": {
                "unsafe_contact": False,
                "fall": False,
                "blocked": False,
            },
            "_turn_tracking_failure": False,
        }

    def count_env_step(self) -> None:
        if self._active is None:
            raise RuntimeError("no active macro action")
        self._active["executed_env_steps"] += 1

    def record_env_step(self, safety: Mapping[str, Any]) -> None:
        """Accumulate one low-level safety observation inside a macro action."""
        if self._active is None:
            raise RuntimeError("no active macro action")
        self.count_env_step()
        risk = float(safety.get("dense_risk", 0.0) or 0.0)
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0 + 1e-6:
            raise ValueError("dense_risk must be finite and in [0, 1]")
        self._active["_risk_sum"] += risk
        self._active["_risk_peak"] = max(self._active["_risk_peak"], risk)
        components = dict(safety.get("risk_components", {}) or {})
        for key, value in components.items():
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError(f"risk component {key!r} must be finite")
            sums = self._active["_risk_component_sums"]
            peaks = self._active["_risk_component_peaks"]
            sums[key] = sums.get(key, 0.0) + resolved
            peaks[key] = max(peaks.get(key, 0.0), resolved)
        for key in self._active["_hard_events"]:
            self._active["_hard_events"][key] = bool(
                self._active["_hard_events"][key] or safety.get(key, False)
            )
        self._active["_turn_tracking_failure"] = bool(
            self._active["_turn_tracking_failure"]
            or safety.get(
                "turn_tracking_failure", safety.get("turn_blocked", False)
            )
        )

    def finish(
        self,
        *,
        distance_after: float,
        reward_override: float | None = None,
        reward_components: Mapping[str, Any] | None = None,
        success: bool = False,
        failed_stop: bool = False,
        missed_stop: bool = False,
        unsafe_contact: bool = False,
        fall: bool = False,
        blocked: bool = False,
        turn_blocked: bool = False,
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
                + self.failed_stop_penalty * float(failed_stop)
                + self.missed_stop_penalty * float(missed_stop)
            )
            resolved_reward_components = {
                "physical_progress": self.progress_scale * progress,
                "step_penalty": self.step_penalty,
                "success": self.success_reward * float(success),
                "failed_stop": self.failed_stop_penalty * float(failed_stop),
                "missed_stop": self.missed_stop_penalty * float(missed_stop),
            }
        else:
            reward = float(reward_override)
            if not math.isfinite(reward):
                raise ValueError("reward_override must be finite")
            resolved_reward_components = deepcopy(dict(reward_components or {}))
        hard_events = transition.pop("_hard_events")
        hard_events["unsafe_contact"] = bool(
            hard_events["unsafe_contact"] or unsafe_contact
        )
        hard_events["fall"] = bool(hard_events["fall"] or fall)
        hard_events["blocked"] = bool(hard_events["blocked"] or blocked)
        turn_tracking_failure = bool(
            transition.pop("_turn_tracking_failure")
            or turn_blocked
            or (safety_diagnostics or {}).get(
                "turn_tracking_failure",
                (safety_diagnostics or {}).get("turn_blocked", False),
            )
        )
        hard_cost = float(any(hard_events.values()))
        count = int(transition["executed_env_steps"])
        risk_mean = transition.pop("_risk_sum") / max(count, 1)
        risk_peak = transition.pop("_risk_peak")
        component_sums = transition.pop("_risk_component_sums")
        component_peaks = transition.pop("_risk_component_peaks")
        component_means = {
            key: value / max(count, 1) for key, value in component_sums.items()
        }
        if self.objective_config:
            aggregation = self.objective_config["cost_profile"]["macro_aggregation"]
            dense_cost = (
                float(aggregation["mean_scale"]) * risk_mean
                + float(aggregation["peak_scale"]) * risk_peak
            )
            dense_cost = min(float(aggregation["max_dense_cost"]), dense_cost)
        else:
            dense_cost = 0.0
            hard_cost = (
                float(hard_events["unsafe_contact"])
                + float(hard_events["fall"])
                + float(hard_events["blocked"])
            )
        cost = hard_cost + dense_cost
        transition.update(
            {
                "distance_after": float(distance_after),
                "physical_progress": progress,
                "reward": reward,
                "reward_components": resolved_reward_components,
                "cost": cost,
                "hard_cost": hard_cost,
                "dense_cost": dense_cost,
                "hard_violation": bool(any(hard_events.values())),
                "cost_components": {
                    "unsafe_contact": float(hard_events["unsafe_contact"]),
                    "fall": float(hard_events["fall"]),
                    "blocked": float(hard_events["blocked"]),
                    # Controller tracking diagnostics are retained for analysis
                    # but are excluded from the CMDP cost and terminal signal.
                    "turn_tracking_failure": float(turn_tracking_failure),
                    "turn_blocked": float(turn_tracking_failure),
                    "collision_event": float(hard_events["unsafe_contact"]),
                    "fall_event": float(hard_events["fall"]),
                    "blocked_event": float(hard_events["blocked"]),
                    "dense_risk_mean": risk_mean,
                    "dense_risk_peak": risk_peak,
                    "risk_component_means": component_means,
                    "risk_component_peaks": component_peaks,
                },
                "safety_diagnostics": deepcopy(dict(safety_diagnostics or {})),
                "success": bool(success),
                "failed_stop": bool(failed_stop),
                "missed_stop": bool(missed_stop),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "done": bool(terminated or truncated),
                "termination_reason": termination_reason,
                "schema_version": self.schema_version,
                "objective_fingerprint": self.objective_config.get("fingerprint"),
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

    def truncate_last(self, reason: str) -> None:
        """Close an otherwise complete final decision at an external limit."""

        if self._active is not None:
            raise RuntimeError("cannot truncate an active macro action")
        if not self.transitions:
            raise RuntimeError("cannot truncate an empty trajectory")
        if not isinstance(reason, str) or not reason:
            raise ValueError("truncation reason must be a non-empty string")
        transition = self.transitions[-1]
        if transition.get("done"):
            raise RuntimeError("final transition is already terminal")
        transition.update(
            {
                "done": True,
                "terminated": False,
                "truncated": True,
                "termination_reason": reason,
                "next_observation_key": None,
            }
        )

    def summary(self, measurements: Mapping[str, Any] | None = None) -> dict[str, Any]:
        measurements = measurements or {}
        cumulative_cost = sum(item.get("cost", 0.0) for item in self.transitions)
        # Match the CMDP used by SafeVLA: constrain expected cumulative
        # episode cost. Averaging by trajectory length would dilute a unit
        # collision/fall cost and could mark a long unsafe episode as safe.
        constraint_cost = cumulative_cost
        policy_success = any(
            bool(item.get("policy_success", item.get("success", False)))
            for item in self.transitions
        )
        system_success = any(
            bool(item.get("system_success", item.get("success", False)))
            for item in self.transitions
        )
        success = bool(system_success or measurements.get("success", False))
        constraint_satisfied = constraint_cost <= self.cost_limit
        last_reason = self.transitions[-1].get("termination_reason") if self.transitions else None
        spl = float(measurements.get("spl", 0.0) or 0.0)
        valid_distances = [
            float(value)
            for item in self.transitions
            for value in (item.get("distance_before"), item.get("distance_after"))
            if value is not None and math.isfinite(float(value))
        ]
        goal_dwell_decisions = sum(
            bool(item.get("in_goal_radius", False)) for item in self.transitions
        )
        oracle_stop_decisions = sum(
            bool(item.get("oracle_valid", False))
            and item.get("oracle_action_id") == 9
            for item in self.transitions
        )
        model_stop_decisions = sum(
            item.get("policy_action_id", item.get("action_id")) == 9
            for item in self.transitions
        )
        model_stop_in_goal = sum(
            item.get("policy_action_id", item.get("action_id")) == 9
            and bool(item.get("in_goal_radius", False))
            for item in self.transitions
        )
        entered_goal = goal_dwell_decisions > 0
        goal_escape_after_entry = False
        seen_goal = False
        for item in self.transitions:
            if item.get("in_goal_radius", False):
                seen_goal = True
            elif seen_goal:
                goal_escape_after_entry = True
        return {
            "success": success,
            "policy_success": policy_success,
            "system_success": system_success,
            "total_high_level_reward": sum(item.get("reward", 0.0) for item in self.transitions),
            "cumulative_cost": cumulative_cost,
            "constraint_cost": constraint_cost,
            "cost_normalization": COST_NORMALIZATION,
            "cumulative_hard_cost": sum(item.get("hard_cost", 0.0) for item in self.transitions),
            "cumulative_dense_cost": sum(item.get("dense_cost", 0.0) for item in self.transitions),
            "hard_violation_count": sum(bool(item.get("hard_violation", False)) for item in self.transitions),
            "unsafe_contact_count": sum(item.get("cost_components", {}).get("unsafe_contact", 0.0) for item in self.transitions),
            "fall_count": sum(item.get("cost_components", {}).get("fall", 0.0) for item in self.transitions),
            "blocked_count": sum(item.get("cost_components", {}).get("blocked", 0.0) for item in self.transitions),
            "turn_tracking_failure_count": sum(
                item.get("cost_components", {}).get(
                    "turn_tracking_failure",
                    item.get("cost_components", {}).get("turn_blocked", 0.0),
                )
                for item in self.transitions
            ),
            "turn_blocked_count": sum(
                item.get("cost_components", {}).get(
                    "turn_tracking_failure",
                    item.get("cost_components", {}).get("turn_blocked", 0.0),
                )
                for item in self.transitions
            ),
            "has_collision": any(item.get("cost_components", {}).get("unsafe_contact", 0.0) > 0 for item in self.transitions),
            "has_blocked": any(item.get("cost_components", {}).get("blocked", 0.0) > 0 for item in self.transitions),
            "has_turn_tracking_failure": any(
                item.get("cost_components", {}).get(
                    "turn_tracking_failure",
                    item.get("cost_components", {}).get("turn_blocked", 0.0),
                )
                > 0
                for item in self.transitions
            ),
            "has_turn_blocked": any(
                item.get("cost_components", {}).get(
                    "turn_tracking_failure",
                    item.get("cost_components", {}).get("turn_blocked", 0.0),
                )
                > 0
                for item in self.transitions
            ),
            "num_macro_actions": len(self.transitions),
            "invalid_action_count": sum(bool(item.get("invalid_action")) for item in self.transitions),
            "entered_goal_radius": entered_goal,
            "minimum_distance_to_goal_m": (
                min(valid_distances) if valid_distances else None
            ),
            "goal_dwell_decisions": goal_dwell_decisions,
            "oracle_stop_decisions": oracle_stop_decisions,
            "model_stop_decisions": model_stop_decisions,
            "missed_stop_count": sum(
                bool(item.get("missed_stop", False)) for item in self.transitions
            ),
            "stop_recall_in_goal": (
                model_stop_in_goal / oracle_stop_decisions
                if oracle_stop_decisions
                else None
            ),
            "shield_intervention_count": sum(
                bool(item.get("shield_intervened", False))
                for item in self.transitions
            ),
            "goal_gate_intervention_count": sum(
                bool(item.get("goal_gate_intervened", item.get("shield_intervened", False)))
                for item in self.transitions
            ),
            "goal_escape_after_entry": goal_escape_after_entry,
            "constraint_satisfied": constraint_satisfied,
            "safe_success": success and constraint_satisfied,
            "safe_spl": spl if constraint_satisfied else 0.0,
            "termination_reason": last_reason,
        }

    def to_dict(self, measurements: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objective_config": deepcopy(self.objective_config) or None,
            "objective_fingerprint": self.objective_config.get("fingerprint"),
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "instruction": self.instruction,
            "gamma": self.gamma,
            "summary": self.summary(measurements),
            "transitions": deepcopy(self.transitions),
        }
