"""Goal-region stop semantics shared by Safe-VLN collection and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .actions import action_from_id


GOAL_STOP_MODES = ("policy", "shield", "sensor-gated")
COLLECTION_POLICIES = ("vlm", "oracle")


@dataclass(frozen=True)
class GoalStopDecision:
    """Result of applying the goal-stop contract to one proposed action."""

    policy_action_id: int
    executed_action_id: int
    in_goal_radius: bool
    goal_distance_valid: bool
    missed_stop: bool
    consecutive_missed_stops: int
    shield_intervened: bool
    goal_gate_reason: str | None
    immediate_terminal: bool
    terminate_after_execution: bool
    success: bool
    failed_stop: bool
    termination_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class GoalStopController:
    """Make stop behavior explicit without contaminating raw-policy metrics.

    ``policy`` mode executes exactly the proposed action.  During training only,
    three consecutive non-stop decisions inside the goal radius terminate the
    episode after the third macro action, preventing an endlessly looping
    rollout while keeping that third action observable.

    ``shield`` mode still queries and records the policy, but immediately
    executes STOP when a non-stop action is proposed inside the goal radius.
    Such interventions are system successes, not policy successes.
    """

    def __init__(
        self,
        *,
        goal_radius: float,
        mode: str = "policy",
        dataset_role: str = "train",
        missed_stop_patience: int = 3,
    ) -> None:
        radius = float(goal_radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("goal_radius must be finite and positive")
        if mode not in GOAL_STOP_MODES:
            raise ValueError(f"goal-stop mode must be one of {GOAL_STOP_MODES}")
        if dataset_role not in {"train", "eval"}:
            raise ValueError("dataset_role must be 'train' or 'eval'")
        if isinstance(missed_stop_patience, bool) or int(missed_stop_patience) <= 0:
            raise ValueError("missed_stop_patience must be a positive integer")
        self.goal_radius = radius
        self.mode = mode
        self.dataset_role = dataset_role
        self.missed_stop_patience = int(missed_stop_patience)
        self.consecutive_missed_stops = 0

    def resolve(
        self,
        action_id: int,
        distance_to_goal: float,
        *,
        navigation_reward_valid: bool,
        action_probabilities=None,
    ) -> GoalStopDecision:
        proposed = action_from_id(action_id).action_id
        distance = float(distance_to_goal)
        distance_valid = bool(
            navigation_reward_valid and math.isfinite(distance) and distance >= 0
        )
        in_goal = bool(distance_valid and distance <= self.goal_radius)

        if self.mode == "sensor-gated" and not distance_valid:
            self.consecutive_missed_stops = 0
            return GoalStopDecision(
                policy_action_id=proposed,
                executed_action_id=9,
                in_goal_radius=False,
                goal_distance_valid=False,
                missed_stop=False,
                consecutive_missed_stops=0,
                shield_intervened=True,
                goal_gate_reason="goal_distance_unavailable",
                immediate_terminal=True,
                terminate_after_execution=False,
                success=False,
                failed_stop=False,
                termination_reason="goal_distance_unavailable",
            )

        if not in_goal:
            self.consecutive_missed_stops = 0

        if proposed == 9:
            self.consecutive_missed_stops = 0
            success = in_goal
            if self.mode == "sensor-gated" and not success:
                probabilities = action_probabilities
                valid = bool(
                    isinstance(probabilities, (list, tuple))
                    and len(probabilities) == 10
                    and all(
                        not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) >= 0.0
                        for value in probabilities
                    )
                    and sum(float(value) for value in probabilities[:9]) > 0.0
                )
                fallback = (
                    max(range(9), key=lambda index: float(probabilities[index]))
                    if valid
                    else 9
                )
                return GoalStopDecision(
                    policy_action_id=proposed,
                    executed_action_id=fallback,
                    in_goal_radius=False,
                    goal_distance_valid=True,
                    missed_stop=False,
                    consecutive_missed_stops=0,
                    shield_intervened=True,
                    goal_gate_reason=(
                        "premature_stop_rejected"
                        if valid
                        else "goal_gate_no_fallback"
                    ),
                    immediate_terminal=not valid,
                    terminate_after_execution=False,
                    success=False,
                    # With a valid goal distance and no executable fallback,
                    # this is the same premature STOP failure as policy mode.
                    failed_stop=not valid,
                    termination_reason=(None if valid else "goal_gate_no_fallback"),
                )
            return GoalStopDecision(
                policy_action_id=proposed,
                executed_action_id=proposed,
                in_goal_radius=in_goal,
                goal_distance_valid=distance_valid,
                missed_stop=False,
                consecutive_missed_stops=0,
                shield_intervened=False,
                goal_gate_reason=None,
                immediate_terminal=True,
                terminate_after_execution=False,
                success=success,
                failed_stop=not success,
                termination_reason="success" if success else "failed_stop",
            )

        missed_stop = in_goal
        if missed_stop:
            self.consecutive_missed_stops += 1

        if self.mode in {"shield", "sensor-gated"} and missed_stop:
            return GoalStopDecision(
                policy_action_id=proposed,
                executed_action_id=9,
                in_goal_radius=True,
                goal_distance_valid=True,
                missed_stop=True,
                consecutive_missed_stops=self.consecutive_missed_stops,
                shield_intervened=True,
                goal_gate_reason="goal_radius_stop",
                immediate_terminal=True,
                terminate_after_execution=False,
                success=True,
                failed_stop=False,
                termination_reason=(
                    "sensor_gated_success"
                    if self.mode == "sensor-gated"
                    else "shield_success"
                ),
            )

        patience_reached = bool(
            self.dataset_role == "train"
            and missed_stop
            and self.consecutive_missed_stops >= self.missed_stop_patience
        )
        return GoalStopDecision(
            policy_action_id=proposed,
            executed_action_id=proposed,
            in_goal_radius=in_goal,
            goal_distance_valid=distance_valid,
            missed_stop=missed_stop,
            consecutive_missed_stops=self.consecutive_missed_stops,
            shield_intervened=False,
            goal_gate_reason=None,
            immediate_terminal=False,
            terminate_after_execution=patience_reached,
            success=False,
            failed_stop=False,
            termination_reason=(
                "missed_stop_patience" if patience_reached else None
            ),
        )


__all__ = [
    "COLLECTION_POLICIES",
    "GOAL_STOP_MODES",
    "GoalStopController",
    "GoalStopDecision",
]
