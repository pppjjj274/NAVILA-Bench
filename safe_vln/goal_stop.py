"""Goal-region stop semantics shared by Safe-VLN collection and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .actions import action_from_id


GOAL_STOP_MODES = ("policy", "shield")
COLLECTION_POLICIES = ("vlm", "oracle")


@dataclass(frozen=True)
class GoalStopDecision:
    """Result of applying the goal-stop contract to one proposed action."""

    policy_action_id: int
    executed_action_id: int
    in_goal_radius: bool
    missed_stop: bool
    consecutive_missed_stops: int
    shield_intervened: bool
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
    ) -> GoalStopDecision:
        proposed = action_from_id(action_id).action_id
        distance = float(distance_to_goal)
        distance_valid = bool(
            navigation_reward_valid and math.isfinite(distance) and distance >= 0
        )
        in_goal = bool(distance_valid and distance <= self.goal_radius)

        if not in_goal:
            self.consecutive_missed_stops = 0

        if proposed == 9:
            self.consecutive_missed_stops = 0
            success = in_goal
            return GoalStopDecision(
                policy_action_id=proposed,
                executed_action_id=proposed,
                in_goal_radius=in_goal,
                missed_stop=False,
                consecutive_missed_stops=0,
                shield_intervened=False,
                immediate_terminal=True,
                terminate_after_execution=False,
                success=success,
                failed_stop=not success,
                termination_reason="success" if success else "failed_stop",
            )

        missed_stop = in_goal
        if missed_stop:
            self.consecutive_missed_stops += 1

        if self.mode == "shield" and missed_stop:
            return GoalStopDecision(
                policy_action_id=proposed,
                executed_action_id=9,
                in_goal_radius=True,
                missed_stop=True,
                consecutive_missed_stops=self.consecutive_missed_stops,
                shield_intervened=True,
                immediate_terminal=True,
                terminate_after_execution=False,
                success=True,
                failed_stop=False,
                termination_reason="shield_success",
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
            missed_stop=missed_stop,
            consecutive_missed_stops=self.consecutive_missed_stops,
            shield_intervened=False,
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
