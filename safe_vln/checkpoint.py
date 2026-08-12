"""Checkpoint contracts for Go2 locomotion and Safe-VLN policies."""

from __future__ import annotations

from os import PathLike
from typing import Any, Mapping

import torch


_LEGACY_MISSING_PREFIXES = ("cost_critic.",)

SAFE_CHECKPOINT_CONTRACT_VERSION = 1
CHECKPOINT_ROLE_CRITIC_ONLY = "critic-only"
CHECKPOINT_ROLE_POLICY = "policy"
CHECKPOINT_ROLE_DIAGNOSTIC = "diagnostic"
SAFE_CHECKPOINT_ROLES = frozenset(
    {
        CHECKPOINT_ROLE_CRITIC_ONLY,
        CHECKPOINT_ROLE_POLICY,
        CHECKPOINT_ROLE_DIAGNOSTIC,
    }
)
POLICY_INTERFACE_NAVILA_GREEDY = "navila-greedy-text-v1"
POLICY_INTERFACE_SAFE_DISCRETE = "safe-vln-discrete-v1"


def _independent_actor_audit(state: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint contains evidence of a held-out audit."""

    calibration_values = state.get("calibration_episode_ids")
    audit_values = state.get("audit_episode_ids")
    if not isinstance(calibration_values, (list, tuple)) or not isinstance(
        audit_values, (list, tuple)
    ):
        return False
    calibration_ids = {str(value) for value in calibration_values}
    audit_ids = {str(value) for value in audit_values}
    return bool(
        state.get("actor/audit_independent") is True
        and calibration_ids
        and audit_ids
        and calibration_ids.isdisjoint(audit_ids)
    )


def safe_checkpoint_contract(state: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and validate the lifecycle role of a Safe-VLN checkpoint.

    Old critic warmups did not record a role.  They are interpreted as
    ``critic-only`` so their untrained candidate scorer can never silently
    replace the original NaViLA policy.  Old accepted Actors are recognized as
    policies only when their state proves that calibration and audit episodes
    were disjoint; all other legacy Actor checkpoints are diagnostic.
    """

    version = state.get("checkpoint_contract_version")
    role = state.get("checkpoint_role")
    interface = state.get("policy_interface")
    independent = _independent_actor_audit(state)

    if version is None and role is None:
        if state.get("actor/accepted") is True and independent:
            role = CHECKPOINT_ROLE_POLICY
            interface = interface or POLICY_INTERFACE_SAFE_DISCRETE
        elif state.get("actor/accepted") is None:
            role = CHECKPOINT_ROLE_CRITIC_ONLY
            interface = interface or POLICY_INTERFACE_NAVILA_GREEDY
        else:
            role = CHECKPOINT_ROLE_DIAGNOSTIC
            interface = interface or POLICY_INTERFACE_SAFE_DISCRETE
        version = 0
    else:
        if int(version) != SAFE_CHECKPOINT_CONTRACT_VERSION:
            raise RuntimeError(
                "unsupported Safe-VLN checkpoint contract version: "
                f"{version!r}"
            )
        if role not in SAFE_CHECKPOINT_ROLES:
            raise RuntimeError(f"invalid Safe-VLN checkpoint role: {role!r}")

    expected_interface = (
        POLICY_INTERFACE_NAVILA_GREEDY
        if role == CHECKPOINT_ROLE_CRITIC_ONLY
        else POLICY_INTERFACE_SAFE_DISCRETE
    )
    if interface != expected_interface:
        raise RuntimeError(
            f"checkpoint role {role!r} requires policy_interface="
            f"{expected_interface!r}, got {interface!r}"
        )
    if role == CHECKPOINT_ROLE_POLICY:
        if state.get("actor/accepted") is not True:
            raise RuntimeError("policy checkpoint has no accepted Actor audit")
        if not independent:
            raise RuntimeError("policy checkpoint has no independent Actor audit")
        audit_target = state.get("actor/audit_target_source")
        if audit_target not in {"dynamic-oracle", "original-navila-policy"}:
            raise RuntimeError(
                "policy checkpoint has no recognized independent Actor audit target"
            )
        minimum_motion = float(
            state.get("actor/minimum_non_stop_macro_accuracy", 0.0) or 0.0
        )
        if minimum_motion <= 0.0:
            raise RuntimeError(
                "policy checkpoint Actor audit used a zero motion threshold"
            )
        if state.get("actor/goal_stop_contract") != "sensor-gated-v1":
            minimum_stop = float(
                state.get("actor/minimum_stop_accuracy", 0.0) or 0.0
            )
            if minimum_stop <= 0.0:
                raise RuntimeError(
                    "policy checkpoint Actor audit used a zero STOP threshold"
                )
    return {
        "checkpoint_contract_version": int(version),
        "checkpoint_role": role,
        "policy_interface": interface,
        "actor_audit_independent": independent,
    }


def require_safe_policy_checkpoint(
    state: Mapping[str, Any],
    *,
    context: str,
    allow_diagnostic: bool = False,
) -> dict[str, Any]:
    """Require a deployable discrete policy, failing closed by default."""

    contract = safe_checkpoint_contract(state)
    if contract["checkpoint_role"] == CHECKPOINT_ROLE_POLICY:
        return contract
    if allow_diagnostic and contract["checkpoint_role"] == CHECKPOINT_ROLE_DIAGNOSTIC:
        return contract
    raise RuntimeError(
        f"{context} requires an independently audited Safe-VLN policy checkpoint; "
        f"got role={contract['checkpoint_role']!r} with "
        f"policy_interface={contract['policy_interface']!r}"
    )


def load_go2_inference_checkpoint(
    runner: Any,
    checkpoint_path: str | PathLike[str],
    *,
    map_location: Any = None,
) -> Any:
    """Load a Go2 policy for inference without restoring its optimizer.

    Legacy locomotion checkpoints predate the lower-level cost critic. They
    are safe to use for the frozen actor policy, so this loader permits only
    missing ``cost_critic.*`` parameters. Every other missing or unexpected
    model parameter remains a hard error.

    Returns:
        The optional ``infos`` object stored in the checkpoint.
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid Go2 checkpoint {checkpoint_path!s}: expected a dictionary.")

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise RuntimeError(
            f"Invalid Go2 checkpoint {checkpoint_path!s}: missing dictionary 'model_state_dict'."
        )

    actor_critic = runner.alg.actor_critic
    expected_keys = set(actor_critic.state_dict())
    checkpoint_keys = set(model_state)
    missing_keys = sorted(expected_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - expected_keys)
    invalid_missing_keys = [
        key for key in missing_keys if not key.startswith(_LEGACY_MISSING_PREFIXES)
    ]

    if invalid_missing_keys or unexpected_keys:
        raise RuntimeError(
            "Incompatible Go2 inference checkpoint: "
            f"missing={invalid_missing_keys}, unexpected={unexpected_keys}. "
            "Only missing cost_critic.* parameters are allowed for legacy checkpoints."
        )

    # strict=False is narrowly justified by the key validation above. Shape
    # mismatches still raise from PyTorch instead of being silently ignored.
    actor_critic.load_state_dict(model_state, strict=False)

    if getattr(runner, "empirical_normalization", False):
        normalizer_keys = ("obs_norm_state_dict", "critic_obs_norm_state_dict")
        absent_normalizers = [key for key in normalizer_keys if key not in checkpoint]
        if absent_normalizers:
            raise RuntimeError(
                "Incompatible Go2 inference checkpoint: missing normalization states "
                f"{absent_normalizers}."
            )
        runner.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(checkpoint["critic_obs_norm_state_dict"])

    runner.current_learning_iteration = checkpoint.get("iter", 0)

    if missing_keys:
        print(
            "[WARNING] Loaded a legacy Go2 inference checkpoint without lower-level "
            f"cost critic parameters: {missing_keys}. The initialized cost critic must "
            "not be used as a trained safety value model."
        )
    else:
        print("[INFO] Loaded complete Go2 inference checkpoint.")

    return checkpoint.get("infos")
