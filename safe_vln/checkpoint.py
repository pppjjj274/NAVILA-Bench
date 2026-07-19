"""Backward-compatible checkpoint loading for frozen Go2 inference policies."""

from __future__ import annotations

from os import PathLike
from typing import Any

import torch


_LEGACY_MISSING_PREFIXES = ("cost_critic.",)


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
