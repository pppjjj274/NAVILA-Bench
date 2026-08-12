"""Executable critic warm-start and constrained PPO passes."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .actor_training import split_actor_partitions
from .actions import has_valid_policy_statistics
from .checkpoint import (
    CHECKPOINT_ROLE_DIAGNOSTIC,
    CHECKPOINT_ROLE_POLICY,
    POLICY_INTERFACE_SAFE_DISCRETE,
    SAFE_CHECKPOINT_CONTRACT_VERSION,
    require_safe_policy_checkpoint,
    safe_checkpoint_contract,
)
from .cmdp import LagrangeController, compute_gae
from .dataset import iter_sample_refs, iter_samples, load_sample_refs
from .learner import evaluate_selected_actions, save_checkpoint, train_critic_epoch
from .live_render import LEGACY_LIVE_SCHEMA_VERSIONS, LIVE_SCHEMA_VERSION
from .navila import load_safe_navila
from .objective import (
    COST_NORMALIZATION,
    SCHEMA_VERSION,
    validate_objective_config,
)
from .sampling import (
    deterministic_shuffle,
    sampling_summary,
    select_balanced_critic,
    select_balanced_oracle,
    select_balanced_ppo,
)
from .trainer import PPOConfig, SafePPOOptimizer, normalize_advantage


VERSIONED_DATASET_SCHEMAS = {
    "safe-vln-go2-v2",
    *LEGACY_LIVE_SCHEMA_VERSIONS,
    LIVE_SCHEMA_VERSION,
}


def _read_json(path):
    resolved = Path(path)
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {resolved}")
    return payload


def _dataset_manifest(dataset_dir):
    manifest = _read_json(Path(dataset_dir) / "manifest.json")
    if not manifest:
        raise RuntimeError(f"Safe-VLN dataset has no manifest: {dataset_dir}")
    return manifest


def _checkpoint_state(checkpoint):
    return _read_json(Path(checkpoint) / "trainer_state.json") if checkpoint else {}


def _require_new_output_dir(output_dir):
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite training output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reject_failed_actor_audit(checkpoint_state):
    """Backward-compatible name for the fail-closed policy gate."""

    return require_safe_policy_checkpoint(
        checkpoint_state, context="Safe-VLN policy training"
    )


def _copy_actor_contract(target, checkpoint_state):
    for key, value in checkpoint_state.items():
        if key.startswith("actor/") or key in {
            "actor_architecture",
            "audit_episode_ids",
            "calibration_episode_ids",
            "checkpoint_contract_version",
            "checkpoint_role",
            "policy_interface",
            "stop_threshold",
        }:
            target[key] = value


def _validate_dataset_objective(manifest):
    schema = manifest.get("schema_version")
    dataset_fingerprint = manifest.get("objective_fingerprint")
    if schema in VERSIONED_DATASET_SCHEMAS:
        if not dataset_fingerprint:
            raise RuntimeError("versioned rollout manifest has no objective fingerprint")
        objective_config = manifest.get("objective_config")
        if not objective_config:
            raise RuntimeError("versioned rollout manifest has no objective configuration")
        try:
            validated_objective = validate_objective_config(objective_config)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("versioned rollout objective configuration is invalid") from error
        if validated_objective["fingerprint"] != dataset_fingerprint:
            raise RuntimeError(
                "versioned rollout manifest fingerprint does not match its objective"
            )
        if (
            schema == LIVE_SCHEMA_VERSION
            and validated_objective.get("schema_version") != SCHEMA_VERSION
        ):
            raise RuntimeError(
                f"{LIVE_SCHEMA_VERSION} data requires the current "
                f"{SCHEMA_VERSION} objective contract"
            )
        return validated_objective
    if dataset_fingerprint:
        raise RuntimeError("legacy v1 data cannot be mixed with a versioned objective")
    return None


def _validate_objective_compatibility(manifest, checkpoint_state):
    schema = manifest.get("schema_version")
    dataset_fingerprint = manifest.get("objective_fingerprint")
    checkpoint_fingerprint = checkpoint_state.get("objective_fingerprint")
    _validate_dataset_objective(manifest)
    if schema in VERSIONED_DATASET_SCHEMAS:
        if checkpoint_fingerprint != dataset_fingerprint:
            raise RuntimeError(
                "checkpoint and rollout objective fingerprints do not match"
            )
    elif checkpoint_fingerprint:
        raise RuntimeError("legacy v1 data cannot be mixed with a versioned objective")
    return schema, dataset_fingerprint


def _training_dtype(args):
    name = getattr(args, "training_dtype", "bfloat16")
    if name == "bfloat16":
        if str(args.device).startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16")
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported training dtype: {name}")


def _optimizer(model, actor_lr, critic_lr):
    actor = [parameter for parameter in model.base_model.parameters() if parameter.requires_grad]
    if getattr(model, "actor_head", None) is not None:
        actor.extend(
            parameter
            for parameter in model.actor_head.parameters()
            if parameter.requires_grad
        )
    critics = list(model.reward_head.parameters()) + list(model.cost_head.parameters())
    groups = []
    if actor:
        groups.append({"params": actor, "lr": actor_lr})
    groups.append({"params": critics, "lr": critic_lr})
    return torch.optim.AdamW(groups)


def _enable_gradient_checkpointing(model):
    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        try:
            model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.base_model.gradient_checkpointing_enable()


def _initial_lagrange_multiplier(args, checkpoint_state):
    explicit = getattr(args, "initial_lagrange_multiplier", None)
    value = (
        explicit
        if explicit is not None
        else checkpoint_state.get("lagrange_multiplier", 0.001)
    )
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("initial Lagrange multiplier must be finite and non-negative")
    return value


def _update_lagrange_for_rollout(controller, mean_episode_cost):
    """Apply exactly one dual update for one immutable rollout batch."""

    before = float(controller.multiplier)
    after = float(controller.update(mean_episode_cost))
    return before, after


def _has_verified_safety_observation(metadata) -> bool:
    """Reject rollouts collected before contact/turn sensors were validated."""
    diagnostics = metadata.get("safety_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    if diagnostics.get("contact_sensor_enabled") is not True:
        return False
    action_id = metadata.get("action_id")
    try:
        action_id = None if action_id is None else int(action_id)
    except (TypeError, ValueError, OverflowError):
        return False
    if action_id is not None and 0 <= action_id <= 8:
        if not isinstance(diagnostics.get("turn_execution"), dict):
            return False
    return True


def _warmup_actor_candidate(args):
    """Legacy candidate-scoring behavior cloning on strictly aligned dynamic-oracle samples."""
    if getattr(args, "actor_target_source", "oracle") != "oracle":
        raise ValueError(
            "--actor-target-source=navila-policy requires a hierarchical Actor"
        )
    if getattr(args, "checkpoint", None):
        raise ValueError("warmup-actor starts from fresh NaViLA LoRA; omit --checkpoint")
    _require_new_output_dir(args.output_dir)
    if args.actor_lr <= 0:
        raise ValueError("--actor-lr must be positive")
    if args.oracle_stop_weight <= 0:
        raise ValueError("--oracle-stop-weight must be positive")
    if args.mini_batch_size <= 0:
        raise ValueError("--mini-batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    for name in (
        "minimum_stop_accuracy",
        "minimum_non_stop_macro_accuracy",
    ):
        value = float(getattr(args, name, 0.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    manifest = _dataset_manifest(args.dataset_dir)
    if manifest.get("dataset_role") != "train":
        raise RuntimeError("actor warmup requires a training dataset")
    if manifest.get("schema_version") != LIVE_SCHEMA_VERSION:
        raise RuntimeError(
            f"actor warmup requires {LIVE_SCHEMA_VERSION} strict live-render data"
        )
    _validate_dataset_objective(manifest)
    objective_fingerprint = manifest.get("objective_fingerprint")
    if not objective_fingerprint:
        raise RuntimeError("actor warmup dataset has no objective fingerprint")

    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        dtype=_training_dtype(args),
        checkpoint=None,
    )
    model.train()
    model.reward_head.requires_grad_(False)
    model.cost_head.requires_grad_(False)
    _enable_gradient_checkpointing(model)
    actor_parameters = [
        parameter
        for parameter in model.base_model.parameters()
        if parameter.requires_grad
    ]
    if not actor_parameters:
        raise RuntimeError("fresh NaViLA model has no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(actor_parameters, lr=args.actor_lr)

    def validate_metadata(metadata):
        if metadata.get("schema_version") != LIVE_SCHEMA_VERSION:
            raise RuntimeError("actor warmup sample schema does not match manifest")
        if metadata.get("objective_fingerprint") != objective_fingerprint:
            raise RuntimeError(
                "actor warmup sample objective fingerprint does not match manifest"
            )
        if not _has_verified_safety_observation(metadata):
            return False
        if not bool(metadata.get("oracle_eligible", False)):
            return False
        oracle_action_id = metadata.get("oracle_action_id")
        if oracle_action_id is None:
            return False
        return True

    eligible_refs = [
        ref
        for ref in iter_sample_refs(args.dataset_dir, args.split)
        if validate_metadata(ref.metadata)
    ]
    if not eligible_refs:
        raise RuntimeError("actor warmup found no oracle-eligible samples")
    audit_episodes_per_scene = (
        args.dev_episodes_per_scene
        if getattr(args, "dev_episodes_per_scene", None) is not None
        else args.audit_episodes_per_scene
    )
    train_refs, calibration_refs, audit_refs = split_actor_partitions(
        eligible_refs,
        seed=int(getattr(args, "sampling_seed", 20260729)),
        calibration_episodes_per_scene=args.calibration_episodes_per_scene,
        audit_episodes_per_scene=audit_episodes_per_scene,
    )

    sampling_strategy = getattr(args, "sampling_strategy", "sequential")
    sampling_seed = int(getattr(args, "sampling_seed", 20260729))
    if sampling_strategy == "balanced-oracle":
        selected_refs = select_balanced_oracle(
            train_refs,
            max_samples=args.max_samples,
            seed=sampling_seed,
        )
    elif sampling_strategy == "sequential":
        selected_refs = train_refs[
            : args.max_samples if args.max_samples is not None else None
        ]
    else:
        raise ValueError(
            "warmup-actor supports --sampling-strategy=sequential or "
            "balanced-oracle"
        )
    samples = load_sample_refs(selected_refs)
    if not samples:
        raise RuntimeError("actor warmup selected no training samples")
    sample_stats = sampling_summary(samples)
    print(
        json.dumps(
            {
                "mode": "warmup-actor-sampling",
                "strategy": sampling_strategy,
                "seed": sampling_seed,
                **sample_stats,
            }
        ),
        flush=True,
    )

    state = {
        "mode": "warmup-actor",
        "training_dtype": getattr(args, "training_dtype", "bfloat16"),
        "schema_version": LIVE_SCHEMA_VERSION,
        "objective_fingerprint": objective_fingerprint,
        "objective_config": manifest.get("objective_config"),
        "policy_version": 0,
        "fresh_lora": True,
        "actor_architecture": "candidate-scoring",
        "oracle_stop_weight": float(args.oracle_stop_weight),
        "sampling_strategy": sampling_strategy,
        "sampling_seed": sampling_seed,
        "sampling": sample_stats,
        "train_episodes": len(
            {str(ref.metadata.get("episode_id")) for ref in train_refs}
        ),
        "calibration_episodes": len(
            {str(ref.metadata.get("episode_id")) for ref in calibration_refs}
        ),
        "audit_episodes": len(
            {str(ref.metadata.get("episode_id")) for ref in audit_refs}
        ),
        "calibration_episode_ids": sorted(
            {str(ref.metadata.get("episode_id")) for ref in calibration_refs}
        ),
        "audit_episode_ids": sorted(
            {str(ref.metadata.get("episode_id")) for ref in audit_refs}
        ),
    }
    update = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_correct = 0
        epoch_stop_correct = 0
        epoch_stop_samples = 0
        epoch_samples = deterministic_shuffle(
            samples,
            seed=sampling_seed + epoch,
            namespace="actor-epoch",
        )
        for start in range(0, len(epoch_samples), args.mini_batch_size):
            mini_batch = epoch_samples[start : start + args.mini_batch_size]
            weights = [
                float(args.oracle_stop_weight)
                if int(metadata["oracle_action_id"]) == 9
                else 1.0
                for _, metadata in mini_batch
            ]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for (frames, metadata), weight in zip(mini_batch, weights):
                prepared = preprocessor(frames, metadata["instruction"])
                output = model(prepared.input_ids, images=prepared.images)
                target = torch.tensor(
                    [int(metadata["oracle_action_id"])],
                    dtype=torch.long,
                    device=output.action_logits.device,
                )
                loss = F.cross_entropy(output.action_logits, target)
                weighted_loss = loss * (weight / len(mini_batch))
                weighted_loss.backward()
                batch_loss += float(weighted_loss.detach().item())
                prediction = int(output.action_logits.detach().argmax(dim=-1).item())
                target_id = int(target.item())
                epoch_correct += int(prediction == target_id)
                if target_id == 9:
                    epoch_stop_samples += 1
                    epoch_stop_correct += int(prediction == 9)
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, 0.5)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("actor warmup gradient norm is not finite")
            optimizer.step()
            update += 1
            epoch_batches += 1
            epoch_loss += batch_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state.update(
            {
                "epoch": epoch + 1,
                "updates": update,
                "actor/loss": epoch_loss / max(epoch_batches, 1),
                "actor/samples": len(samples),
                "actor/accuracy": epoch_correct / len(samples),
                "actor/stop_samples": epoch_stop_samples,
                "actor/stop_accuracy": (
                    epoch_stop_correct / epoch_stop_samples
                    if epoch_stop_samples
                    else None
                ),
            }
        )
        print(json.dumps(state), flush=True)

    model.eval()
    class_samples: dict[int, int] = defaultdict(int)
    class_correct: dict[int, int] = defaultdict(int)
    audit_samples = load_sample_refs(audit_refs)
    with torch.no_grad():
        for frames, metadata in audit_samples:
            prepared = preprocessor(frames, metadata["instruction"])
            output = model(prepared.input_ids, images=prepared.images)
            target_id = int(metadata["oracle_action_id"])
            prediction = int(output.action_logits.argmax(dim=-1).item())
            class_samples[target_id] += 1
            class_correct[target_id] += int(prediction == target_id)
    per_class_accuracy = {
        str(action_id): class_correct[action_id] / count
        for action_id, count in sorted(class_samples.items())
    }
    non_stop_accuracies = [
        accuracy
        for action_id, accuracy in per_class_accuracy.items()
        if int(action_id) != 9
    ]
    stop_accuracy = per_class_accuracy.get("9")
    non_stop_macro_accuracy = (
        sum(non_stop_accuracies) / len(non_stop_accuracies)
        if non_stop_accuracies
        else None
    )
    minimum_stop_accuracy = float(
        getattr(args, "minimum_stop_accuracy", 0.5)
    )
    minimum_non_stop_macro_accuracy = float(
        getattr(args, "minimum_non_stop_macro_accuracy", 0.4)
    )
    all_action_classes_present = all(
        str(action_id) in per_class_accuracy for action_id in range(10)
    )
    accepted = (
        all_action_classes_present
        and stop_accuracy is not None
        and stop_accuracy >= minimum_stop_accuracy
        and non_stop_macro_accuracy is not None
        and non_stop_macro_accuracy >= minimum_non_stop_macro_accuracy
    )
    state.update(
        {
            "actor/audit_class_samples": {
                str(action_id): count
                for action_id, count in sorted(class_samples.items())
            },
            "actor/audit_per_class_accuracy": per_class_accuracy,
            "actor/audit_stop_accuracy": stop_accuracy,
            "actor/audit_non_stop_macro_accuracy": non_stop_macro_accuracy,
            "actor/audit_samples": len(audit_samples),
            "actor/audit_independent": True,
            "actor/audit_target_source": "dynamic-oracle",
            "actor/goal_stop_contract": "policy-v1",
            "actor/audit_all_action_classes_present": all_action_classes_present,
            "actor/minimum_stop_accuracy": minimum_stop_accuracy,
            "actor/minimum_non_stop_macro_accuracy": (
                minimum_non_stop_macro_accuracy
            ),
            "actor/accepted": accepted,
            "checkpoint_contract_version": SAFE_CHECKPOINT_CONTRACT_VERSION,
            "checkpoint_role": (
                CHECKPOINT_ROLE_POLICY
                if accepted
                else CHECKPOINT_ROLE_DIAGNOSTIC
            ),
            "policy_interface": POLICY_INTERFACE_SAFE_DISCRETE,
        }
    )
    print(json.dumps({"mode": "warmup-actor-audit", **state}), flush=True)
    save_checkpoint(model, optimizer, args.output_dir, state)
    if not accepted:
        raise RuntimeError(
            "actor audit failed; checkpoint was saved for diagnosis but must not "
            "be used for critic warmup or PPO"
        )
    return 0


def warmup_actor(args):
    """Train either the v5 hierarchical actor or a legacy candidate actor."""
    architecture = getattr(
        args, "actor_architecture", "candidate-scoring"
    )
    if architecture == "candidate-scoring":
        return _warmup_actor_candidate(args)
    if architecture not in {
        "hierarchical-stop-motion",
        "hierarchical-stop-direction-magnitude",
    }:
        raise ValueError(f"unsupported actor architecture: {architecture}")
    manifest = _dataset_manifest(args.dataset_dir)
    if manifest.get("dataset_role") != "train":
        raise RuntimeError("actor warmup requires a training dataset")
    if manifest.get("schema_version") != LIVE_SCHEMA_VERSION:
        raise RuntimeError(
            f"hierarchical actor requires {LIVE_SCHEMA_VERSION} data"
        )
    _validate_dataset_objective(manifest)
    if not manifest.get("objective_fingerprint"):
        raise RuntimeError("actor warmup dataset has no objective fingerprint")
    from .actor_pipeline import warmup_hierarchical_actor

    return warmup_hierarchical_actor(
        args, manifest, _training_dtype(args)
    )


def audit_actor(args):
    """Run a read-only diagnostic audit for a hierarchical checkpoint."""
    manifest = _dataset_manifest(args.dataset_dir)
    if manifest.get("dataset_role") != "train":
        raise RuntimeError("actor audit requires a training dataset")
    if manifest.get("schema_version") != LIVE_SCHEMA_VERSION:
        raise RuntimeError(
            f"actor audit requires {LIVE_SCHEMA_VERSION} data"
        )
    _validate_dataset_objective(manifest)
    if not manifest.get("objective_fingerprint"):
        raise RuntimeError("actor audit dataset has no objective fingerprint")
    from .actor_pipeline import diagnose_hierarchical_actor

    return diagnose_hierarchical_actor(
        args, manifest, _training_dtype(args)
    )


def dagger_actor(args):
    """Fine-tune a certified Actor from strict on-policy recovery rollouts."""
    rollout_manifest = _dataset_manifest(args.rollout_dir)
    anchor_manifest = _dataset_manifest(args.anchor_dataset_dir)
    checkpoint_state = _checkpoint_state(args.checkpoint)
    if rollout_manifest.get("dataset_role") != "train":
        raise RuntimeError("online DAgger rollouts must be training data")
    if anchor_manifest.get("dataset_role") != "train":
        raise RuntimeError("online DAgger anchors must be training data")
    if rollout_manifest.get("schema_version") != LIVE_SCHEMA_VERSION:
        raise RuntimeError("online DAgger requires strict live-render rollouts")
    if anchor_manifest.get("schema_version") != LIVE_SCHEMA_VERSION:
        raise RuntimeError("online DAgger requires strict live-render anchors")
    _validate_objective_compatibility(rollout_manifest, checkpoint_state)
    _validate_objective_compatibility(anchor_manifest, checkpoint_state)
    from .actor_pipeline import online_dagger_actor

    return online_dagger_actor(args, _training_dtype(args))


def warmup(args):
    if getattr(args, "reset_critics", False) and not args.checkpoint:
        raise ValueError("--reset-critics requires --checkpoint")
    _require_new_output_dir(args.output_dir)
    manifest = _dataset_manifest(args.dataset_dir)
    if manifest.get("dataset_role") == "eval":
        raise RuntimeError("evaluation datasets cannot be used for critic warmup")
    checkpoint_state = _checkpoint_state(args.checkpoint)
    source_contract = safe_checkpoint_contract(checkpoint_state)
    if source_contract["checkpoint_role"] == CHECKPOINT_ROLE_DIAGNOSTIC:
        raise RuntimeError(
            "critic warmup refuses a diagnostic Actor checkpoint; start from "
            "original NaViLA/critic-only weights or an independently audited policy"
        )
    schema = manifest.get("schema_version")
    objective_fingerprint = manifest.get("objective_fingerprint")
    _validate_dataset_objective(manifest)
    if schema in VERSIONED_DATASET_SCHEMAS and not objective_fingerprint:
        raise RuntimeError("versioned warmup dataset has no objective fingerprint")
    if (
        schema in VERSIONED_DATASET_SCHEMAS
        and args.checkpoint
        and not getattr(args, "reset_critics", False)
        and checkpoint_state.get("objective_fingerprint") != objective_fingerprint
    ):
        raise RuntimeError(
            "checkpoint and warmup dataset objective fingerprints do not match; "
            "use --reset-critics when migrating an actor to a new objective"
        )
    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        dtype=_training_dtype(args),
        checkpoint=args.checkpoint,
        reset_critics=getattr(args, "reset_critics", False),
    )
    model.base_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(model.reward_head.parameters()) + list(model.cost_head.parameters()),
        lr=args.critic_lr,
    )
    state = {
        "mode": "warmup-critics",
        "training_dtype": getattr(args, "training_dtype", "bfloat16"),
        "schema_version": schema,
        "objective_fingerprint": objective_fingerprint,
        "objective_config": manifest.get("objective_config"),
        "policy_version": int(checkpoint_state.get("policy_version", 0)),
        "critics_reset": bool(getattr(args, "reset_critics", False)),
    }
    if "lagrange_multiplier" in checkpoint_state:
        state["lagrange_multiplier"] = float(
            checkpoint_state["lagrange_multiplier"]
        )
    _copy_actor_contract(state, checkpoint_state)
    state.update(
        {
            "checkpoint_contract_version": SAFE_CHECKPOINT_CONTRACT_VERSION,
            "checkpoint_role": source_contract["checkpoint_role"],
            "policy_interface": source_contract["policy_interface"],
        }
    )

    def validate_metadata(metadata):
        sample_schema = metadata.get("schema_version")
        valid_schema = (
            sample_schema in {None, "safe-vln-go2-v1"}
            if schema == "safe-vln-go2-v1"
            else sample_schema == schema
        )
        if not valid_schema:
            raise RuntimeError("warmup sample schema does not match manifest")
        if metadata.get("objective_fingerprint") != objective_fingerprint:
            raise RuntimeError(
                "warmup sample objective fingerprint does not match manifest"
            )
        if schema == LIVE_SCHEMA_VERSION and not _has_verified_safety_observation(metadata):
            return False
        return (
            metadata.get("reward_return") is not None
            and metadata.get("cost_return") is not None
            and (
                bool(metadata.get("reward_critic_eligible", True))
                or bool(metadata.get("cost_critic_eligible", True))
            )
        )

    sampling_strategy = getattr(args, "sampling_strategy", "sequential")
    sampling_seed = int(getattr(args, "sampling_seed", 20260729))
    selected_samples = None
    if sampling_strategy == "balanced-critic":
        refs = [
            ref
            for ref in iter_sample_refs(args.dataset_dir, args.split)
            if validate_metadata(ref.metadata)
        ]
        selected_refs = select_balanced_critic(
            refs,
            max_samples=args.max_samples,
            seed=sampling_seed,
        )
        selected_samples = load_sample_refs(selected_refs)
        state.update(
            {
                "sampling_strategy": sampling_strategy,
                "sampling_seed": sampling_seed,
                "sampling": sampling_summary(
                    selected_samples, action_field="action_id"
                ),
            }
        )
        print(
            json.dumps(
                {
                    "mode": "warmup-critics-sampling",
                    "strategy": sampling_strategy,
                    "seed": sampling_seed,
                    **state["sampling"],
                }
            ),
            flush=True,
        )
    elif sampling_strategy != "sequential":
        raise ValueError(
            "warmup-critics supports --sampling-strategy=sequential or "
            "balanced-critic"
        )

    def validated_samples():
        if selected_samples is not None:
            yield from selected_samples
            return
        for frames, metadata in iter_samples(args.dataset_dir, args.split):
            if validate_metadata(metadata):
                yield frames, metadata

    for epoch in range(args.epochs):
        stats = train_critic_epoch(
            model,
            validated_samples(),
            preprocessor,
            optimizer,
            max_samples=(
                None if selected_samples is not None else args.max_samples
            ),
        )
        if int(stats.get("critic/samples", 0)) <= 0:
            raise RuntimeError(
                "critic warmup found no eligible reward/cost targets"
            )
        state.update({"epoch": epoch + 1, **stats})
        print(json.dumps(state), flush=True)
    save_checkpoint(model, optimizer, args.output_dir, state)
    return 0


def _load_on_policy_samples(args):
    required = (
        "old_log_prob",
        "reward_value",
        "cost_value",
        "reward",
        "cost",
        "action_id",
        "episode_id",
        "index",
        "done",
    )
    episodes = defaultdict(list)
    for frames, metadata in iter_samples(args.rollout_dir, args.split):
        if any(metadata.get(key) is None for key in required):
            continue
        episodes[str(metadata["episode_id"])].append((frames, metadata))
    if not episodes:
        raise RuntimeError("no structured on-policy transitions in rollout dataset")

    samples = []
    episode_costs = {}
    for episode_id, episode_samples in episodes.items():
        episode_samples.sort(key=lambda sample: int(sample[1]["index"]))
        metadata = [sample[1] for sample in episode_samples]
        strict_v4 = getattr(args, "rollout_schema", None) == LIVE_SCHEMA_VERSION
        for item in metadata:
            for key in (
                "old_log_prob",
                "reward_value",
                "cost_value",
                "reward",
                "cost",
            ):
                try:
                    value = float(item[key])
                except (KeyError, TypeError, ValueError, OverflowError) as error:
                    raise RuntimeError(
                        f"rollout episode {episode_id} has invalid {key}"
                    ) from error
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"rollout episode {episode_id} has non-finite {key}"
                    )
                if key == "cost" and value < 0.0:
                    raise RuntimeError(
                        f"rollout episode {episode_id} has negative cost"
                    )
            action_id = item.get("action_id")
            index = item.get("index")
            if (
                isinstance(action_id, bool)
                or not isinstance(action_id, int)
                or not 0 <= action_id <= 9
            ):
                raise RuntimeError(
                    f"rollout episode {episode_id} has invalid action_id"
                )
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise RuntimeError(
                    f"rollout episode {episode_id} has invalid transition index"
                )
            if not isinstance(item.get("done"), bool):
                raise RuntimeError(
                    f"rollout episode {episode_id} has non-boolean done flag"
                )
            if strict_v4:
                policy_stats = {
                    **item,
                    "log_prob": item["old_log_prob"],
                }
                if not has_valid_policy_statistics(
                    policy_stats,
                    objective_fingerprint=getattr(
                        args, "rollout_objective_fingerprint", None
                    ),
                ):
                    raise RuntimeError(
                        f"rollout episode {episode_id} has inconsistent policy statistics"
                    )
        if strict_v4:
            unverified = [
                int(item["index"])
                for item in metadata
                if not _has_verified_safety_observation(item)
            ]
            if unverified:
                raise RuntimeError(
                    f"rollout episode {episode_id} contains unverified safety "
                    f"observations at indices {unverified[:5]}"
                )
            if any(bool(item.get("oracle_eligible", False)) for item in metadata):
                raise RuntimeError(
                    f"rollout episode {episode_id} contains privileged online "
                    "Oracle labels; collect mainline VLM data without "
                    "--allow-online-oracle"
                )
            indices = [int(item["index"]) for item in metadata]
            if indices != list(range(len(indices))) or not bool(metadata[-1]["done"]):
                raise RuntimeError(
                    f"v4 rollout episode {episode_id} is incomplete or non-contiguous"
                )
        episode_costs[episode_id] = sum(
            float(item["cost"]) for item in metadata
        )
        if any(
            not bool(
                item.get(
                    "ppo_eligible",
                    item.get("actor_eligible", not strict_v4),
                )
            )
            for item in metadata
        ):
            # A missing navigation reward invalidates the temporal GAE chain.
            # Preserve its episode cost for the constraint statistic, but do
            # not construct actor/reward advantages from a partial episode.
            continue
        dones = [bool(item["done"]) for item in metadata]
        reward_advantages, reward_returns = compute_gae(
            [item["reward"] for item in metadata],
            [item["reward_value"] for item in metadata],
            dones,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        cost_advantages, cost_returns = compute_gae(
            [item["cost"] for item in metadata],
            [item["cost_value"] for item in metadata],
            dones,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        for index, sample in enumerate(episode_samples):
            sample[1]["reward_advantage"] = float(reward_advantages[index])
            sample[1]["cost_advantage"] = float(cost_advantages[index])
            sample[1]["ppo_reward_return"] = float(reward_returns[index])
            sample[1]["ppo_cost_return"] = float(cost_returns[index])
            samples.append(sample)
    if not samples:
        raise RuntimeError("no actor-eligible complete episodes in rollout dataset")
    sampling_strategy = getattr(args, "sampling_strategy", "sequential")
    sampling_seed = int(getattr(args, "sampling_seed", 20260729))
    if sampling_strategy == "balanced-ppo":
        samples = select_balanced_ppo(
            samples,
            max_samples=args.max_samples,
            seed=sampling_seed,
        )
    elif sampling_strategy == "sequential":
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
    else:
        raise ValueError(
            "train supports --sampling-strategy=sequential or balanced-ppo"
        )
    if not samples:
        raise RuntimeError("rollout sampling selected no PPO transitions")
    return samples, episode_costs


def _normalize_rollout_advantages(samples):
    for key in ("reward_advantage", "cost_advantage"):
        values = torch.tensor(
            [sample[1][key] for sample in samples], dtype=torch.float32
        )
        normalized = normalize_advantage(values)
        for sample, value in zip(samples, normalized.tolist()):
            sample[1][f"normalized_{key}"] = value


def _validate_rollout_policy_version(samples, expected_version):
    versions = {
        sample[1].get("policy_version")
        for sample in samples
        if sample[1].get("policy_version") is not None
    }
    has_unversioned = any(
        sample[1].get("policy_version") is None for sample in samples
    )
    if has_unversioned:
        if versions:
            raise RuntimeError(
                "rollout dataset mixes versioned and unversioned policies"
            )
        if expected_version != 0:
            raise RuntimeError(
                "unversioned rollout data is accepted only for policy version 0"
            )
        return
    if versions != {expected_version}:
        raise RuntimeError(
            f"rollout policy versions {sorted(versions)} do not match "
            f"--policy-version={expected_version}"
        )


def _validate_sample_objective(samples, manifest):
    expected_schema = manifest.get("schema_version")
    expected_fingerprint = manifest.get("objective_fingerprint")
    schemas = {sample[1].get("schema_version") for sample in samples}
    fingerprints = {sample[1].get("objective_fingerprint") for sample in samples}
    if expected_schema == "safe-vln-go2-v1":
        if not schemas.issubset({None, expected_schema}):
            raise RuntimeError(
                f"legacy rollout sample schemas {schemas} do not match "
                f"manifest {expected_schema!r}"
            )
        if fingerprints != {None}:
            raise RuntimeError(
                "legacy rollout samples must not contain an objective fingerprint"
            )
        return
    if schemas != {expected_schema}:
        raise RuntimeError(
            f"rollout sample schemas {schemas} do not match manifest {expected_schema!r}"
        )
    if fingerprints != {expected_fingerprint}:
        raise RuntimeError(
            "rollout sample objective fingerprints do not match the manifest"
        )
    if expected_schema in VERSIONED_DATASET_SCHEMAS:
        policy_fingerprints = {
            sample[1].get("policy_objective_fingerprint") for sample in samples
        }
        if policy_fingerprints != {expected_fingerprint}:
            raise RuntimeError(
                "rollout cost/reward values were produced by a checkpoint with "
                "a different objective fingerprint"
            )


def train(args):
    if not args.checkpoint:
        raise ValueError("constrained PPO requires a warmup --checkpoint")
    _require_new_output_dir(args.output_dir)
    if getattr(args, "reset_critics", False):
        raise ValueError("--reset-critics is valid only for warmup-critics")
    if float(getattr(args, "oracle_ce_coef", 0.0)) < 0:
        raise ValueError("--oracle-ce-coef must be non-negative")
    if float(getattr(args, "oracle_stop_weight", 5.0)) <= 0:
        raise ValueError("--oracle-stop-weight must be positive")
    if int(getattr(args, "gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    manifest = _dataset_manifest(args.rollout_dir)
    if manifest.get("dataset_role") == "eval":
        raise RuntimeError("evaluation datasets cannot be used for PPO training")
    checkpoint_state = _checkpoint_state(args.checkpoint)
    _reject_failed_actor_audit(checkpoint_state)
    checkpoint_policy_version = checkpoint_state.get("policy_version")
    if (
        checkpoint_policy_version is not None
        and int(checkpoint_policy_version) != int(args.policy_version)
    ):
        raise RuntimeError(
            f"checkpoint policy version {checkpoint_policy_version} does not match "
            f"--policy-version={args.policy_version}"
        )
    schema, objective_fingerprint = _validate_objective_compatibility(
        manifest, checkpoint_state
    )
    objective_config = manifest.get("objective_config") or {}
    profile_limit = (
        objective_config.get("cost_profile", {}).get("cost_limit")
        if objective_config
        else None
    )
    if args.cost_limit is None:
        args.cost_limit = float(profile_limit if profile_limit is not None else 0.1)
    elif profile_limit is not None and abs(float(args.cost_limit) - float(profile_limit)) > 1e-8:
        raise RuntimeError(
            "--cost-limit does not match the rollout objective cost limit"
        )
    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        dtype=_training_dtype(args),
        checkpoint=args.checkpoint,
    )
    model.train()
    _enable_gradient_checkpointing(model)
    navila_model = model._navila_model()
    if navila_model is not None and hasattr(navila_model.llm, "model"):
        if not navila_model.llm.model.training:
            raise RuntimeError("NaViLA Llama backbone must be in training mode")
        if not getattr(navila_model.llm.model, "gradient_checkpointing", False):
            raise RuntimeError("NaViLA Llama gradient checkpointing was not enabled")
    optimizer = _optimizer(model, args.actor_lr, args.critic_lr)
    ppo = SafePPOOptimizer(
        optimizer,
        PPOConfig(
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
            normalize_advantages=False,
        ),
    )
    args.rollout_schema = schema
    args.rollout_objective_fingerprint = objective_fingerprint
    samples, episode_costs = _load_on_policy_samples(args)
    expected_episodes = manifest.get("completed_episodes")
    if (
        schema == LIVE_SCHEMA_VERSION
        and expected_episodes is not None
        and len(episode_costs) != int(expected_episodes)
    ):
        raise RuntimeError(
            "constraint statistic is missing complete rollout episodes: "
            f"loaded {len(episode_costs)}, manifest has {expected_episodes}"
        )
    _validate_sample_objective(samples, manifest)
    _validate_rollout_policy_version(samples, args.policy_version)
    _normalize_rollout_advantages(samples)
    ppo_sampling = sampling_summary(samples, action_field="action_id")
    print(
        json.dumps(
            {
                "mode": "ppo-sampling",
                "strategy": getattr(args, "sampling_strategy", "sequential"),
                "seed": int(getattr(args, "sampling_seed", 20260729)),
                "constraint_episodes": len(episode_costs),
                **ppo_sampling,
            }
        ),
        flush=True,
    )
    mean_cost = sum(episode_costs.values()) / len(episode_costs)
    lagrange_before = _initial_lagrange_multiplier(args, checkpoint_state)
    lagrange = LagrangeController(
        cost_limit=args.cost_limit,
        multiplier=lagrange_before,
        learning_rate=args.lagrange_lr,
    )
    lambda_before_update, lambda_after_update = _update_lagrange_for_rollout(
        lagrange, mean_cost
    )
    print(
        json.dumps(
            {
                "constraint/mean_episode_cost": mean_cost,
                "constraint/cost_limit": args.cost_limit,
                "constraint/excess": mean_cost - args.cost_limit,
                "constraint/lambda_before": lambda_before_update,
                "constraint/lambda_after": lambda_after_update,
                "constraint/cost_normalization": COST_NORMALIZATION,
            }
        ),
        flush=True,
    )

    update = 0
    for epoch in range(args.ppo_epochs):
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "constraint/lambda": lagrange.multiplier,
                    "constraint/mean_episode_cost": mean_cost,
                    "constraint/cost_limit": args.cost_limit,
                }
            ),
            flush=True,
        )
        accumulation_steps = max(
            1, int(getattr(args, "gradient_accumulation_steps", 1))
        )
        for window_start in range(
            0, len(samples), args.mini_batch_size * accumulation_steps
        ):
            micro_batches = [
                samples[start : start + args.mini_batch_size]
                for start in range(
                    window_start,
                    min(
                        len(samples),
                        window_start + args.mini_batch_size * accumulation_steps,
                    ),
                    args.mini_batch_size,
                )
            ]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            ppo.begin_accumulation()
            micro_stats = []
            for mini_batch in micro_batches:
                states = [
                    preprocessor(frames, metadata["instruction"])
                    for frames, metadata in mini_batch
                ]
                action_ids = torch.tensor(
                    [metadata["action_id"] for _, metadata in mini_batch],
                    device=args.device,
                )
                evaluated = evaluate_selected_actions(model, states, action_ids)
                device = evaluated["new_log_probs"].device

                def tensor(key):
                    return torch.tensor(
                        [metadata[key] for _, metadata in mini_batch],
                        dtype=torch.float32,
                        device=device,
                    )

                batch = {
                    **evaluated,
                    "old_log_probs": tensor("old_log_prob"),
                    "reward_advantages": tensor("normalized_reward_advantage"),
                    "cost_advantages": tensor("normalized_cost_advantage"),
                    "reward_returns": tensor("ppo_reward_return"),
                    "cost_returns": tensor("ppo_cost_return"),
                    "oracle_action_ids": torch.tensor(
                        [
                            int(metadata.get("oracle_action_id") or 0)
                            for _, metadata in mini_batch
                        ],
                        dtype=torch.long,
                        device=device,
                    ),
                    "oracle_mask": torch.tensor(
                        [
                            bool(metadata.get("oracle_eligible", False))
                            for _, metadata in mini_batch
                        ],
                        dtype=torch.bool,
                        device=device,
                    ),
                    "oracle_sample_weights": torch.tensor(
                        [
                            (
                                float(args.oracle_stop_weight)
                                if metadata.get("oracle_action_id") == 9
                                else 1.0
                            )
                            for _, metadata in mini_batch
                        ],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "oracle_ce_coef": (
                        float(args.oracle_ce_coef)
                        if schema == LIVE_SCHEMA_VERSION
                        else 0.0
                    ),
                }
                micro_stats.append(
                    ppo.accumulate(
                        batch,
                        lagrange.multiplier,
                        gradient_scale=1.0 / len(micro_batches),
                    )
                )
                del states, evaluated, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            stats = {}
            for key in micro_stats[0]:
                values = [float(item[key]) for item in micro_stats]
                stats[key] = sum(values) / len(values)
            stats["oracle/samples"] = sum(
                int(item["oracle/samples"]) for item in micro_stats
            )
            stats["oracle/stop_samples"] = sum(
                int(item["oracle/stop_samples"]) for item in micro_stats
            )
            stats["optimizer/grad_norm"] = ppo.finish_accumulation()
            stats["optimizer/micro_batches"] = len(micro_batches)
            if torch.cuda.is_available():
                stats["memory/peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                )
            update += 1
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "update": update,
                        "lambda": lagrange.multiplier,
                        **stats,
                    }
                ),
                flush=True,
            )

    state = {
        "mode": "constrained-ppo",
        "policy_version": args.policy_version + 1,
        "updates": update,
        "lagrange_multiplier": lagrange.multiplier,
        "lagrange_multiplier_before_update": lagrange_before,
        "constraint_excess": mean_cost - args.cost_limit,
        "mean_episode_cost": mean_cost,
        "cost_limit": args.cost_limit,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "training_dtype": getattr(args, "training_dtype", "bfloat16"),
        "schema_version": schema,
        "objective_fingerprint": objective_fingerprint,
        "objective_config": objective_config or None,
        "oracle_stop_weight": float(args.oracle_stop_weight),
        "sampling_strategy": getattr(
            args, "sampling_strategy", "sequential"
        ),
        "sampling_seed": int(getattr(args, "sampling_seed", 20260729)),
        "sampling": ppo_sampling,
        "constraint_episode_count": len(episode_costs),
        "constraint_cost_normalization": COST_NORMALIZATION,
        "gradient_accumulation_steps": int(
            getattr(args, "gradient_accumulation_steps", 1)
        ),
    }
    _copy_actor_contract(state, checkpoint_state)
    save_checkpoint(model, optimizer, args.output_dir, state)
    return 0
