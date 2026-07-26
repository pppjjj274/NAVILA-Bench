"""Executable critic warm-start and constrained PPO passes."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import torch

from .cmdp import LagrangeController, compute_gae
from .dataset import iter_samples
from .learner import evaluate_selected_actions, save_checkpoint, train_critic_epoch
from .navila import load_safe_navila
from .objective import validate_objective_config
from .trainer import PPOConfig, SafePPOOptimizer, normalize_advantage


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


def _validate_objective_compatibility(manifest, checkpoint_state):
    schema = manifest.get("schema_version")
    dataset_fingerprint = manifest.get("objective_fingerprint")
    checkpoint_fingerprint = checkpoint_state.get("objective_fingerprint")
    if schema == "safe-vln-go2-v2":
        if not dataset_fingerprint:
            raise RuntimeError("v2 rollout manifest has no objective fingerprint")
        objective_config = manifest.get("objective_config")
        if not objective_config:
            raise RuntimeError("v2 rollout manifest has no objective configuration")
        try:
            validated_objective = validate_objective_config(objective_config)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("v2 rollout objective configuration is invalid") from error
        if validated_objective["fingerprint"] != dataset_fingerprint:
            raise RuntimeError(
                "v2 rollout manifest fingerprint does not match its objective"
            )
        if checkpoint_fingerprint != dataset_fingerprint:
            raise RuntimeError(
                "checkpoint and rollout objective fingerprints do not match"
            )
    elif dataset_fingerprint or checkpoint_fingerprint:
        raise RuntimeError("legacy v1 data cannot be mixed with a v2 objective")
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
    critics = list(model.reward_head.parameters()) + list(model.cost_head.parameters())
    groups = []
    if actor:
        groups.append({"params": actor, "lr": actor_lr})
    groups.append({"params": critics, "lr": critic_lr})
    return torch.optim.AdamW(groups)


def warmup(args):
    if getattr(args, "reset_critics", False) and not args.checkpoint:
        raise ValueError("--reset-critics requires --checkpoint")
    manifest = _dataset_manifest(args.dataset_dir)
    checkpoint_state = _checkpoint_state(args.checkpoint)
    schema = manifest.get("schema_version")
    objective_fingerprint = manifest.get("objective_fingerprint")
    if schema == "safe-vln-go2-v2" and not objective_fingerprint:
        raise RuntimeError("v2 warmup dataset has no objective fingerprint")
    if (
        schema == "safe-vln-go2-v2"
        and args.checkpoint
        and not getattr(args, "reset_critics", False)
        and checkpoint_state.get("objective_fingerprint") != objective_fingerprint
    ):
        raise RuntimeError(
            "checkpoint and warmup dataset objective fingerprints do not match; "
            "use --reset-critics when migrating an actor to the v2 objective"
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

    def validated_samples():
        for frames, metadata in iter_samples(args.dataset_dir, args.split):
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
            yield frames, metadata

    for epoch in range(args.epochs):
        stats = train_critic_epoch(
            model,
            validated_samples(),
            preprocessor,
            optimizer,
            max_samples=args.max_samples,
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
    count = 0
    for frames, metadata in iter_samples(args.rollout_dir, args.split):
        if any(metadata.get(key) is None for key in required):
            continue
        episodes[str(metadata["episode_id"])].append((frames, metadata))
        count += 1
        if args.max_samples and count >= args.max_samples:
            break
    if not episodes:
        raise RuntimeError("no structured on-policy transitions in rollout dataset")

    samples = []
    episode_costs = {}
    for episode_id, episode_samples in episodes.items():
        episode_samples.sort(key=lambda sample: int(sample[1]["index"]))
        metadata = [sample[1] for sample in episode_samples]
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
        episode_costs[episode_id] = sum(float(item["cost"]) for item in metadata)
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
                "legacy rollout samples must not contain a v2 objective fingerprint"
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
    if expected_schema == "safe-vln-go2-v2":
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
    if getattr(args, "reset_critics", False):
        raise ValueError("--reset-critics is valid only for warmup-critics")
    manifest = _dataset_manifest(args.rollout_dir)
    checkpoint_state = _checkpoint_state(args.checkpoint)
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
    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        try:
            model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.base_model.gradient_checkpointing_enable()
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
    samples, episode_costs = _load_on_policy_samples(args)
    _validate_sample_objective(samples, manifest)
    _validate_rollout_policy_version(samples, args.policy_version)
    _normalize_rollout_advantages(samples)
    mean_cost = sum(episode_costs.values()) / len(episode_costs)
    lagrange = LagrangeController(
        cost_limit=args.cost_limit,
        multiplier=args.initial_lagrange_multiplier,
        learning_rate=args.lagrange_lr,
    )
    lagrange.update(mean_cost)

    update = 0
    for epoch in range(args.ppo_epochs):
        for start in range(0, len(samples), args.mini_batch_size):
            mini_batch = samples[start : start + args.mini_batch_size]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
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
            }
            stats = ppo.step(batch, lagrange.multiplier)
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
            del states, evaluated, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    state = {
        "mode": "constrained-ppo",
        "policy_version": args.policy_version + 1,
        "updates": update,
        "lagrange_multiplier": lagrange.multiplier,
        "mean_episode_cost": mean_cost,
        "cost_limit": args.cost_limit,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "training_dtype": getattr(args, "training_dtype", "bfloat16"),
        "schema_version": schema,
        "objective_fingerprint": objective_fingerprint,
        "objective_config": objective_config or None,
    }
    save_checkpoint(model, optimizer, args.output_dir, state)
    return 0
