"""Executable critic warm-start and constrained PPO passes."""

from __future__ import annotations

from collections import defaultdict
import json

import torch

from .cmdp import LagrangeController, compute_gae
from .dataset import iter_samples
from .learner import evaluate_selected_actions, save_checkpoint, train_critic_epoch
from .navila import load_safe_navila
from .trainer import PPOConfig, SafePPOOptimizer


def _optimizer(model, actor_lr, critic_lr):
    actor = [parameter for parameter in model.base_model.parameters() if parameter.requires_grad]
    critics = list(model.reward_head.parameters()) + list(model.cost_head.parameters())
    groups = []
    if actor:
        groups.append({"params": actor, "lr": actor_lr})
    groups.append({"params": critics, "lr": critic_lr})
    return torch.optim.AdamW(groups)


def warmup(args):
    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    model.base_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(model.reward_head.parameters()) + list(model.cost_head.parameters()),
        lr=args.critic_lr,
    )
    state = {"mode": "warmup-critics"}
    for epoch in range(args.epochs):
        stats = train_critic_epoch(
            model,
            iter_samples(args.dataset_dir, args.split),
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


def train(args):
    if not args.checkpoint:
        raise ValueError("constrained PPO requires a warmup --checkpoint")
    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    optimizer = _optimizer(model, args.actor_lr, args.critic_lr)
    ppo = SafePPOOptimizer(
        optimizer,
        PPOConfig(
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
        ),
    )
    samples, episode_costs = _load_on_policy_samples(args)
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
                "reward_advantages": tensor("reward_advantage"),
                "cost_advantages": tensor("cost_advantage"),
                "reward_returns": tensor("ppo_reward_return"),
                "cost_returns": tensor("ppo_cost_return"),
            }
            stats = ppo.step(batch, lagrange.multiplier)
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
        "mean_episode_cost": mean_cost,
        "cost_limit": args.cost_limit,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
    }
    save_checkpoint(model, optimizer, args.output_dir, state)
    return 0
