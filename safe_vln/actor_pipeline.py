"""Two-stage training for the Safe-VLN v5 hierarchical actor."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import torch

from .actor_training import (
    audit_hierarchical_actor,
    hierarchical_actor_loss,
    identity_of,
    metadata_of,
    split_actor_episodes,
    stratified_actor_schedule,
)
from .dataset import iter_sample_refs, load_sample_refs
from .learner import save_checkpoint
from .live_render import (
    LIVE_SCHEMA_VERSION,
    NAVILA_HISTORY_SAMPLING_POLICY,
)
from .model import ACTOR_ARCHITECTURE_HIERARCHICAL
from .navila import load_safe_navila
from .sampling import sampling_summary


def _load_one(ref):
    loaded = load_sample_refs([ref])
    if len(loaded) != 1:
        raise RuntimeError("failed to load one actor sample")
    return loaded[0]


def _episode_count(items):
    return len({str(metadata_of(item).get("episode_id")) for item in items})


def _scene_count(items):
    return len({str(metadata_of(item).get("scene_id")) for item in items})


def _validate_v5_refs(refs, *, allow_small_dataset: bool):
    counts = Counter(
        int(ref.metadata["oracle_action_id"])
        for ref in refs
        if ref.metadata.get("oracle_action_id") is not None
    )
    missing = [action_id for action_id in range(10) if action_id not in counts]
    if missing:
        raise RuntimeError(f"v5 actor dataset misses action classes: {missing}")
    if not allow_small_dataset:
        failures = []
        if _episode_count(refs) != 500:
            failures.append(f"episodes={_episode_count(refs)} (expected 500)")
        if _scene_count(refs) != 61:
            failures.append(f"scenes={_scene_count(refs)} (expected 61)")
        rare = {key: value for key, value in counts.items() if value < 50}
        if rare:
            failures.append(f"actions_below_50={rare}")
        if counts[9] < 150:
            failures.append(f"stop={counts[9]} (expected >=150)")
        if failures:
            raise RuntimeError(
                "v5 actor dataset acceptance failed: " + "; ".join(failures)
            )
    return counts


def _cache_features(model, preprocessor, refs, cache_path, model_path):
    schedule_ids = [identity_of(ref) for ref in refs]
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            cached.get("schema_version") == LIVE_SCHEMA_VERSION
            and cached.get("model_path") == str(Path(model_path).resolve())
            and cached.get("schedule_ids") == schedule_ids
        ):
            print(
                json.dumps(
                    {
                        "mode": "warmup-actor-cache",
                        "reused": True,
                        "samples": len(refs),
                    }
                ),
                flush=True,
            )
            return cached
    unique_refs = {}
    for ref in refs:
        unique_refs.setdefault(identity_of(ref), ref)
    features_by_id = {}
    model.eval()
    with torch.no_grad():
        for index, (ref_id, ref) in enumerate(unique_refs.items(), 1):
            frames, metadata = _load_one(ref)
            prepared = preprocessor(frames, metadata["instruction"])
            hidden = model.encode_state(
                prepared.input_ids, images=prepared.images
            )
            features_by_id[ref_id] = hidden.detach().float().cpu()
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "mode": "warmup-actor-cache",
                            "cached_unique": index,
                            "total_unique": len(unique_refs),
                            "scheduled": len(refs),
                        }
                    ),
                    flush=True,
                )
    payload = {
        "schema_version": LIVE_SCHEMA_VERSION,
        "model_path": str(Path(model_path).resolve()),
        "schedule_ids": schedule_ids,
        "unique_samples": len(unique_refs),
        "features": torch.cat(
            [features_by_id[ref_id] for ref_id in schedule_ids], dim=0
        ),
        "targets": torch.tensor(
            [int(ref.metadata["oracle_action_id"]) for ref in refs],
            dtype=torch.long,
        ),
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".incomplete")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return payload


def _head_warmup(model, cache, args):
    model.actor_head.train()
    optimizer = torch.optim.AdamW(
        model.actor_head.parameters(), lr=args.head_warmup_lr
    )
    features = cache["features"]
    targets = cache["targets"]
    device = next(model.actor_head.parameters()).device
    for epoch in range(args.head_warmup_epochs):
        order = torch.arange(len(targets))
        order = torch.roll(
            order, shifts=4 * (epoch % max(len(targets) // 4, 1))
        )
        total_loss = total_stop = total_motion = 0.0
        batches = 0
        for start in range(0, len(order), args.head_batch_size):
            indices = order[start : start + args.head_batch_size]
            hidden = features[indices].to(device)
            target = targets[indices].to(device)
            stop_logits, motion_logits = model.actor_head(hidden)
            loss, stop_loss, motion_loss = hierarchical_actor_loss(
                stop_logits, motion_logits, target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.actor_head.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("head warmup gradient is not finite")
            optimizer.step()
            total_loss += float(loss.detach())
            total_stop += float(stop_loss.detach())
            total_motion += float(motion_loss.detach())
            batches += 1
        print(
            json.dumps(
                {
                    "mode": "warmup-actor-head",
                    "epoch": epoch + 1,
                    "loss/total": total_loss / max(batches, 1),
                    "loss/stop": total_stop / max(batches, 1),
                    "loss/motion": total_motion / max(batches, 1),
                }
            ),
            flush=True,
        )
    return optimizer


def _finetune_lora(model, preprocessor, schedule, lora_parameters, args):
    for parameter in lora_parameters:
        parameter.requires_grad_(True)
    model.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.actor_lr},
            {"params": list(model.actor_head.parameters()), "lr": args.head_lr},
        ]
    )
    update = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        epoch_schedule = stratified_actor_schedule(
            schedule,
            sample_count=len(schedule),
            stop_fraction=args.stop_fraction,
            seed=args.sampling_seed + 1000 + epoch,
        )
        total_loss = total_stop = total_motion = 0.0
        for index, ref in enumerate(epoch_schedule, 1):
            frames, metadata = _load_one(ref)
            prepared = preprocessor(frames, metadata["instruction"])
            output = model(prepared.input_ids, images=prepared.images)
            target = torch.tensor(
                [int(metadata["oracle_action_id"])],
                device=output.action_logits.device,
            )
            loss, stop_loss, motion_loss = hierarchical_actor_loss(
                output.stop_logits, output.motion_logits, target
            )
            (loss / args.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach())
            total_stop += float(stop_loss.detach())
            total_motion += float(motion_loss.detach())
            boundary = (
                index % args.gradient_accumulation_steps == 0
                or index == len(epoch_schedule)
            )
            if boundary:
                parameters = lora_parameters + list(model.actor_head.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, args.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("actor fine-tune gradient is not finite")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "mode": "warmup-actor-lora",
                            "epoch": epoch + 1,
                            "sample": index,
                            "total": len(epoch_schedule),
                            "updates": update,
                            "loss/total": total_loss / index,
                            "loss/stop": total_stop / index,
                            "loss/motion": total_motion / index,
                        }
                    ),
                    flush=True,
                )
    return optimizer, update


def _audit(model, preprocessor, dev_refs, stop_threshold):
    stop_probabilities = []
    motion_predictions = []
    targets = []
    normalized = True
    model.eval()
    with torch.no_grad():
        for index, ref in enumerate(dev_refs, 1):
            frames, metadata = _load_one(ref)
            prepared = preprocessor(frames, metadata["instruction"])
            output = model(prepared.input_ids, images=prepared.images)
            probabilities = output.distribution.probs[0].float()
            normalized = bool(
                normalized
                and torch.isfinite(probabilities).all()
                and math.isclose(
                    float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-5
                )
            )
            stop_probabilities.append(
                float(torch.sigmoid(output.stop_logits).item())
            )
            motion_predictions.append(int(output.motion_logits.argmax().item()))
            targets.append(int(metadata["oracle_action_id"]))
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "mode": "warmup-actor-dev-audit",
                            "evaluated": index,
                            "total": len(dev_refs),
                        }
                    ),
                    flush=True,
                )
    audit = audit_hierarchical_actor(
        stop_probabilities,
        motion_predictions,
        targets,
        stop_threshold=stop_threshold,
    )
    audit["probabilities_normalized"] = normalized
    return audit


def warmup_hierarchical_actor(args, manifest, training_dtype):
    positive = {
        "actor_lr": args.actor_lr,
        "head_lr": args.head_lr,
        "head_warmup_lr": args.head_warmup_lr,
        "head_warmup_epochs": args.head_warmup_epochs,
        "head_batch_size": args.head_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "epochs": args.epochs,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"v5 actor parameters must be positive: {invalid}")
    if args.head_batch_size % 4:
        raise ValueError("head batch size must be divisible by four")
    if args.gradient_accumulation_steps % 4:
        raise ValueError("gradient accumulation must be divisible by four")
    for name in (
        "stop_fraction",
        "stop_threshold",
        "minimum_stop_accuracy",
        "maximum_false_stop_rate",
        "minimum_non_stop_macro_accuracy",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if getattr(args, "checkpoint", None):
        raise ValueError("warmup-actor starts from original NaViLA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_fingerprint = manifest.get("objective_fingerprint")

    def eligible(ref):
        metadata = ref.metadata
        if metadata.get("schema_version") != LIVE_SCHEMA_VERSION:
            raise RuntimeError("v5 actor sample schema does not match manifest")
        if metadata.get("objective_fingerprint") != objective_fingerprint:
            raise RuntimeError("v5 actor sample objective fingerprint mismatch")
        if metadata.get("history_sampling_policy") != NAVILA_HISTORY_SAMPLING_POLICY:
            raise RuntimeError("v5 actor sample history policy mismatch")
        return bool(
            metadata.get("oracle_eligible", False)
            and metadata.get("oracle_action_id") is not None
        )

    refs = [
        ref
        for ref in iter_sample_refs(args.dataset_dir, args.split)
        if eligible(ref)
    ]
    counts = _validate_v5_refs(
        refs, allow_small_dataset=args.allow_small_dataset
    )
    train_refs, dev_refs = split_actor_episodes(
        refs,
        seed=args.sampling_seed,
        dev_episodes_per_scene=args.dev_episodes_per_scene,
    )
    sample_count = args.max_samples or len(train_refs)
    schedule = stratified_actor_schedule(
        train_refs,
        sample_count=sample_count,
        stop_fraction=args.stop_fraction,
        seed=args.sampling_seed,
    )
    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        dtype=training_dtype,
        checkpoint=None,
        actor_architecture=ACTOR_ARCHITECTURE_HIERARCHICAL,
        stop_threshold=args.stop_threshold,
    )
    model.reward_head.requires_grad_(False)
    model.cost_head.requires_grad_(False)
    lora_parameters = [
        parameter
        for parameter in model.base_model.parameters()
        if parameter.requires_grad
    ]
    if not lora_parameters:
        raise RuntimeError("original NaViLA has no trainable LoRA parameters")
    for parameter in lora_parameters:
        parameter.requires_grad_(False)

    cache_path = output_dir / "actor_feature_cache.pt"
    cache = _cache_features(
        model, preprocessor, schedule, cache_path, args.model_path
    )
    _head_warmup(model, cache, args)
    optimizer, updates = _finetune_lora(
        model, preprocessor, schedule, lora_parameters, args
    )
    audit = _audit(model, preprocessor, dev_refs, args.stop_threshold)
    all_motion_classes_present = all(
        str(action_id) in audit["class_samples"]
        for action_id in range(9)
    )
    accepted = bool(
        all_motion_classes_present
        and "9" in audit["class_samples"]
        and audit["finite"]
        and audit["probabilities_normalized"]
        and audit["stop_recall"] is not None
        and audit["stop_recall"] >= args.minimum_stop_accuracy
        and audit["false_stop_rate_non_goal"] is not None
        and audit["false_stop_rate_non_goal"] <= args.maximum_false_stop_rate
        and audit["non_stop_macro_accuracy"] is not None
        and audit["non_stop_macro_accuracy"]
        >= args.minimum_non_stop_macro_accuracy
    )
    state = {
        "mode": "warmup-actor",
        "training_dtype": args.training_dtype,
        "schema_version": LIVE_SCHEMA_VERSION,
        "objective_fingerprint": objective_fingerprint,
        "objective_config": manifest.get("objective_config"),
        "policy_version": 0,
        "fresh_lora": True,
        "actor_architecture": ACTOR_ARCHITECTURE_HIERARCHICAL,
        "stop_threshold": args.stop_threshold,
        "updates": updates,
        "sampling_seed": args.sampling_seed,
        "sampling": sampling_summary(schedule),
        "dataset_action_counts": {
            str(key): value for key, value in sorted(counts.items())
        },
        "train_episodes": _episode_count(train_refs),
        "dev_episodes": _episode_count(dev_refs),
        "train_scenes": _scene_count(train_refs),
        "dev_scenes": _scene_count(dev_refs),
        "dev_episode_ids": sorted(
            {str(ref.metadata.get("episode_id")) for ref in dev_refs}
        ),
        "actor/audit_stop_accuracy": audit["stop_recall"],
        "actor/audit_stop_recall": audit["stop_recall"],
        "actor/audit_false_stop_rate_non_goal": audit[
            "false_stop_rate_non_goal"
        ],
        "actor/audit_non_stop_macro_accuracy": audit[
            "non_stop_macro_accuracy"
        ],
        "actor/audit_per_class_accuracy": audit["per_class_accuracy"],
        "actor/audit_class_samples": audit["class_samples"],
        "actor/audit_all_motion_classes_present": (
            all_motion_classes_present
        ),
        "actor/audit_probabilities_normalized": audit[
            "probabilities_normalized"
        ],
        "actor/minimum_stop_accuracy": args.minimum_stop_accuracy,
        "actor/maximum_false_stop_rate": args.maximum_false_stop_rate,
        "actor/minimum_non_stop_macro_accuracy": (
            args.minimum_non_stop_macro_accuracy
        ),
        "actor/accepted": accepted,
    }
    print(json.dumps({"mode": "warmup-actor-audit", **state}), flush=True)
    save_checkpoint(model, optimizer, output_dir, state)
    if not accepted:
        raise RuntimeError(
            "v5 dev actor audit failed; checkpoint is diagnostic-only"
        )
    return 0
