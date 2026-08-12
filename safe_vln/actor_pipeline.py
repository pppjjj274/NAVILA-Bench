"""Two-stage training for the Safe-VLN v5 hierarchical actor."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import shutil

import torch

from .checkpoint import (
    CHECKPOINT_ROLE_DIAGNOSTIC,
    CHECKPOINT_ROLE_POLICY,
    POLICY_INTERFACE_SAFE_DISCRETE,
    SAFE_CHECKPOINT_CONTRACT_VERSION,
    require_safe_policy_checkpoint,
)
from .actor_training import (
    audit_hierarchical_actor,
    calibrate_stop_threshold,
    factorized_actor_loss,
    hierarchical_actor_loss,
    identity_of,
    metadata_of,
    split_actor_partitions,
    stable_rank,
    stratified_actor_schedule,
    target_action_id,
)
from .dataset import iter_sample_refs, load_sample_refs
from .learner import save_checkpoint
from .live_render import (
    LIVE_SCHEMA_VERSION,
    NAVILA_HISTORY_SAMPLING_POLICY,
)
from .model import ACTOR_ARCHITECTURE_FACTORIZED
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


def online_dagger_weight(metadata):
    """Prioritize the closed-loop mistakes that cause persistent turning."""
    oracle_action = metadata.get("oracle_action_id")
    policy_action = metadata.get("policy_action_id")
    if not isinstance(oracle_action, int) or not isinstance(policy_action, int):
        return 1
    if 6 <= oracle_action <= 8 and 0 <= policy_action <= 5:
        return 4
    return 2 if oracle_action != policy_action else 1


def online_dagger_schedule(
    online_refs,
    anchor_refs,
    *,
    sample_count,
    online_fraction,
    seed,
):
    """Mix on-policy recovery states with static oracle anchors deterministically."""
    if sample_count < 2:
        raise ValueError("online DAgger requires at least two samples")
    if not 0.0 < float(online_fraction) < 1.0:
        raise ValueError("online_fraction must be in (0, 1)")
    if not online_refs or not anchor_refs:
        raise RuntimeError("online DAgger requires online and anchor samples")

    def draw(refs, count, namespace, weighted):
        expanded = []
        for ref in refs:
            repeats = online_dagger_weight(ref.metadata) if weighted else 1
            expanded.extend((ref, slot) for slot in range(repeats))
        ordered = sorted(
            expanded,
            key=lambda item: (
                stable_rank(item[0], seed=seed, namespace=namespace), item[1]
            ),
        )
        return [ordered[index % len(ordered)][0] for index in range(count)]

    online_count = max(1, min(sample_count - 1, round(sample_count * online_fraction)))
    anchor_count = sample_count - online_count
    merged = [
        *( ("online", ref) for ref in draw(online_refs, online_count, "dagger-online", True) ),
        *( ("anchor", ref) for ref in draw(anchor_refs, anchor_count, "dagger-anchor", False) ),
    ]
    merged.sort(
        key=lambda item: stable_rank(
            item[1], seed=seed, namespace=f"dagger-merge:{item[0]}"
        )
    )
    return [ref for _, ref in merged]


def _has_verified_v5_safety(metadata):
    diagnostics = metadata.get("safety_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    if diagnostics.get("contact_sensor_enabled") is not True:
        return False
    try:
        action_id = None if metadata.get("action_id") is None else int(
            metadata["action_id"]
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return not (
        action_id is not None
        and 0 <= action_id <= 8
        and not isinstance(diagnostics.get("turn_execution"), dict)
    )


def _validate_v5_refs(
    refs,
    *,
    allow_small_dataset: bool,
    target_field: str = "oracle_action_id",
):
    unverified = [
        str(ref.metadata.get("episode_id"))
        for ref in refs
        if not _has_verified_v5_safety(ref.metadata)
    ]
    if unverified:
        raise RuntimeError(
            "actor dataset contains samples collected without verified Isaac "
            f"contact processing (episodes={sorted(set(unverified))[:5]})"
        )
    counts = Counter(
        target_action_id(ref, target_field)
        for ref in refs
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


def _cache_features(
    model,
    preprocessor,
    refs,
    cache_path,
    model_path,
    *,
    target_field: str = "oracle_action_id",
):
    schedule_ids = [identity_of(ref) for ref in refs]
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            cached.get("schema_version") == LIVE_SCHEMA_VERSION
            and cached.get("model_path") == str(Path(model_path).resolve())
            and cached.get("schedule_ids") == schedule_ids
            and cached.get("target_field", "oracle_action_id") == target_field
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
        "target_field": target_field,
        "schedule_ids": schedule_ids,
        "unique_samples": len(unique_refs),
        "features": torch.cat(
            [features_by_id[ref_id] for ref_id in schedule_ids], dim=0
        ),
        "targets": torch.tensor(
            [target_action_id(ref, target_field) for ref in refs],
            dtype=torch.long,
        ),
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".incomplete")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return payload


def _actor_losses(model, targets, *, hidden=None, output=None):
    if (hidden is None) == (output is None):
        raise ValueError("provide exactly one of hidden or output")
    if hidden is not None:
        actor_outputs = model.actor_head(hidden)
        if model.actor_architecture == ACTOR_ARCHITECTURE_FACTORIZED:
            stop_logits, direction_logits, magnitude_logits = actor_outputs
        else:
            stop_logits, motion_logits = actor_outputs
    elif model.actor_architecture == ACTOR_ARCHITECTURE_FACTORIZED:
        stop_logits = output.stop_logits
        direction_logits = output.direction_logits
        magnitude_logits = output.magnitude_logits
    else:
        stop_logits = output.stop_logits
        motion_logits = output.motion_logits

    if model.actor_architecture == ACTOR_ARCHITECTURE_FACTORIZED:
        losses = factorized_actor_loss(
            stop_logits,
            direction_logits,
            magnitude_logits,
            targets,
        )
        return losses[0], {
            "stop": losses[1],
            "direction": losses[2],
            "magnitude_hard": losses[3],
            "magnitude_ordinal": losses[4],
        }
    losses = hierarchical_actor_loss(
        stop_logits, motion_logits, targets
    )
    return losses[0], {
        "stop": losses[1],
        "motion": losses[2],
    }


def _zero_loss_totals(model):
    names = (
        ("stop", "direction", "magnitude_hard", "magnitude_ordinal")
        if model.actor_architecture == ACTOR_ARCHITECTURE_FACTORIZED
        else ("stop", "motion")
    )
    return {name: 0.0 for name in names}


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
        total_loss = 0.0
        component_totals = _zero_loss_totals(model)
        batches = 0
        for start in range(0, len(order), args.head_batch_size):
            indices = order[start : start + args.head_batch_size]
            hidden = features[indices].to(device)
            target = targets[indices].to(device)
            loss, components = _actor_losses(model, target, hidden=hidden)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.actor_head.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("head warmup gradient is not finite")
            optimizer.step()
            total_loss += float(loss.detach())
            for name, value in components.items():
                component_totals[name] += float(value.detach())
            batches += 1
        print(
            json.dumps(
                {
                    "mode": "warmup-actor-head",
                    "epoch": epoch + 1,
                    "loss/total": total_loss / max(batches, 1),
                    **{
                        f"loss/{name}": value / max(batches, 1)
                        for name, value in component_totals.items()
                    },
                }
            ),
            flush=True,
        )
    return optimizer


def _finetune_lora(
    model,
    preprocessor,
    schedule,
    lora_parameters,
    args,
    *,
    schedule_for_epoch=None,
    target_field: str = "oracle_action_id",
):
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
        epoch_schedule = (
            schedule_for_epoch(epoch)
            if schedule_for_epoch is not None
            else stratified_actor_schedule(
                schedule,
                sample_count=len(schedule),
                stop_fraction=args.stop_fraction,
                seed=args.sampling_seed + 1000 + epoch,
                hard_stop_negative_fraction=args.hard_stop_negative_fraction,
                hard_stop_negative_margin_m=args.hard_stop_negative_margin_m,
                target_field=target_field,
            )
        )
        total_loss = 0.0
        component_totals = _zero_loss_totals(model)
        for index, ref in enumerate(epoch_schedule, 1):
            frames, metadata = _load_one(ref)
            prepared = preprocessor(frames, metadata["instruction"])
            output = model(prepared.input_ids, images=prepared.images)
            target = torch.tensor(
                [target_action_id(metadata, target_field)],
                device=output.action_logits.device,
            )
            loss, components = _actor_losses(model, target, output=output)
            (loss / args.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach())
            for name, value in components.items():
                component_totals[name] += float(value.detach())
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
                            **{
                                f"loss/{name}": value / index
                                for name, value in component_totals.items()
                            },
                        }
                    ),
                    flush=True,
                )
    return optimizer, update


def _collect_predictions(
    model,
    preprocessor,
    refs,
    *,
    label,
    target_field: str = "oracle_action_id",
):
    stop_probabilities = []
    motion_predictions = []
    targets = []
    records = []
    normalized = True
    model.eval()
    with torch.no_grad():
        for index, ref in enumerate(refs, 1):
            frames, metadata = _load_one(ref)
            prepared = preprocessor(frames, metadata["instruction"])
            output = model(prepared.input_ids, images=prepared.images)
            probabilities = output.distribution.probs[0].float()
            motion_probabilities = torch.softmax(
                output.motion_logits[0].float(), dim=-1
            )
            normalized = bool(
                normalized
                and torch.isfinite(probabilities).all()
                and torch.isfinite(motion_probabilities).all()
                and math.isclose(
                    float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-5
                )
                and math.isclose(
                    float(motion_probabilities.sum()),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
            )
            stop_probability = float(torch.sigmoid(output.stop_logits).item())
            motion_prediction = int(output.motion_logits.argmax().item())
            target = target_action_id(metadata, target_field)
            stop_probabilities.append(stop_probability)
            motion_predictions.append(motion_prediction)
            targets.append(target)
            record = {
                "sample_id": identity_of(ref),
                "shard_path": str(getattr(ref, "shard_path", "")),
                "metadata_name": str(getattr(ref, "metadata_name", "")),
                "scene_id": str(metadata.get("scene_id")),
                "episode_id": str(metadata.get("episode_id")),
                "index": metadata.get("index"),
                "target_action_id": target,
                "stop_probability": stop_probability,
                "motion_prediction": motion_prediction,
                "motion_probabilities": motion_probabilities.cpu().tolist(),
                "action_probabilities": probabilities.cpu().tolist(),
            }
            if output.direction_logits is not None:
                record["direction_probabilities"] = torch.softmax(
                    output.direction_logits[0].float(), dim=-1
                ).cpu().tolist()
                record["magnitude_probabilities"] = torch.softmax(
                    output.magnitude_logits[0].float(), dim=-1
                ).cpu().tolist()
            records.append(record)
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "mode": f"warmup-actor-{label}",
                            "evaluated": index,
                            "total": len(refs),
                        }
                    ),
                    flush=True,
                )
    return {
        "stop_probabilities": stop_probabilities,
        "motion_predictions": motion_predictions,
        "targets": targets,
        "records": records,
        "probabilities_normalized": normalized,
    }


def _audit_predictions(predictions, stop_threshold):
    audit = audit_hierarchical_actor(
        predictions["stop_probabilities"],
        predictions["motion_predictions"],
        predictions["targets"],
        stop_threshold=stop_threshold,
    )
    audit["probabilities_normalized"] = predictions[
        "probabilities_normalized"
    ]
    return audit


def _motion_only_audit(predictions):
    samples = Counter()
    correct = Counter()
    for prediction, target in zip(
        predictions["motion_predictions"], predictions["targets"]
    ):
        if int(target) == 9:
            continue
        samples[int(target)] += 1
        correct[int(target)] += int(prediction == target)
    per_class = {
        str(action_id): correct[action_id] / samples[action_id]
        for action_id in sorted(samples)
    }
    return {
        "class_samples": {str(key): value for key, value in sorted(samples.items())},
        "per_class_accuracy": per_class,
        "macro_accuracy": (
            sum(per_class.values()) / len(per_class) if per_class else None
        ),
        "all_motion_classes_present": all(action_id in samples for action_id in range(9)),
        "probabilities_normalized": predictions["probabilities_normalized"],
    }


def _write_diagnostics(output_dir, label, predictions, report):
    output_dir = Path(output_dir)
    (output_dir / f"actor_{label}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [json.dumps(record) for record in predictions["records"]]
    (output_dir / f"actor_{label}_predictions.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def warmup_hierarchical_actor(args, manifest, training_dtype):
    if getattr(args, "sampling_strategy", None) != "stratified":
        raise ValueError(
            "hierarchical Actor training requires "
            "--sampling-strategy=stratified; sequential/balanced-oracle are "
            "candidate-scoring modes"
        )
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
    if not 0.0 < args.stop_fraction < 1.0:
        raise ValueError("stop_fraction must be in (0, 1)")
    if not 0.0 < args.stop_threshold_grid_step < 1.0:
        raise ValueError("stop_threshold_grid_step must be in (0, 1)")
    if getattr(args, "checkpoint", None):
        raise ValueError("warmup-actor starts from original NaViLA")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite Actor output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_fingerprint = manifest.get("objective_fingerprint")
    target_source = getattr(args, "actor_target_source", "oracle")
    target_contracts = {
        "oracle": ("oracle_action_id", "dynamic-oracle"),
        "navila-policy": (
            "actor_teacher_action_id",
            "original-navila-policy",
        ),
    }
    try:
        target_field, audit_target_source = target_contracts[target_source]
    except KeyError as error:
        raise ValueError(
            f"unsupported actor target source: {target_source!r}"
        ) from error

    def eligible(ref):
        metadata = ref.metadata
        if metadata.get("schema_version") != LIVE_SCHEMA_VERSION:
            raise RuntimeError("v5 actor sample schema does not match manifest")
        if metadata.get("objective_fingerprint") != objective_fingerprint:
            raise RuntimeError("v5 actor sample objective fingerprint mismatch")
        if metadata.get("history_sampling_policy") != NAVILA_HISTORY_SAMPLING_POLICY:
            raise RuntimeError("v5 actor sample history policy mismatch")
        if target_source == "oracle":
            return bool(
                metadata.get("oracle_eligible", False)
                and isinstance(metadata.get(target_field), int)
            )
        return bool(
            metadata.get("actor_distillation_eligible", False)
            and metadata.get("actor_teacher_interface")
            == "navila-greedy-text-v1"
            and isinstance(metadata.get(target_field), int)
        )

    refs = [
        ref
        for ref in iter_sample_refs(args.dataset_dir, args.split)
        if eligible(ref)
    ]
    counts = _validate_v5_refs(
        refs,
        allow_small_dataset=args.allow_small_dataset,
        target_field=target_field,
    )
    audit_episodes_per_scene = (
        args.dev_episodes_per_scene
        if args.dev_episodes_per_scene is not None
        else args.audit_episodes_per_scene
    )
    train_refs, calibration_refs, audit_refs = split_actor_partitions(
        refs,
        seed=args.sampling_seed,
        calibration_episodes_per_scene=args.calibration_episodes_per_scene,
        audit_episodes_per_scene=audit_episodes_per_scene,
        target_field=target_field,
    )
    sample_count = args.max_samples or len(train_refs)
    schedule = stratified_actor_schedule(
        train_refs,
        sample_count=sample_count,
        stop_fraction=args.stop_fraction,
        seed=args.sampling_seed,
        hard_stop_negative_fraction=args.hard_stop_negative_fraction,
        hard_stop_negative_margin_m=args.hard_stop_negative_margin_m,
        target_field=target_field,
    )
    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        dtype=training_dtype,
        checkpoint=None,
        actor_architecture=args.actor_architecture,
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
        model,
        preprocessor,
        schedule,
        cache_path,
        args.model_path,
        target_field=target_field,
    )
    _head_warmup(model, cache, args)
    optimizer, updates = _finetune_lora(
        model,
        preprocessor,
        schedule,
        lora_parameters,
        args,
        target_field=target_field,
    )
    calibration_predictions = _collect_predictions(
        model,
        preprocessor,
        calibration_refs,
        label="calibration",
        target_field=target_field,
    )
    if args.calibrate_stop_threshold:
        threshold_selection = calibrate_stop_threshold(
            calibration_predictions["stop_probabilities"],
            calibration_predictions["motion_predictions"],
            calibration_predictions["targets"],
            minimum_stop_recall=args.minimum_stop_accuracy,
            maximum_false_stop_rate=args.maximum_false_stop_rate,
            grid_step=args.stop_threshold_grid_step,
        )
    else:
        calibration_at_fixed_threshold = _audit_predictions(
            calibration_predictions, args.stop_threshold
        )
        threshold_selection = {
            "accepted": bool(
                calibration_at_fixed_threshold["stop_recall"] is not None
                and calibration_at_fixed_threshold["stop_recall"]
                >= args.minimum_stop_accuracy
                and calibration_at_fixed_threshold[
                    "false_stop_rate_non_goal"
                ]
                <= args.maximum_false_stop_rate
            ),
            "selected_threshold": float(args.stop_threshold),
            "selected_metrics": {
                key: calibration_at_fixed_threshold[key]
                for key in (
                    "stop_recall",
                    "false_stop_rate_non_goal",
                    "non_stop_macro_accuracy",
                    "macro_accuracy",
                )
            },
            "grid_step": None,
            "curve": [],
        }
    selected_threshold = float(threshold_selection["selected_threshold"])
    model.stop_threshold = selected_threshold
    calibration_audit = _audit_predictions(
        calibration_predictions, selected_threshold
    )
    _write_diagnostics(
        output_dir,
        "calibration",
        calibration_predictions,
        {
            "threshold_selection": threshold_selection,
            "selected_threshold_audit": calibration_audit,
        },
    )

    audit = None
    audit_predictions = None
    if threshold_selection["accepted"]:
        audit_predictions = _collect_predictions(
            model,
            preprocessor,
            audit_refs,
            label="audit",
            target_field=target_field,
        )
        audit = _audit_predictions(audit_predictions, selected_threshold)
        _write_diagnostics(output_dir, "audit", audit_predictions, audit)
    all_motion_classes_present = bool(
        audit is not None
        and all(str(action_id) in audit["class_samples"] for action_id in range(9))
    )
    accepted = bool(
        threshold_selection["accepted"]
        and audit is not None
        and all_motion_classes_present
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
        "actor_architecture": args.actor_architecture,
        "actor_target_source": target_source,
        "stop_threshold": selected_threshold,
        "updates": updates,
        "sampling_seed": args.sampling_seed,
        "sampling_strategy": "stratified",
        "sampling": sampling_summary(schedule, action_field=target_field),
        "hard_stop_negative_fraction": args.hard_stop_negative_fraction,
        "hard_stop_negative_margin_m": args.hard_stop_negative_margin_m,
        "dataset_action_counts": {
            str(key): value for key, value in sorted(counts.items())
        },
        "train_episodes": _episode_count(train_refs),
        "calibration_episodes": _episode_count(calibration_refs),
        "audit_episodes": _episode_count(audit_refs),
        "train_scenes": _scene_count(train_refs),
        "calibration_scenes": _scene_count(calibration_refs),
        "audit_scenes": _scene_count(audit_refs),
        "calibration_episode_ids": sorted(
            {str(ref.metadata.get("episode_id")) for ref in calibration_refs}
        ),
        "audit_episode_ids": sorted(
            {str(ref.metadata.get("episode_id")) for ref in audit_refs}
        ),
        "actor/calibration_threshold_accepted": threshold_selection[
            "accepted"
        ],
        "actor/calibration_selected_threshold": selected_threshold,
        "actor/calibration_metrics": threshold_selection["selected_metrics"],
        "actor/audit_stop_accuracy": (
            audit["stop_recall"] if audit is not None else None
        ),
        "actor/audit_stop_recall": (
            audit["stop_recall"] if audit is not None else None
        ),
        "actor/audit_false_stop_rate_non_goal": (
            audit["false_stop_rate_non_goal"] if audit is not None else None
        ),
        "actor/audit_non_stop_macro_accuracy": (
            audit["non_stop_macro_accuracy"] if audit is not None else None
        ),
        "actor/audit_per_class_accuracy": (
            audit["per_class_accuracy"] if audit is not None else {}
        ),
        "actor/audit_class_samples": (
            audit["class_samples"] if audit is not None else {}
        ),
        "actor/audit_confusion_matrix": (
            audit["confusion_matrix"] if audit is not None else None
        ),
        "actor/audit_all_motion_classes_present": (
            all_motion_classes_present
        ),
        "actor/audit_probabilities_normalized": bool(
            audit is not None and audit["probabilities_normalized"]
        ),
        "actor/audit_independent": True,
        "actor/audit_target_source": audit_target_source,
        "actor/goal_stop_contract": (
            "sensor-gated-v1"
            if target_source == "navila-policy"
            else "policy-v1"
        ),
        "actor/minimum_stop_accuracy": args.minimum_stop_accuracy,
        "actor/maximum_false_stop_rate": args.maximum_false_stop_rate,
        "actor/minimum_non_stop_macro_accuracy": (
            args.minimum_non_stop_macro_accuracy
        ),
        "actor/accepted": accepted,
        "checkpoint_contract_version": SAFE_CHECKPOINT_CONTRACT_VERSION,
        "checkpoint_role": (
            CHECKPOINT_ROLE_POLICY if accepted else CHECKPOINT_ROLE_DIAGNOSTIC
        ),
        "policy_interface": POLICY_INTERFACE_SAFE_DISCRETE,
    }
    print(json.dumps({"mode": "warmup-actor-audit", **state}), flush=True)
    save_checkpoint(model, optimizer, output_dir, state)
    if not accepted:
        reason = (
            "calibration found no threshold satisfying STOP constraints"
            if not threshold_selection["accepted"]
            else "independent audit failed"
        )
        raise RuntimeError(
            f"v5 actor {reason}; checkpoint is diagnostic-only"
        )
    return 0


def online_dagger_actor(args, training_dtype):
    """Continue a certified Actor on states reached by that Actor itself."""
    if args.epochs <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("online DAgger epochs and accumulation must be positive")
    if args.max_samples is not None and args.max_samples < 2:
        raise ValueError("online DAgger max_samples must be at least two")
    if args.actor_lr <= 0 or args.head_lr <= 0 or args.max_grad_norm <= 0:
        raise ValueError("online DAgger optimization parameters must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite online DAgger output: {output_dir}")

    checkpoint = Path(args.checkpoint)
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError("online DAgger checkpoint has no trainer_state.json")
    source_state = json.loads(state_path.read_text(encoding="utf-8"))
    require_safe_policy_checkpoint(
        source_state, context="online DAgger"
    )
    if source_state.get("actor/goal_stop_contract") != "sensor-gated-v1":
        raise RuntimeError("online DAgger requires a sensor-gated Actor checkpoint")
    objective_fingerprint = source_state.get("objective_fingerprint")
    if not objective_fingerprint:
        raise RuntimeError("online DAgger checkpoint lacks an objective fingerprint")

    def eligible(ref, *, online):
        metadata = ref.metadata
        if metadata.get("schema_version") != LIVE_SCHEMA_VERSION:
            return False
        if metadata.get("objective_fingerprint") != objective_fingerprint:
            return False
        if metadata.get("history_sampling_policy") != NAVILA_HISTORY_SAMPLING_POLICY:
            return False
        if not (
            metadata.get("oracle_eligible", False)
            and isinstance(metadata.get("oracle_action_id"), int)
        ):
            return False
        if not online:
            return True
        return bool(
            metadata.get("dataset_role") == "train"
            and metadata.get("collection_policy") == "vlm"
            and isinstance(metadata.get("policy_action_id"), int)
            and metadata.get("strict_observation_state_alignment", False)
            and metadata.get("online_dagger_eligible", False)
        )

    online_refs = [
        ref for ref in iter_sample_refs(args.rollout_dir, args.split)
        if eligible(ref, online=True)
    ]
    anchor_refs = [
        ref for ref in iter_sample_refs(args.anchor_dataset_dir, args.split)
        if eligible(ref, online=False)
    ]
    if not online_refs:
        raise RuntimeError("online DAgger found no strict on-policy recovery samples")
    _validate_v5_refs(anchor_refs, allow_small_dataset=args.allow_small_dataset)
    sample_count = args.max_samples or (len(online_refs) + len(anchor_refs))

    def schedule_for_epoch(epoch):
        return online_dagger_schedule(
            online_refs,
            anchor_refs,
            sample_count=sample_count,
            online_fraction=args.online_fraction,
            seed=args.sampling_seed + epoch,
        )

    schedule = schedule_for_epoch(0)
    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        dtype=training_dtype,
        checkpoint=str(checkpoint),
    )
    if model.actor_head is None:
        raise RuntimeError("online DAgger requires a hierarchical Actor checkpoint")
    model.reward_head.requires_grad_(False)
    model.cost_head.requires_grad_(False)
    lora_parameters = [
        parameter for parameter in model.base_model.parameters()
        if parameter.requires_grad
    ]
    if not lora_parameters:
        raise RuntimeError("online DAgger checkpoint has no trainable LoRA parameters")
    optimizer, updates = _finetune_lora(
        model,
        preprocessor,
        schedule,
        lora_parameters,
        args,
        schedule_for_epoch=schedule_for_epoch,
    )
    online_identities = {ref.identity for ref in online_refs}
    online_schedule = [
        ref for ref in schedule if ref.identity in online_identities
    ]
    state = dict(source_state)
    state.update(
        {
            "mode": "online-dagger-actor",
            "training_dtype": args.training_dtype,
            "fresh_lora": False,
            "source_checkpoint": str(checkpoint.resolve()),
            "online_round": args.online_round,
            "updates": updates,
            "sampling_seed": args.sampling_seed,
            "sampling": sampling_summary(schedule),
            "online_samples_available": len(online_refs),
            "anchor_samples_available": len(anchor_refs),
            "online_fraction": args.online_fraction,
            "online_schedule_samples": len(online_schedule),
            "online_weight_counts": {
                str(weight): sum(
                    online_dagger_weight(ref.metadata) == weight
                    for ref in online_refs
                )
                for weight in (1, 2, 4)
            },
            "actor/accepted": False,
            "actor/requires_sensor_gated_audit": True,
            "checkpoint_contract_version": SAFE_CHECKPOINT_CONTRACT_VERSION,
            "checkpoint_role": CHECKPOINT_ROLE_DIAGNOSTIC,
            "policy_interface": POLICY_INTERFACE_SAFE_DISCRETE,
        }
    )
    save_checkpoint(model, optimizer, output_dir, state)
    config_path = output_dir / "actor_config.json"
    actor_config = json.loads(config_path.read_text(encoding="utf-8"))
    actor_config["goal_stop_contract"] = "sensor-gated-v1"
    config_path.write_text(json.dumps(actor_config, indent=2), encoding="utf-8")
    print(json.dumps({"mode": "online-dagger-actor", **state}), flush=True)
    return 0


def diagnose_hierarchical_actor(args, manifest, training_dtype):
    """Audit an existing checkpoint and optionally certify a sensor-gated copy."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_fingerprint = manifest.get("objective_fingerprint")

    def eligible(ref):
        metadata = ref.metadata
        if metadata.get("schema_version") != LIVE_SCHEMA_VERSION:
            raise RuntimeError("actor sample schema does not match manifest")
        if metadata.get("objective_fingerprint") != objective_fingerprint:
            raise RuntimeError("actor sample objective fingerprint mismatch")
        if metadata.get("history_sampling_policy") != NAVILA_HISTORY_SAMPLING_POLICY:
            raise RuntimeError("actor sample history policy mismatch")
        return bool(
            metadata.get("oracle_eligible", False)
            and metadata.get("oracle_action_id") is not None
        )

    refs = [
        ref
        for ref in iter_sample_refs(args.dataset_dir, args.split)
        if eligible(ref)
    ]
    _validate_v5_refs(refs, allow_small_dataset=args.allow_small_dataset)
    checkpoint_state_path = Path(args.checkpoint) / "trainer_state.json"
    checkpoint_state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
    calibration_ids = set(checkpoint_state.get("calibration_episode_ids", []))
    audit_ids = set(checkpoint_state.get("audit_episode_ids", []))
    if calibration_ids and audit_ids:
        calibration_refs = [
            ref for ref in refs
            if str(ref.metadata.get("episode_id")) in calibration_ids
        ]
        diagnostic_refs = [
            ref for ref in refs
            if str(ref.metadata.get("episode_id")) in audit_ids
        ]
    else:
        _, calibration_refs, diagnostic_refs = split_actor_partitions(
            refs,
            seed=args.sampling_seed,
            calibration_episodes_per_scene=1,
            audit_episodes_per_scene=args.dev_episodes_per_scene,
        )
    model, preprocessor = load_safe_navila(
        args.model_path,
        device=args.device,
        dtype=training_dtype,
        checkpoint=args.checkpoint,
        actor_architecture=None,
        stop_threshold=None,
    )
    if model.actor_head is None:
        raise RuntimeError("audit-actor requires a hierarchical checkpoint")
    calibration_predictions = _collect_predictions(
        model, preprocessor, calibration_refs, label="diagnostic-calibration"
    )
    predictions = _collect_predictions(
        model, preprocessor, diagnostic_refs, label="diagnostic"
    )
    selection = calibrate_stop_threshold(
        calibration_predictions["stop_probabilities"],
        calibration_predictions["motion_predictions"],
        calibration_predictions["targets"],
        minimum_stop_recall=(
            0.0
            if args.goal_stop_contract == "sensor-gated-v1"
            else args.minimum_stop_accuracy
        ),
        maximum_false_stop_rate=args.maximum_false_stop_rate,
        grid_step=args.stop_threshold_grid_step,
    )
    selected_audit = _audit_predictions(
        predictions, selection["selected_threshold"]
    )
    motion_audit = _motion_only_audit(predictions)
    accepted = bool(
        args.goal_stop_contract == "sensor-gated-v1"
        and motion_audit["all_motion_classes_present"]
        and motion_audit["probabilities_normalized"]
        and motion_audit["macro_accuracy"] is not None
        and motion_audit["macro_accuracy"]
        >= args.minimum_non_stop_macro_accuracy
    )
    report = {
        "diagnostic_only": not (args.certify and accepted),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "goal_stop_contract": args.goal_stop_contract,
        "episodes": _episode_count(diagnostic_refs),
        "scenes": _scene_count(diagnostic_refs),
        "threshold_selection": selection,
        "selected_threshold_audit": selected_audit,
        "motion_only_audit": motion_audit,
        "minimum_non_stop_macro_accuracy": args.minimum_non_stop_macro_accuracy,
        "accepted": accepted,
    }
    _write_diagnostics(output_dir, "diagnostic", predictions, report)
    if args.certify:
        if args.goal_stop_contract != "sensor-gated-v1":
            raise ValueError("--certify requires --goal-stop-contract=sensor-gated-v1")
        if not accepted:
            raise RuntimeError("sensor-gated actor audit failed; checkpoint remains diagnostic-only")
        source = Path(args.checkpoint)
        for name in (
            "README.md",
            "action_space.json",
            "actor_head.pt",
            "adapter_config.json",
            "adapter_model.safetensors",
            "cost_critic.pt",
            "reward_critic.pt",
        ):
            path = source / name
            if path.is_file():
                shutil.copy2(path, output_dir / name)
        actor_config = json.loads((source / "actor_config.json").read_text(encoding="utf-8"))
        actor_config["stop_threshold"] = selection["selected_threshold"]
        actor_config["goal_stop_contract"] = args.goal_stop_contract
        (output_dir / "actor_config.json").write_text(
            json.dumps(actor_config, indent=2), encoding="utf-8"
        )
        certified_state = dict(checkpoint_state)
        certified_state.update(
            {
                "stop_threshold": selection["selected_threshold"],
                "actor/goal_stop_contract": args.goal_stop_contract,
                "actor/motion_only_audit": motion_audit,
                "actor/audit_target_source": checkpoint_state.get(
                    "actor/audit_target_source", "dynamic-oracle"
                ),
                "actor/minimum_non_stop_macro_accuracy": args.minimum_non_stop_macro_accuracy,
                "actor/audit_independent": True,
                "actor/accepted": True,
                "actor/certified_from": str(source.resolve()),
                "checkpoint_contract_version": SAFE_CHECKPOINT_CONTRACT_VERSION,
                "checkpoint_role": CHECKPOINT_ROLE_POLICY,
                "policy_interface": POLICY_INTERFACE_SAFE_DISCRETE,
            }
        )
        (output_dir / "trainer_state.json").write_text(
            json.dumps(certified_state, indent=2), encoding="utf-8"
        )
    print(json.dumps({"mode": "audit-actor", **report}), flush=True)
    return 0
