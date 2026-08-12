from collections import Counter
from pathlib import Path

import pytest
import torch

from safe_vln.actor_training import (
    audit_hierarchical_actor,
    calibrate_stop_threshold,
    factorized_actor_loss,
    hierarchical_actor_loss,
    metadata_of,
    split_actor_episodes,
    split_actor_partitions,
    stratified_actor_schedule,
    target_action_id,
)
from safe_vln.actor_pipeline import (
    _motion_only_audit,
    online_dagger_schedule,
    online_dagger_weight,
)
from safe_vln.dataset import SafeVLNSampleRef


def _dataset():
    samples = []
    for scene in ("a", "b"):
        for episode_index in range(3):
            episode_id = f"{scene}-{episode_index}"
            for action_id in range(10):
                samples.append(
                    {
                        "scene_id": scene,
                        "episode_id": episode_id,
                        "index": action_id,
                        "oracle_action_id": action_id,
                        "system_success": bool(
                            episode_index == 1 and action_id == 9
                        ),
                    }
                )
    return samples


def test_actor_split_holds_out_whole_episodes_per_scene():
    train, dev = split_actor_episodes(_dataset(), seed=7)
    train_ids = {metadata_of(item)["episode_id"] for item in train}
    dev_ids = {metadata_of(item)["episode_id"] for item in dev}
    assert not train_ids & dev_ids
    assert len(dev_ids) == 2
    assert {metadata_of(item)["scene_id"] for item in dev} == {"a", "b"}
    assert dev_ids == {"a-1", "b-1"}


def test_actor_three_way_split_has_no_episode_leakage():
    train, calibration, audit = split_actor_partitions(_dataset(), seed=7)
    partitions = [
        {metadata_of(item)["episode_id"] for item in items}
        for items in (train, calibration, audit)
    ]
    assert all(partitions)
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert all(len(partition) == 2 for partition in partitions)
    assert any(
        item["system_success"] and item["oracle_action_id"] == 9
        for item in calibration
    )
    assert any(
        item["system_success"] and item["oracle_action_id"] == 9
        for item in audit
    )


def test_actor_schedule_has_ten_percent_stop_and_uniform_motion():
    schedule = stratified_actor_schedule(
        _dataset(), sample_count=100, stop_fraction=0.10, seed=11
    )
    counts = Counter(item["oracle_action_id"] for item in schedule)
    assert counts[9] == 10
    assert sum(counts[action_id] for action_id in range(9)) == 90
    assert max(counts[action_id] for action_id in range(9)) - min(
        counts[action_id] for action_id in range(9)
    ) <= 1
    assert all(item["oracle_action_id"] == 9 for item in schedule[4::10])


def test_actor_schedule_can_distill_original_navila_targets():
    samples = []
    for item in _dataset():
        copied = dict(item)
        copied["actor_teacher_action_id"] = copied["oracle_action_id"]
        copied["oracle_action_id"] = (copied["oracle_action_id"] + 1) % 10
        samples.append(copied)
    schedule = stratified_actor_schedule(
        samples,
        sample_count=100,
        stop_fraction=0.10,
        seed=11,
        target_field="actor_teacher_action_id",
    )
    counts = Counter(
        target_action_id(item, "actor_teacher_action_id") for item in schedule
    )
    assert counts[9] == 10
    assert all(counts[action_id] == 10 for action_id in range(9))


def test_actor_target_rejects_missing_or_out_of_range_labels():
    with pytest.raises(ValueError, match="invalid actor_teacher_action_id"):
        target_action_id({}, "actor_teacher_action_id")
    with pytest.raises(ValueError, match="invalid actor_teacher_action_id"):
        target_action_id(
            {"actor_teacher_action_id": 10}, "actor_teacher_action_id"
        )


def test_actor_schedule_includes_boundary_stop_negatives():
    samples = _dataset()
    hard = next(
        item
        for item in samples
        if item["oracle_action_id"] == 0 and item["episode_id"] == "a-0"
    )
    hard.update({"distance_before": 3.5, "goal_radius_m": 3.0})
    schedule = stratified_actor_schedule(
        samples,
        sample_count=100,
        stop_fraction=0.10,
        seed=11,
        hard_stop_negative_fraction=0.25,
        hard_stop_negative_margin_m=1.0,
    )
    assert hard in schedule


def test_hierarchical_loss_separates_stop_and_motion():
    stop_logits = torch.tensor([-2.0, 2.0], requires_grad=True)
    motion_logits = torch.zeros(2, 9, requires_grad=True)
    targets = torch.tensor([3, 9])
    loss, stop_loss, motion_loss = hierarchical_actor_loss(
        stop_logits, motion_logits, targets
    )
    assert loss.item() == pytest.approx(
        stop_loss.item() + motion_loss.item()
    )
    loss.backward()
    assert torch.isfinite(stop_logits.grad).all()
    assert torch.isfinite(motion_logits.grad).all()


def test_factorized_loss_separates_stop_direction_and_magnitude():
    stop_logits = torch.tensor([-2.0, -2.0, 2.0], requires_grad=True)
    direction_logits = torch.zeros(3, 3, requires_grad=True)
    magnitude_logits = torch.zeros(3, 3, 3, requires_grad=True)
    targets = torch.tensor([0, 8, 9])
    losses = factorized_actor_loss(
        stop_logits, direction_logits, magnitude_logits, targets
    )
    total, stop, direction, magnitude_hard, magnitude_ordinal = losses
    assert total.item() == pytest.approx(
        stop.item()
        + direction.item()
        + 0.5 * magnitude_hard.item()
        + 0.5 * magnitude_ordinal.item()
    )
    total.backward()
    assert torch.isfinite(stop_logits.grad).all()
    assert torch.isfinite(direction_logits.grad).all()
    assert torch.isfinite(magnitude_logits.grad).all()


def test_actor_audit_reports_recall_false_stop_and_motion_macro():

    audit = audit_hierarchical_actor(
        [0.9, 0.8, 0.1, 0.7],
        [0, 1, 2, 3],
        [9, 9, 2, 3],
    )
    assert audit["stop_recall"] == 1.0
    assert audit["false_stop_rate_non_goal"] == 0.5
    assert audit["non_stop_macro_accuracy"] == 0.5
    assert audit["finite"] is True
    assert len(audit["confusion_matrix"]) == 10


def test_stop_threshold_calibration_obeys_constraints():
    calibrated = calibrate_stop_threshold(
        [0.90, 0.70, 0.60, 0.20, 0.10, 0.05],
        [0, 0, 0, 1, 1, 1],
        [9, 9, 0, 0, 1, 1],
        minimum_stop_recall=0.5,
        maximum_false_stop_rate=0.25,
        grid_step=0.1,
    )
    assert calibrated["accepted"] is True
    assert calibrated["selected_metrics"]["stop_recall"] >= 0.5
    assert (
        calibrated["selected_metrics"]["false_stop_rate_non_goal"]
        <= 0.25
    )


def test_motion_only_audit_ignores_stop_predictions():
    predictions = {
        "motion_predictions": list(range(9)) + [0],
        "targets": list(range(9)) + [9],
        "probabilities_normalized": True,
    }
    report = _motion_only_audit(predictions)
    assert report["macro_accuracy"] == 1.0
    assert report["all_motion_classes_present"]
    assert "9" not in report["class_samples"]


def test_three_way_split_rejects_scenes_without_train_episode():
    with pytest.raises(ValueError, match="strict three-way split"):
        split_actor_partitions(_dataset()[:10], seed=7)


def _dagger_ref(name, *, oracle, policy, online=True):
    return SafeVLNSampleRef(
        shard_path=Path(f"/{name}.tar"),
        metadata_name=f"{name}.json",
        metadata={
            "episode_id": name,
            "index": 0,
            "oracle_action_id": oracle,
            "policy_action_id": policy,
            "online_dagger_eligible": online,
        },
    )


def test_online_dagger_prioritizes_forward_after_turn_errors():
    forward_after_turn = _dagger_ref("forward-after-turn", oracle=8, policy=0)
    mismatch = _dagger_ref("mismatch", oracle=5, policy=0)
    matched = _dagger_ref("matched", oracle=5, policy=5)
    assert online_dagger_weight(forward_after_turn.metadata) == 4
    assert online_dagger_weight(mismatch.metadata) == 2
    assert online_dagger_weight(matched.metadata) == 1


def test_online_dagger_schedule_mixes_online_recovery_and_anchor_data():
    online = [
        _dagger_ref("forward-after-turn", oracle=8, policy=0),
        _dagger_ref("mismatch", oracle=5, policy=0),
        _dagger_ref("matched", oracle=5, policy=5),
    ]
    anchors = [_dagger_ref(f"anchor-{index}", oracle=index, policy=index) for index in range(3)]
    schedule = online_dagger_schedule(
        online, anchors, sample_count=20, online_fraction=0.60, seed=7
    )
    online_names = {ref.metadata_name for ref in online}
    online_count = sum(ref.metadata_name in online_names for ref in schedule)
    assert online_count == 12
    assert len(schedule) == 20
    assert any(ref.metadata_name.startswith("anchor-") for ref in schedule)
