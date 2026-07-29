from collections import Counter

import pytest
import torch

from safe_vln.actor_training import (
    audit_hierarchical_actor,
    hierarchical_actor_loss,
    metadata_of,
    split_actor_episodes,
    stratified_actor_schedule,
)


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


def test_actor_schedule_has_fixed_stop_share_and_uniform_motion():
    schedule = stratified_actor_schedule(
        _dataset(), sample_count=100, stop_fraction=0.25, seed=11
    )
    counts = Counter(item["oracle_action_id"] for item in schedule)
    assert counts[9] == 25
    assert sum(counts[action_id] for action_id in range(9)) == 75
    assert max(counts[action_id] for action_id in range(9)) - min(
        counts[action_id] for action_id in range(9)
    ) <= 1
    assert all(
        sum(item["oracle_action_id"] == 9 for item in schedule[start : start + 4]) == 1
        for start in range(0, len(schedule), 4)
    )


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
