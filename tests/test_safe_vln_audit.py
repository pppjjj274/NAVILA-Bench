import json
import sys
from io import BytesIO
import tarfile

from PIL import Image

from safe_vln.dataset import SafeVLNEpisodeWriter
from safe_vln.live_render import (
    LIVE_SCHEMA_VERSION,
    NAVILA_HISTORY_SAMPLING_POLICY,
)
from safe_vln.objective import build_objective_config, default_cost_profile
from scripts import audit_safe_vln_v5


def _teacher_metadata(objective, *, index=0, done=True):
    observation_key = f"episode-1/state{index:06d}"
    return {
        "schema_version": LIVE_SCHEMA_VERSION,
        "objective_fingerprint": objective["fingerprint"],
        "episode_id": "episode-1",
        "scene_id": "scene-1",
        "index": index,
        "done": done,
        "instruction": "go forward",
        "action_id": 8,
        "observation_key": observation_key,
        "strict_observation_state_alignment": True,
        "history_sampling_policy": NAVILA_HISTORY_SAMPLING_POLICY,
        "frame_alignment": [
            {
                "history_padding": frame_index < 7,
                "strict_observation_state_alignment": frame_index == 7,
            }
            for frame_index in range(8)
        ],
        "safety_diagnostics": {
            "contact_sensor_enabled": True,
            "turn_execution": {"active": False, "blocked": False},
        },
        "collection_policy": "vlm",
        "policy_interface": "navila-greedy-text-v1",
        "actor_distillation_eligible": True,
        "actor_teacher_interface": "navila-greedy-text-v1",
        "actor_teacher_action_id": 8,
    }


def _native_teacher_metadata(objective):
    metadata = _teacher_metadata(objective)
    metadata.update(
        {
            "observation_alignment": "native_isaac_camera",
            "dataset_role": "train",
            "vlnce_source_split": "train",
            "vlnce_source_dataset_role": "train",
            "vlnce_coordinate_system": "isaac_xyz_z_up_wxyz",
            "vlnce_source_metadata_sha256": "a" * 64,
            "vlnce_source_gt_sha256": "b" * 64,
        }
    )
    pose = {
        "position": [1.0, 2.0, 3.0],
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    for frame in metadata["frame_alignment"]:
        if not frame["history_padding"]:
            frame["isaac_pose"] = pose
            frame["camera_pose"] = pose
    return metadata


def _audit(tmp_path, monkeypatch, capsys, *extra_args):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_safe_vln_v5.py",
            "--dataset-dir",
            str(tmp_path),
            "--allow-small-dataset",
            "--require-navila-teacher",
            *extra_args,
        ],
    )
    status = audit_safe_vln_v5.main()
    return status, json.loads(capsys.readouterr().out)


def test_allow_small_still_enforces_explicit_episode_count(
    tmp_path, monkeypatch, capsys
):
    objective = build_objective_config(default_cost_profile())
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    metadata = _teacher_metadata(objective)
    with SafeVLNEpisodeWriter(
        tmp_path,
        "episode-1",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        writer.add(metadata["observation_key"], frames, metadata)

    status, report = _audit(
        tmp_path,
        monkeypatch,
        capsys,
        "--expected-episodes",
        "2",
        "--expected-scenes",
        "1",
    )
    assert status == 1
    assert "episodes=1 expected=2" in report["failures"]
    assert report["expected_episodes"] == 2
    assert report["expected_scenes"] == 1


def test_strict_audit_accepts_transactional_navila_teacher_dataset(
    tmp_path, monkeypatch, capsys
):
    objective = build_objective_config(default_cost_profile())
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    metadata = _teacher_metadata(objective)
    with SafeVLNEpisodeWriter(
        tmp_path,
        "episode-1",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        writer.add(metadata["observation_key"], frames, metadata)

    status, report = _audit(tmp_path, monkeypatch, capsys)
    assert status == 0
    assert report["accepted"] is True
    assert report["actor_distillation_eligible_samples"] == 1
    assert report["action_count_source"] == "actor_teacher_action_id"
    assert report["action_counts"]["8"] == 1


def test_strict_audit_rejects_missing_image_member(tmp_path, monkeypatch, capsys):
    objective = build_objective_config(default_cost_profile())
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    metadata = _teacher_metadata(objective)
    with SafeVLNEpisodeWriter(
        tmp_path,
        "episode-1",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        writer.add(metadata["observation_key"], frames, metadata)

    shard = next(tmp_path.glob("completed/*/train-*.tar"))
    retained = []
    with tarfile.open(shard, "r") as archive:
        for member in archive:
            if member.name.endswith(".3.jpg"):
                continue
            extracted = archive.extractfile(member)
            retained.append((member, extracted.read() if extracted else b""))
    with tarfile.open(shard, "w") as archive:
        for member, payload in retained:
            archive.addfile(member, BytesIO(payload))

    status, report = _audit(tmp_path, monkeypatch, capsys)
    assert status == 1
    assert any(
        "tar members missing or duplicated" in failure
        for failure in report["failures"]
    )


def test_strict_audit_rejects_done_before_final_transition(
    tmp_path, monkeypatch, capsys
):
    objective = build_objective_config(default_cost_profile())
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    with SafeVLNEpisodeWriter(
        tmp_path,
        "episode-1",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        for index in range(2):
            metadata = _teacher_metadata(objective, index=index, done=True)
            writer.add(metadata["observation_key"], frames, metadata)

    status, report = _audit(tmp_path, monkeypatch, capsys)
    assert status == 1
    assert any(
        "incomplete or non-contiguous" in failure
        for failure in report["failures"]
    )


def test_strict_audit_rejects_terminal_transition_with_next_observation(
    tmp_path, monkeypatch, capsys
):
    objective = build_objective_config(default_cost_profile())
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    metadata = _teacher_metadata(objective)
    metadata["next_observation_key"] = "episode-1/state000001"
    with SafeVLNEpisodeWriter(
        tmp_path,
        "episode-1",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        writer.add(metadata["observation_key"], frames, metadata)

    status, report = _audit(tmp_path, monkeypatch, capsys)
    assert status == 1
    assert any(
        "broken observation link" in failure for failure in report["failures"]
    )


def test_strict_audit_accepts_native_source_and_camera_pose_provenance(
    tmp_path, monkeypatch, capsys
):
    objective = build_objective_config(default_cost_profile())
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    metadata = _native_teacher_metadata(objective)
    with SafeVLNEpisodeWriter(
        tmp_path,
        "episode-1",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        writer.add(metadata["observation_key"], frames, metadata)

    status, report = _audit(tmp_path, monkeypatch, capsys)
    assert status == 0
    assert report["accepted"] is True


def test_strict_audit_rejects_native_frame_without_camera_pose(
    tmp_path, monkeypatch, capsys
):
    objective = build_objective_config(default_cost_profile())
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    metadata = _native_teacher_metadata(objective)
    metadata["frame_alignment"][-1].pop("camera_pose")
    with SafeVLNEpisodeWriter(
        tmp_path,
        "episode-1",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        writer.add(metadata["observation_key"], frames, metadata)

    status, report = _audit(tmp_path, monkeypatch, capsys)
    assert status == 1
    assert any(
        "native frame pose provenance is invalid" in failure
        for failure in report["failures"]
    )
