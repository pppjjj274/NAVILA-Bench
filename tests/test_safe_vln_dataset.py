import json

from PIL import Image
import pytest

from safe_vln.dataset import (
    SafeVLNEpisodeWriter,
    SafeVLNShardWriter,
    _encode_jpeg,
    iter_samples,
)
from safe_vln.live_render import LIVE_SCHEMA_VERSION
from safe_vln.objective import build_objective_config, default_cost_profile


def test_atomic_shard_round_trip(tmp_path):
    frames = [Image.new("RGB", (8, 8), (index, 0, 0)) for index in range(8)]
    metadata = {"episode_id": "1", "instruction": "go", "reward_return": 1.0, "cost_return": 0.0}
    with SafeVLNShardWriter(tmp_path, samples_per_shard=1) as writer:
        writer.add("episode1/state000000", frames, metadata)

    assert not list(tmp_path.glob("*.incomplete"))
    shards = list(tmp_path.glob("train-*.tar"))
    assert len(shards) == 1
    loaded_frames, loaded_metadata = next(iter_samples(tmp_path))
    assert len(loaded_frames) == 8
    assert loaded_metadata == metadata
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "safe-vln-go2-v1"
    assert manifest["samples_written_this_run"] == 1
    assert manifest["total_samples"] == 1


def test_shard_writer_rejects_duplicate_sample_keys(tmp_path):
    frames = [Image.new("RGB", (8, 8)) for _ in range(8)]
    writer = SafeVLNShardWriter(tmp_path)
    writer.add("episode1/state000000", frames, {"episode_id": "1"})
    with pytest.raises(ValueError, match="duplicate sample key"):
        writer.add("episode1/state000000", frames, {"episode_id": "1"})


def test_shard_writer_rejects_duplicate_keys_from_an_earlier_run(tmp_path):
    frames = [Image.new("RGB", (8, 8)) for _ in range(8)]
    with SafeVLNShardWriter(tmp_path) as writer:
        writer.add("episode1/state000000", frames, {"episode_id": "1"})

    with pytest.raises(ValueError, match="duplicate sample key"):
        with SafeVLNShardWriter(tmp_path) as writer:
            writer.add("episode1/state000000", frames, {"episode_id": "1"})


def test_shard_writer_rejects_non_finite_metadata(tmp_path):
    frames = [Image.new("RGB", (8, 8)) for _ in range(8)]
    with pytest.raises(ValueError, match="Out of range float values"):
        with SafeVLNShardWriter(tmp_path) as writer:
            writer.add("episode1/state000000", frames, {"cost": float("nan")})


def test_jpeg_encoder_does_not_use_pillow_save_when_opencv_is_available(
    monkeypatch,
):
    frame = Image.new("RGB", (8, 8), (1, 2, 3))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Pillow JPEG encoder was called")

    monkeypatch.setattr(Image.Image, "save", fail_if_called)
    payload = _encode_jpeg(frame, 90)
    assert payload.startswith(b"\xff\xd8")
    assert payload.endswith(b"\xff\xd9")


def test_writer_rejects_appending_a_different_objective(tmp_path):
    objective = build_objective_config(default_cost_profile())
    SafeVLNShardWriter(tmp_path, objective_config=objective).close()
    changed = dict(objective)
    changed["fingerprint"] = "different"
    with pytest.raises(ValueError, match="objective fingerprint"):
        SafeVLNShardWriter(tmp_path, objective_config=changed)


def test_transactional_episode_is_visible_only_after_commit(tmp_path):
    objective = build_objective_config(default_cost_profile())
    frames = [Image.new("RGB", (8, 8), (index, 0, 0)) for index in range(8)]
    writer = SafeVLNEpisodeWriter(
        tmp_path,
        "episode-1",
        dataset_role="eval",
        split="val_unseen",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    )
    writer.add(
        "episode-1/state000000",
        frames,
        {
            "episode_id": "episode-1",
            "instruction": "go",
            "schema_version": LIVE_SCHEMA_VERSION,
        },
    )
    assert list(iter_samples(tmp_path, "val_unseen")) == []
    writer.commit()

    loaded = list(iter_samples(tmp_path, "val_unseen"))
    assert len(loaded) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset_role"] == "eval"
    assert manifest["schema_version"] == LIVE_SCHEMA_VERSION
    assert manifest["completed_episodes"] == 1


def test_transactional_episode_abort_publishes_nothing(tmp_path):
    objective = build_objective_config(default_cost_profile())
    writer = SafeVLNEpisodeWriter(
        tmp_path,
        "bad",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    )
    writer.abort()
    assert not (tmp_path / "manifest.json").exists()
    assert not list((tmp_path / "completed").glob("*"))


def test_transactional_episode_rejects_empty_commit(tmp_path):
    writer = SafeVLNEpisodeWriter(
        tmp_path,
        "empty",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=build_objective_config(default_cost_profile()),
    )
    with pytest.raises(RuntimeError, match="empty Safe-VLN episode"):
        writer.commit()
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "completed" / "empty").exists()
