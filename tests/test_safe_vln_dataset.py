import json
import tarfile

from PIL import Image
import pytest

from safe_vln.dataset import SafeVLNShardWriter, iter_samples
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


def test_writer_rejects_appending_a_different_objective(tmp_path):
    objective = build_objective_config(default_cost_profile())
    SafeVLNShardWriter(tmp_path, objective_config=objective).close()
    changed = dict(objective)
    changed["fingerprint"] = "different"
    with pytest.raises(ValueError, match="objective fingerprint"):
        SafeVLNShardWriter(tmp_path, objective_config=changed)
