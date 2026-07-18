import json
import tarfile

from PIL import Image

from safe_vln.dataset import SafeVLNShardWriter, iter_samples


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
