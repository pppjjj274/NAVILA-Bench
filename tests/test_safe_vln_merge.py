import json
import sys

from PIL import Image

from safe_vln.dataset import SafeVLNEpisodeWriter, iter_metadata
from safe_vln.live_render import LIVE_SCHEMA_VERSION
from safe_vln.objective import build_objective_config, default_cost_profile
from scripts import merge_safe_vln_datasets


def _write_episode(root, episode_id, objective, *, source_hash=None):
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    with SafeVLNEpisodeWriter(
        root,
        episode_id,
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        metadata = {
            "episode_id": str(episode_id),
            "index": 0,
            "instruction": "go",
            "schema_version": LIVE_SCHEMA_VERSION,
            "objective_fingerprint": objective["fingerprint"],
        }
        if source_hash is not None:
            metadata.update(
                {
                    "observation_alignment": "native_isaac_camera",
                    "vlnce_source_split": "train",
                    "vlnce_source_dataset_role": "train",
                    "vlnce_coordinate_system": "isaac_xyz_z_up_wxyz",
                    "vlnce_source_metadata_sha256": source_hash,
                    "vlnce_source_gt_sha256": "f" * 64,
                }
            )
        writer.add(
            f"episode{episode_id}/state000000",
            frames,
            metadata,
        )


def test_merge_publishes_only_compatible_transactional_sources(
    tmp_path, monkeypatch, capsys
):
    objective = build_objective_config(default_cost_profile())
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _write_episode(first, "1", objective)
    _write_episode(second, "2", objective)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_safe_vln_datasets.py",
            "--source-dir",
            str(first),
            "--source-dir",
            str(second),
            "--output-dir",
            str(output),
        ],
    )
    assert merge_safe_vln_datasets.main() is None
    report = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["episodes"] == 2
    assert manifest["completed_episodes"] == 2
    assert manifest["total_samples"] == 2
    assert {row["episode_id"] for row in iter_metadata(output)} == {"1", "2"}


def test_merge_rejects_different_native_source_provenance(
    tmp_path, monkeypatch
):
    objective = build_objective_config(default_cost_profile())
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_episode(first, "1", objective, source_hash="a" * 64)
    _write_episode(second, "2", objective, source_hash="b" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_safe_vln_datasets.py",
            "--source-dir",
            str(first),
            "--source-dir",
            str(second),
            "--output-dir",
            str(tmp_path / "merged"),
        ],
    )

    try:
        merge_safe_vln_datasets.main()
    except RuntimeError as error:
        assert "different VLN-CE sources" in str(error)
    else:
        raise AssertionError("merge accepted different native sources")
