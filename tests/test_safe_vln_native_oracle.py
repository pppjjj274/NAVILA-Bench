import json
import sys

from PIL import Image

from safe_vln.dataset import SafeVLNEpisodeWriter, iter_metadata
from safe_vln.live_render import (
    LIVE_SCHEMA_VERSION,
    NAVILA_HISTORY_SAMPLING_POLICY,
)
from safe_vln.objective import build_objective_config, default_cost_profile
from scripts import apply_native_oracle_labels


def test_apply_native_oracle_preserves_transactional_episode_contract(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    objective = build_objective_config(default_cost_profile())
    pose = {
        "position": [1.0, 2.0, 0.3],
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "yaw": 0.0,
    }
    alignment = [
        {
            "history_padding": index < 7,
            "strict_observation_state_alignment": index == 7,
        }
        for index in range(8)
    ]
    contract = {
        "source_split": "train",
        "dataset_role": "train",
        "coordinate_system": "isaac_xyz_z_up_wxyz",
        "source_metadata_sha256": "a" * 64,
        "source_gt_sha256": "b" * 64,
    }
    metadata = {
        "schema_version": LIVE_SCHEMA_VERSION,
        "objective_fingerprint": objective["fingerprint"],
        "episode_id": "1",
        "scene_id": "mp3d/scene/scene.glb",
        "index": 0,
        "done": True,
        "next_observation_key": None,
        "observation_key": "episode1/state000000",
        "history_sampling_policy": NAVILA_HISTORY_SAMPLING_POLICY,
        "frame_alignment": alignment,
        "isaac_pose_before": pose,
        "vlnce_source_split": contract["source_split"],
        "vlnce_source_dataset_role": contract["dataset_role"],
        "vlnce_coordinate_system": contract["coordinate_system"],
        "vlnce_source_metadata_sha256": contract["source_metadata_sha256"],
        "vlnce_source_gt_sha256": contract["source_gt_sha256"],
    }
    frames = [Image.new("RGB", (8, 8), color=index) for index in range(8)]
    with SafeVLNEpisodeWriter(
        source,
        "1",
        dataset_role="train",
        split="train",
        schema_version=LIVE_SCHEMA_VERSION,
        objective_config=objective,
    ) as writer:
        writer.add(metadata["observation_key"], frames, metadata)

    sidecar = tmp_path / "labels.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "safe-vln-native-oracle-v1",
                "teacher_source": "habitat_navmesh_shortest_path",
                "diagnostic_only": True,
                "source_contract": contract,
                "records": [
                    {
                        "episode_id": "1",
                        "transition_index": 0,
                        "observation_key": metadata["observation_key"],
                        "isaac_pose_before": pose,
                        "frame_alignment": alignment,
                        "source_contract": contract,
                        "oracle_valid": True,
                        "oracle_eligible": True,
                        "oracle_action_id": 8,
                        "oracle_action": {
                            "action_id": 8,
                            "text": "move forward 75 centimeters",
                        },
                        "teacher_source": "habitat_navmesh_shortest_path",
                        "navmesh_metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_native_oracle_labels.py",
            "--source-dir",
            str(source),
            "--labels",
            str(sidecar),
            "--output-dir",
            str(output),
        ],
    )

    assert apply_native_oracle_labels.main() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    row = next(iter_metadata(output))
    assert manifest["completed_episodes"] == 1
    assert manifest["diagnostic_only"] is True
    assert row["oracle_action_id"] == 8
    assert row["teacher_source"] == "habitat_navmesh_shortest_path"
