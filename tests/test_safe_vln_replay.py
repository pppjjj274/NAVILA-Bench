import gzip
import json
import math

import pytest
from PIL import Image

from safe_vln.replay import (
    habitat_heading_to_isaac,
    habitat_position_to_isaac,
    load_r2r_replay_episode,
    load_vlnce_episode_metadata,
)


def _write_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (6, 4), (value, 0, 0)).save(path, format="PNG")


def _write_annotations(root, records):
    root.mkdir(parents=True, exist_ok=True)
    (root / "annotations.json").write_text(json.dumps(records), encoding="utf-8")


def _write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output:
        json.dump(payload, output)


def _record(video_id, frames, *, action="The next action is move forward 25 cm.", query="go"):
    return {"video_id": video_id, "q": query, "a": action, "frames": frames}


def test_loads_exact_episode_deduplicates_and_sorts_numeric_suffix(tmp_path):
    root = tmp_path / "R2R"
    frames = ["5372/frame_0.jpg"]
    _write_image(root / "train" / frames[0], 23)
    step_two = _record("5372-2", frames, action="The next action is turn left 30 degree.")
    records = [
        _record("5372-10", frames),
        step_two,
        _record("53720-0", ["53720/frame_0.jpg"]),
        dict(step_two),
        _record("5372-not-a-number", frames),
    ]
    _write_annotations(root, records)

    episode = load_r2r_replay_episode(root, "5372")

    assert episode.episode_id == "5372"
    assert episode.instruction == "go"
    assert [step.video_id for step in episode] == ["5372-2", "5372-10"]
    assert [step.step_index for step in episode] == [2, 10]
    assert episode[0].oracle_action.action_id == 1
    assert episode[0].oracle_action.text == "turn left 30 degrees"


def test_conflicting_duplicate_video_id_is_rejected(tmp_path):
    root = tmp_path / "R2R"
    records = [
        _record("7-0", ["7/frame_0.jpg"]),
        _record("7-0", ["7/frame_1.jpg"]),
    ]
    _write_annotations(root, records)

    with pytest.raises(ValueError, match="Conflicting duplicate.*7-0"):
        load_r2r_replay_episode(root, 7)


@pytest.mark.parametrize(
    ("frame_count", "expected_values"),
    [
        (1, [1, 1, 1, 1, 1, 1, 1, 1]),
        (7, [1, 1, 2, 3, 4, 5, 6, 7]),
        (8, [1, 2, 3, 4, 5, 6, 7, 8]),
        (15, [1, 3, 5, 7, 9, 11, 13, 15]),
    ],
)
def test_load_frames_matches_navila_eight_frame_sampling(tmp_path, frame_count, expected_values):
    root = tmp_path / "R2R"
    relative_paths = [f"42/frame_{index}.jpg" for index in range(frame_count)]
    for index, relative_path in enumerate(relative_paths, start=1):
        _write_image(root / "train" / relative_path, index)
    _write_annotations(root, [_record("42-0", relative_paths)])

    step = load_r2r_replay_episode(root, 42)[0]
    frames = step.load_frames()

    assert len(frames) == 8
    assert all(frame.mode == "RGB" for frame in frames)
    assert all(frame.size == (6, 4) for frame in frames)
    assert [frame.getpixel((0, 0))[0] for frame in frames] == expected_values
    assert frames[-1].getpixel((0, 0))[0] == frame_count


def test_only_selected_source_images_are_required(tmp_path):
    root = tmp_path / "R2R"
    relative_paths = [f"5/frame_{index}.jpg" for index in range(15)]
    selected_indices = {0, 2, 4, 6, 8, 10, 12, 14}
    for index, relative_path in enumerate(relative_paths):
        if index in selected_indices:
            _write_image(root / "train" / relative_path, index + 1)
    _write_annotations(root, [_record("5-0", relative_paths)])

    frames = load_r2r_replay_episode(root, 5)[0].load_frames()

    assert len(frames) == 8


def test_missing_selected_image_fails_during_episode_load(tmp_path):
    root = tmp_path / "R2R"
    _write_annotations(root, [_record("8-0", ["8/missing.jpg"])])

    with pytest.raises(ValueError, match="missing or unreadable.*missing.jpg"):
        load_r2r_replay_episode(root, 8)


def test_corrupt_selected_image_fails_during_episode_load(tmp_path):
    root = tmp_path / "R2R"
    corrupt = root / "train" / "9" / "frame_0.jpg"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not an image")
    _write_annotations(root, [_record("9-0", ["9/frame_0.jpg"])])

    with pytest.raises(ValueError, match="missing or unreadable.*frame_0.jpg"):
        load_r2r_replay_episode(root, 9)


def test_unknown_episode_and_invalid_oracle_action_are_rejected(tmp_path):
    root = tmp_path / "R2R"
    frame = "1/frame_0.jpg"
    _write_image(root / "train" / frame, 1)
    _write_annotations(root, [_record("1-0", [frame], action="jump")])

    with pytest.raises(ValueError, match="unknown oracle action"):
        load_r2r_replay_episode(root, 1)
    with pytest.raises(ValueError, match="episode '2' was not found"):
        load_r2r_replay_episode(root, 2)


@pytest.mark.parametrize(
    "action",
    [
        "The next action is turn left 90 degree.",
        "The next action is move forward 100 cm.",
        "Please stop now.",
    ],
)
def test_action_like_but_non_official_oracle_labels_are_rejected(
    tmp_path, action
):
    root = tmp_path / "R2R"
    frame = "3/frame_0.jpg"
    _write_image(root / "train" / frame, 1)
    _write_annotations(root, [_record("3-0", [frame], action=action)])

    with pytest.raises(ValueError, match="unknown oracle action"):
        load_r2r_replay_episode(root, 3)


def test_habitat_pose_conversion_matches_matterport_usd_axes():
    assert habitat_position_to_isaac([15.0, 0.2, -4.5]) == (
        15.0,
        4.5,
        0.2,
    )
    rotation = habitat_heading_to_isaac(
        [0.0, 0.5, 0.0, math.sqrt(3.0) / 2.0]
    )
    assert rotation == pytest.approx(
        [math.cos(math.radians(75)), 0.0, 0.0, math.sin(math.radians(75))]
    )


def test_loads_original_vlnce_metadata_gt_and_builds_isaac_episode(tmp_path):
    metadata_path = tmp_path / "train" / "train.json.gz"
    gt_path = tmp_path / "train" / "train_gt.json.gz"
    _write_gzip_json(
        metadata_path,
        {
            "episodes": [
                {
                    "episode_id": 5372,
                    "trajectory_id": 3558,
                    "scene_id": "mp3d/scene/scene.glb",
                    "start_position": [-12.5, 0.1, -18.8],
                    "start_rotation": [0.0, math.sqrt(3) / 2, 0.0, 0.5],
                    "info": {"geodesic_distance": 9.7},
                    "goals": [{"position": [-13.6, 0.1, -9.1], "radius": 3.0}],
                    "instruction": {
                        "instruction_text": "go",
                        "instruction_tokens": [1, 2],
                    },
                    "reference_path": [
                        [-12.5, 0.1, -18.8],
                        [-13.6, 0.1, -9.1],
                    ],
                }
            ]
        },
    )
    _write_gzip_json(
        gt_path,
        {
            "5372": {
                "actions": [2, 1, 0],
                "locations": [
                    [-12.5, 0.1, -18.8],
                    [-13.6, 0.1, -9.1],
                ],
                "forward_steps": 1,
            }
        },
    )

    metadata = load_vlnce_episode_metadata(metadata_path, 5372)
    episode = metadata.to_isaac_episode()

    assert metadata.episode_id == "5372"
    assert metadata.gt_path == gt_path.resolve()
    assert episode["episode_id"] == 5372
    assert episode["scene_id"] == "mp3d/scene/scene.glb"
    assert episode["start_position"] == pytest.approx([-12.5, 18.8, 0.1])
    assert episode["goals"][0]["position"] == pytest.approx([-13.6, 9.1, 0.1])
    assert episode["gt_actions"] == [2, 1, 0]
    assert episode["gt_locations"][-1] == pytest.approx([-13.6, 9.1, 0.1])
    assert metadata.alignment_record()["vlnce_episode_id"] == "5372"


def test_vlnce_metadata_requires_matching_episode_and_gt(tmp_path):
    metadata_path = tmp_path / "train.json.gz"
    _write_gzip_json(metadata_path, {"episodes": []})
    _write_gzip_json(tmp_path / "train_gt.json.gz", {})

    with pytest.raises(ValueError, match="found 0 times"):
        load_vlnce_episode_metadata(metadata_path, 99)
