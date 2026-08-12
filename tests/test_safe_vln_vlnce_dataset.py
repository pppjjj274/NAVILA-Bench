import pytest

from safe_vln.vlnce_dataset import (
    ISAAC_COORDINATE_SYSTEM,
    convert_vlnce_payload,
    validate_isaac_vlnce_payload,
)


def _episode(episode_id, scene, x):
    return {
        "episode_id": episode_id,
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "start_position": [x, 1.0, 2.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [x + 1.0, 1.0, 2.0], "radius": 3.0}],
        "reference_path": [[x, 1.0, 2.0], [x + 1.0, 1.0, 2.0]],
        "instruction": {"instruction_text": "go", "instruction_tokens": []},
    }


def test_convert_vlnce_payload_is_scene_balanced_and_in_isaac_coordinates():
    episodes = [
        _episode("a1", "a", 0.0),
        _episode("a2", "a", 2.0),
        _episode("b1", "b", 4.0),
        _episode("b2", "b", 6.0),
    ]
    ground_truth = {
        item["episode_id"]: {
            "locations": item["reference_path"],
            "actions": [1, 0],
            "forward_steps": 1,
        }
        for item in episodes
    }

    payload = convert_vlnce_payload(
        {"episodes": episodes},
        ground_truth,
        source_split="train",
        balanced_seed=7,
    )

    converted = payload["episodes"]
    assert len({item["scene_id"].split("/")[-2] for item in converted[:2]}) == 2
    assert payload["safe_vln_conversion"]["coordinate_system"] == ISAAC_COORDINATE_SYSTEM
    original = next(item for item in converted if item["episode_id"] == "a1")
    assert original["start_position"] == pytest.approx([0.0, -2.0, 1.0])
    assert original["start_rotation"] == pytest.approx(
        [2**-0.5, 0.0, 0.0, 2**-0.5]
    )
    assert original["gt_locations"][-1] == pytest.approx([1.0, -2.0, 1.0])


def test_native_dataset_validation_rejects_split_leakage_and_no_provenance():
    episode = _episode("one", "a", 0.0)
    payload = convert_vlnce_payload(
        {"episodes": [episode]},
        {"one": {"locations": episode["reference_path"], "actions": [0]}},
        source_split="val_unseen",
        balanced_seed=1,
    )

    with pytest.raises(ValueError, match="does not match requested role"):
        validate_isaac_vlnce_payload(payload, expected_role="train")
    with pytest.raises(ValueError, match="lacks conversion provenance"):
        validate_isaac_vlnce_payload(
            {"episodes": payload["episodes"]}, expected_role="eval"
        )


def test_native_dataset_validation_checks_expected_scene_coverage():
    episode = _episode("one", "a", 0.0)
    payload = convert_vlnce_payload(
        {"episodes": [episode]},
        {"one": {"locations": episode["reference_path"], "actions": [0]}},
        source_split="train",
        balanced_seed=1,
    )
    with pytest.raises(ValueError, match="expected 61"):
        validate_isaac_vlnce_payload(
            payload, expected_role="train", expected_scene_count=61
        )


def test_native_dataset_load_contract_requires_source_hashes():
    episode = _episode("1", "a", 0.0)
    payload = convert_vlnce_payload(
        {"episodes": [episode]},
        {1: {"locations": episode["reference_path"], "actions": [0]}},
        source_split="train",
        balanced_seed=1,
    )
    with pytest.raises(ValueError, match="source_metadata_sha256"):
        validate_isaac_vlnce_payload(
            payload,
            expected_role="train",
            require_source_hashes=True,
        )
    payload["safe_vln_conversion"].update(
        {
            "source_metadata_sha256": "a" * 64,
            "source_gt_sha256": "b" * 64,
        }
    )
    validate_isaac_vlnce_payload(
        payload,
        expected_role="train",
        require_source_hashes=True,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(episode_id=None), "non-empty ID"),
        (lambda item: item.update(scene_id=None), "scene_id"),
        (lambda item: item.update(start_rotation=[0.0] * 4), "non-unit"),
        (lambda item: item["goals"][0].update(radius=0.0), "goal radius"),
        (lambda item: item.update(instruction={}), "instruction"),
        (lambda item: item.update(reference_path=[]), "reference_path"),
        (lambda item: item.update(gt_locations=[[float("nan"), 0.0, 0.0]]), "gt_locations"),
        (lambda item: item.update(gt_actions=[]), "GT actions"),
        (lambda item: item.update(gt_actions=[4]), "GT actions"),
        (lambda item: item.update(gt_forward_steps=-1), "forward_steps"),
    ],
)
def test_native_dataset_validation_rejects_corrupt_episode_fields(
    mutation, message
):
    episode = _episode("1", "a", 0.0)
    payload = convert_vlnce_payload(
        {"episodes": [episode]},
        {"1": {"locations": episode["reference_path"], "actions": [0]}},
        source_split="train",
        balanced_seed=1,
    )
    mutation(payload["episodes"][0])

    with pytest.raises(ValueError, match=message):
        validate_isaac_vlnce_payload(payload, expected_role="train")


@pytest.mark.parametrize("invalid_action", [True, 1.5, "1", 4, -1])
def test_convert_rejects_noncanonical_gt_actions(invalid_action):
    episode = _episode("1", "a", 0.0)
    with pytest.raises(ValueError, match="invalid GT actions"):
        convert_vlnce_payload(
            {"episodes": [episode]},
            {
                "1": {
                    "locations": episode["reference_path"],
                    "actions": [invalid_action],
                    "forward_steps": 0,
                }
            },
            source_split="train",
            balanced_seed=1,
        )


@pytest.mark.parametrize("invalid_forward_steps", [True, -1, 1.5, "1"])
def test_convert_rejects_invalid_forward_steps(invalid_forward_steps):
    episode = _episode("1", "a", 0.0)
    with pytest.raises(ValueError, match="invalid forward_steps"):
        convert_vlnce_payload(
            {"episodes": [episode]},
            {
                "1": {
                    "locations": episode["reference_path"],
                    "actions": [0],
                    "forward_steps": invalid_forward_steps,
                }
            },
            source_split="train",
            balanced_seed=1,
        )
