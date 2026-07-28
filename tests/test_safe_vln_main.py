from types import SimpleNamespace

import pytest

from scripts import safe_vln_main
from scripts.select_vlnce_episode_ids import balanced_episode_ids


def _args(**updates):
    values = {
        "safe_live_render": False,
        "safe_replay": True,
        "safe_replay_id": None,
        "safe_replay_ids": [10, 20],
        "start_idx": 3,
        "end_idx": None,
        "vlnce_episode_id": None,
        "vlnce_episode_ids": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_safe_replay_episode_plan_pairs_replay_and_physical_episodes():
    replay_ids, end = safe_vln_main._resolve_episode_plan(_args(), 10)

    assert replay_ids == [10, 20]
    assert end == 5


def test_safe_replay_episode_plan_rejects_conflicts_and_bad_range():
    with pytest.raises(ValueError, match="only one"):
        safe_vln_main._resolve_episode_plan(
            _args(safe_replay_id=10), 10
        )
    with pytest.raises(ValueError, match="out of bounds"):
        safe_vln_main._resolve_episode_plan(
            _args(start_idx=9), 10
        )


def test_python_launcher_uses_exported_glibc(monkeypatch):
    monkeypatch.setenv("GLIBC_LOADER", "/glibc/loader")
    monkeypatch.setenv("GLIBC_LIB", "/glibc/lib")
    monkeypatch.setenv("CONDA_PREFIX", "/conda")

    command = safe_vln_main._python_launcher()

    assert command[:3] == [
        "/glibc/loader",
        "--library-path",
        "/glibc/lib:/conda/lib:/lib64:/usr/lib64",
    ]


def test_safe_live_episode_plan_uses_explicit_vlnce_ids():
    ids, end = safe_vln_main._resolve_episode_plan(
        _args(
            safe_live_render=True,
            safe_replay=False,
            safe_replay_ids=None,
            vlnce_episode_ids=[1, 8, 20],
            start_idx=0,
        ),
        10,
    )

    assert ids == [1, 8, 20]
    assert end == 3


def test_balanced_episode_ids_cover_scenes_before_repeating():
    episodes = [
        {"episode_id": 1, "scene_id": "data/a/a.glb"},
        {"episode_id": 2, "scene_id": "data/a/a.glb"},
        {"episode_id": 3, "scene_id": "data/b/b.glb"},
        {"episode_id": 4, "scene_id": "data/b/b.glb"},
        {"episode_id": 5, "scene_id": "data/c/c.glb"},
        {"episode_id": 6, "scene_id": "data/c/c.glb"},
    ]
    selected = balanced_episode_ids(episodes, seed=7)
    first_scenes = {
        next(
            item["scene_id"]
            for item in episodes
            if str(item["episode_id"]) == episode_id
        )
        for episode_id in selected[:3]
    }
    assert len(first_scenes) == 3
    assert balanced_episode_ids(episodes, seed=7) == selected
