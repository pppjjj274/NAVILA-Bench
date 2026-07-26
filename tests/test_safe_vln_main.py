from types import SimpleNamespace

import pytest

from scripts import safe_vln_main


def _args(**updates):
    values = {
        "safe_replay": True,
        "safe_replay_id": None,
        "safe_replay_ids": [10, 20],
        "start_idx": 3,
        "end_idx": None,
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
