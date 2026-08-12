"""Frame-history sampling for Isaac's native RGB camera."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .live_render import sample_navila_history


def sample_native_history(
    entries: Sequence[Mapping[str, Any]],
    *,
    num_frames: int = 8,
) -> list[dict[str, Any]]:
    """Select a NaViLA history and preserve frame/state alignment metadata.

    Missing history is padded with copies of the first real frame.  Padding is
    explicit in the returned metadata so it cannot be mistaken for a second
    observation of the robot at a different physics step.
    """

    if not entries:
        raise ValueError("native camera history must contain at least one frame")
    if num_frames < 2:
        raise ValueError("num_frames must be at least two")

    source = []
    for entry in entries:
        metadata = deepcopy(dict(entry.get("metadata", {})))
        metadata.setdefault("history_padding", False)
        metadata.setdefault("strict_observation_state_alignment", True)
        source.append({"image": entry["image"], "metadata": metadata})
    def repeat_first_padding():
        first = source[0]
        metadata = deepcopy(first["metadata"])
        metadata.update(
            {
                "history_padding": True,
                "padding_policy": "repeat_first",
                "padding_source_frame_index": metadata.get("frame_index"),
                "strict_observation_state_alignment": False,
                "physics_step": None,
            }
        )
        return {"image": first["image"].copy(), "metadata": metadata}

    return sample_navila_history(
        source,
        num_frames=num_frames,
        padding_factory=repeat_first_padding,
    )
