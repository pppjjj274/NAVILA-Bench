"""Offline R2R observations for the Safe-VLN replay mode.

The NaViLA R2R annotations contain one record per high-level action.  Each
record carries the cumulative list of frames available at that decision
point.  This module deliberately has no Isaac or NumPy dependency so replay
data can be checked before the simulator is started.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from .actions import NAVILA_ACTION_RESPONSES, SafeAction, action_from_id


_NUM_REPLAY_FRAMES = 8
_OFFICIAL_ORACLE_ACTION_IDS = {
    text.lower(): action_id
    for action_id, text in enumerate(NAVILA_ACTION_RESPONSES)
}


def _parse_official_oracle_action(raw_action: str, *, video_id: str) -> SafeAction:
    normalized = " ".join(raw_action.lower().split())
    action_id = _OFFICIAL_ORACLE_ACTION_IDS.get(normalized)
    if action_id is None:
        raise ValueError(
            f"R2R annotation {video_id!r} has an unknown oracle action: "
            f"{raw_action!r}"
        )
    return action_from_id(action_id)


def _sample_frame_paths(frame_paths: Sequence[Path]) -> tuple[Path | None, ...]:
    """Apply ``navila_eval.sample_eight_images`` to paths.

    ``None`` represents a black padding frame.  Keeping the sampling at the
    path level ensures that long cumulative histories do not cause every
    source image to be decoded.
    """
    if not frame_paths:
        raise ValueError("R2R replay step did not provide any frames")

    if len(frame_paths) < _NUM_REPLAY_FRAMES:
        padding = (None,) * (_NUM_REPLAY_FRAMES - len(frame_paths))
        candidates: tuple[Path | None, ...] = padding + tuple(frame_paths)
    else:
        candidates = tuple(frame_paths)

    num_images = len(candidates)
    indices = [int(index * (num_images - 1) / 7) for index in range(7)]
    sampled = tuple(candidates[index] for index in indices) + (candidates[-1],)
    assert len(sampled) == _NUM_REPLAY_FRAMES
    return sampled


def _open_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            result = image.convert("RGB")
            result.load()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
        raise ValueError(f"R2R replay image is missing or unreadable: {path}") from error
    return result


@dataclass(frozen=True)
class R2RReplayStep:
    """One high-level R2R decision and its cumulative visual history."""

    episode_id: str
    video_id: str
    step_index: int
    instruction: str
    raw_oracle_action: str
    oracle_action: SafeAction
    frame_paths: tuple[Path, ...]
    sampled_frame_paths: tuple[Path | None, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampled_frame_paths", _sample_frame_paths(self.frame_paths))

    def load_frames(self) -> list[Image.Image]:
        """Decode exactly eight sampled RGB frames.

        Padding is inserted before the real observations and uses the current
        observation's dimensions.  The last returned frame is therefore
        always the current observation.
        """
        real_frames: dict[Path, Image.Image] = {}
        try:
            for path in dict.fromkeys(
                path for path in self.sampled_frame_paths if path is not None
            ):
                real_frames[path] = _open_rgb(path)
            current_path = self.sampled_frame_paths[-1]
            assert current_path is not None
            current = real_frames[current_path]
            frames = [
                Image.new("RGB", current.size, (0, 0, 0))
                if path is None
                else real_frames[path].copy()
                for path in self.sampled_frame_paths
            ]
        finally:
            for frame in real_frames.values():
                frame.close()
        assert len(frames) == _NUM_REPLAY_FRAMES
        return frames

    def validate_images(self) -> None:
        """Decode all real images selected for this step."""
        frames = self.load_frames()
        for frame in frames:
            frame.close()


@dataclass(frozen=True)
class R2RReplayEpisode:
    """An ordered sequence of R2R high-level replay decisions."""

    episode_id: str
    instruction: str
    steps: tuple[R2RReplayStep, ...]

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[R2RReplayStep]:
        return iter(self.steps)

    def __getitem__(self, index: int) -> R2RReplayStep:
        return self.steps[index]

    def validate_images(self) -> None:
        """Fail early if any image selected by this episode cannot be decoded."""
        validated: set[Path] = set()
        for step in self.steps:
            for path in step.sampled_frame_paths:
                if path is None or path in validated:
                    continue
                image = _open_rgb(path)
                image.close()
                validated.add(path)


def _read_annotations(path: Path) -> list[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as annotations_file:
            payload = json.load(annotations_file)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read R2R annotations: {path}") from error
    if not isinstance(payload, list):
        raise ValueError(f"R2R annotations must be a JSON list: {path}")
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise ValueError(f"R2R annotation at index {index} must be an object")
    return payload


def _resolve_frame_path(train_root: Path, value: Any, *, video_id: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"R2R annotation {video_id!r} contains an invalid frame path: {value!r}")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"R2R annotation {video_id!r} contains an absolute frame path: {value!r}")
    root = train_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"R2R annotation {video_id!r} frame escapes train root: {value!r}") from error
    return candidate


def load_r2r_replay_episode(
    root: str | Path,
    episode_id: str | int,
    *,
    annotations_path: str | Path | None = None,
    validate_images: bool = True,
) -> R2RReplayEpisode:
    """Load one episode from an extracted NaViLA R2R dataset.

    Args:
        root: Dataset directory containing ``annotations.json`` and ``train/``.
        episode_id: Exact episode prefix, for example ``5372``.  Only
            ``5372-<integer>`` video IDs match.
        annotations_path: Optional annotation file override.
        validate_images: Decode every uniquely sampled image before returning.
    """
    dataset_root = Path(root).expanduser()
    annotation_file = (
        Path(annotations_path).expanduser()
        if annotations_path is not None
        else dataset_root / "annotations.json"
    )
    train_root = dataset_root / "train"
    requested_id = str(episode_id)
    if not requested_id or "-" in requested_id:
        raise ValueError(f"episode_id must be a non-empty R2R episode prefix, got {requested_id!r}")
    prefix = f"{requested_id}-"

    unique: dict[str, Mapping[str, Any]] = {}
    suffixes: dict[str, int] = {}
    for record in _read_annotations(annotation_file):
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not video_id.startswith(prefix):
            continue
        suffix = video_id[len(prefix) :]
        if not suffix.isdecimal():
            continue
        previous = unique.get(video_id)
        if previous is not None:
            if dict(previous) != dict(record):
                raise ValueError(f"Conflicting duplicate R2R annotation for video_id {video_id!r}")
            continue
        unique[video_id] = record
        suffixes[video_id] = int(suffix)

    if not unique:
        raise ValueError(f"R2R replay episode {requested_id!r} was not found in {annotation_file}")

    ordered_records = sorted(unique.values(), key=lambda item: suffixes[str(item["video_id"])])
    steps: list[R2RReplayStep] = []
    instruction: str | None = None
    for record in ordered_records:
        video_id = str(record["video_id"])
        query = record.get("q")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"R2R annotation {video_id!r} has no instruction")
        if instruction is None:
            instruction = query
        elif query != instruction:
            raise ValueError(f"R2R episode {requested_id!r} contains conflicting instructions")

        raw_action = record.get("a")
        if not isinstance(raw_action, str):
            raise ValueError(f"R2R annotation {video_id!r} has no oracle action")
        oracle_action = _parse_official_oracle_action(
            raw_action, video_id=video_id
        )

        raw_frames = record.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError(f"R2R annotation {video_id!r} must contain a non-empty frames list")
        frame_paths = tuple(
            _resolve_frame_path(train_root, value, video_id=video_id) for value in raw_frames
        )
        steps.append(
            R2RReplayStep(
                episode_id=requested_id,
                video_id=video_id,
                step_index=suffixes[video_id],
                instruction=query,
                raw_oracle_action=raw_action,
                oracle_action=oracle_action,
                frame_paths=frame_paths,
            )
        )

    assert instruction is not None
    episode = R2RReplayEpisode(requested_id, instruction, tuple(steps))
    if validate_images:
        episode.validate_images()
    return episode


__all__ = [
    "R2RReplayEpisode",
    "R2RReplayStep",
    "load_r2r_replay_episode",
]
