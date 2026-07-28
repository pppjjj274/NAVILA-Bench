"""Offline R2R observations for the Safe-VLN replay mode.

The NaViLA R2R annotations contain one record per high-level action.  Each
record carries the cumulative list of frames available at that decision
point.  This module deliberately has no Isaac or NumPy dependency so replay
data can be checked before the simulator is started.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from .actions import NAVILA_ACTION_RESPONSES, SafeAction, action_from_id


_NUM_REPLAY_FRAMES = 8
DEFAULT_VLNCE_TRAIN_METADATA = Path(
    "~/NaVILA/evaluation/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
)
_OFFICIAL_ORACLE_ACTION_IDS = {
    text.lower(): action_id
    for action_id, text in enumerate(NAVILA_ACTION_RESPONSES)
}


def habitat_position_to_isaac(
    position: Sequence[float],
) -> tuple[float, float, float]:
    """Convert Habitat's ``(x, y-up, z)`` coordinates to Isaac ``(x, y, z-up)``."""
    if len(position) != 3:
        raise ValueError(f"expected a 3D Habitat position, got {position!r}")
    x, y, z = (float(value) for value in position)
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError(f"Habitat position must be finite, got {position!r}")
    return (x, -z, y)


def habitat_heading_to_isaac(
    rotation_xyzw: Sequence[float],
) -> tuple[float, float, float, float]:
    """Convert an R2R Habitat heading quaternion to Isaac's WXYZ convention.

    R2R episode rotations are yaw-only quaternions in Habitat's XYZW,
    Y-up coordinate frame.  The Matterport USD conversion maps Habitat
    ``(x, y, z)`` to Isaac ``(x, -z, y)``, which adds 90 degrees to the
    planar heading.
    """
    if len(rotation_xyzw) != 4:
        raise ValueError(
            f"expected a 4D Habitat XYZW quaternion, got {rotation_xyzw!r}"
        )
    x, y, z, w = (float(value) for value in rotation_xyzw)
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise ValueError(
            f"Habitat rotation must be finite, got {rotation_xyzw!r}"
        )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError("Habitat rotation quaternion has zero norm")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    if abs(x) > 1e-5 or abs(z) > 1e-5:
        raise ValueError(
            "R2R start_rotation must be a yaw-only Habitat quaternion"
        )
    habitat_yaw = 2.0 * math.atan2(y, w)
    isaac_yaw = habitat_yaw + math.pi / 2.0
    return (
        math.cos(isaac_yaw / 2.0),
        0.0,
        0.0,
        math.sin(isaac_yaw / 2.0),
    )


def _raw_position_tuple(
    value: Any, *, field_name: str
) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a 3D position")
    try:
        position = tuple(float(component) for component in value)
        habitat_position_to_isaac(position)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field_name}: {value!r}") from error
    return position


@dataclass(frozen=True)
class VLNCEEpisodeMetadata:
    """Original VLN-CE metadata and oracle trajectory for one R2R episode."""

    episode_id: str
    trajectory_id: int | str | None
    scene_id: str
    instruction: str
    instruction_tokens: tuple[int, ...]
    start_position_habitat: tuple[float, float, float]
    start_rotation_habitat_xyzw: tuple[float, float, float, float]
    goal_position_habitat: tuple[float, float, float]
    goal_radius: float
    reference_path_habitat: tuple[tuple[float, float, float], ...]
    geodesic_distance: float | None
    gt_actions: tuple[int, ...]
    gt_locations_habitat: tuple[tuple[float, float, float], ...]
    gt_forward_steps: int
    metadata_path: Path
    gt_path: Path

    @property
    def scene_name(self) -> str:
        return Path(self.scene_id).stem

    @property
    def start_position_isaac(self) -> tuple[float, float, float]:
        return habitat_position_to_isaac(self.start_position_habitat)

    @property
    def start_rotation_isaac_wxyz(self) -> tuple[float, float, float, float]:
        return habitat_heading_to_isaac(self.start_rotation_habitat_xyzw)

    @property
    def goal_position_isaac(self) -> tuple[float, float, float]:
        return habitat_position_to_isaac(self.goal_position_habitat)

    @property
    def reference_path_isaac(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            habitat_position_to_isaac(position)
            for position in self.reference_path_habitat
        )

    @property
    def gt_locations_isaac(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            habitat_position_to_isaac(position)
            for position in self.gt_locations_habitat
        )

    def to_isaac_episode(self) -> dict[str, Any]:
        """Return the episode schema consumed by NaVILA-Bench's Isaac wrapper."""
        episode_id: int | str = (
            int(self.episode_id)
            if self.episode_id.isdecimal()
            else self.episode_id
        )
        info: dict[str, float] = {}
        if self.geodesic_distance is not None:
            info["geodesic_distance"] = self.geodesic_distance
        return {
            "episode_id": episode_id,
            "trajectory_id": self.trajectory_id,
            "scene_id": self.scene_id,
            "start_position": list(self.start_position_isaac),
            "start_rotation": list(self.start_rotation_isaac_wxyz),
            "info": info,
            "goals": [
                {
                    "position": list(self.goal_position_isaac),
                    "radius": self.goal_radius,
                }
            ],
            "instruction": {
                "instruction_text": self.instruction,
                "instruction_tokens": list(self.instruction_tokens),
            },
            "reference_path": [
                list(position) for position in self.reference_path_isaac
            ],
            "gt_actions": list(self.gt_actions),
            "gt_locations": [
                list(position) for position in self.gt_locations_isaac
            ],
            "gt_forward_steps": self.gt_forward_steps,
        }

    def alignment_record(self) -> dict[str, Any]:
        """Serializable provenance stored once per Safe-Replay episode."""
        return {
            "vlnce_episode_id": self.episode_id,
            "vlnce_trajectory_id": self.trajectory_id,
            "vlnce_scene_id": self.scene_id,
            "vlnce_metadata_path": str(self.metadata_path),
            "vlnce_gt_path": str(self.gt_path),
            "start_position_habitat": list(self.start_position_habitat),
            "start_rotation_habitat_xyzw": list(
                self.start_rotation_habitat_xyzw
            ),
            "start_position_isaac": list(self.start_position_isaac),
            "start_rotation_isaac_wxyz": list(
                self.start_rotation_isaac_wxyz
            ),
            "goal_position_habitat": list(self.goal_position_habitat),
            "goal_position_isaac": list(self.goal_position_isaac),
            "goal_radius": self.goal_radius,
            "reference_path_habitat": [
                list(position) for position in self.reference_path_habitat
            ],
            "reference_path_isaac": [
                list(position) for position in self.reference_path_isaac
            ],
            "gt_actions": list(self.gt_actions),
            "gt_locations_habitat": [
                list(position) for position in self.gt_locations_habitat
            ],
            "gt_locations_isaac": [
                list(position) for position in self.gt_locations_isaac
            ],
            "gt_forward_steps": self.gt_forward_steps,
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


def _read_gzip_json(path: Path, *, description: str) -> Any:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as input_file:
            return json.load(input_file)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read {description}: {path}") from error


def _default_gt_path(metadata_path: Path) -> Path:
    name = metadata_path.name
    suffix = ".json.gz"
    if not name.endswith(suffix):
        raise ValueError(
            "VLN-CE metadata filename must end in .json.gz when the GT path "
            "is not specified"
        )
    return metadata_path.with_name(f"{name[:-len(suffix)]}_gt{suffix}")


def load_vlnce_episode_metadata(
    metadata_path: str | Path,
    episode_id: str | int,
    *,
    gt_path: str | Path | None = None,
) -> VLNCEEpisodeMetadata:
    """Load and convert an original VLN-CE episode plus its oracle trajectory."""
    resolved_metadata_path = Path(metadata_path).expanduser().resolve()
    resolved_gt_path = (
        Path(gt_path).expanduser().resolve()
        if gt_path is not None
        else _default_gt_path(resolved_metadata_path)
    )
    requested_id = str(episode_id)

    payload = _read_gzip_json(
        resolved_metadata_path, description="VLN-CE episode metadata"
    )
    episodes = payload.get("episodes") if isinstance(payload, Mapping) else None
    if not isinstance(episodes, list):
        raise ValueError(
            f"VLN-CE metadata has no episodes list: {resolved_metadata_path}"
        )
    matches = [
        episode
        for episode in episodes
        if isinstance(episode, Mapping)
        and str(episode.get("episode_id")) == requested_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"VLN-CE episode {requested_id!r} was found {len(matches)} times "
            f"in {resolved_metadata_path}"
        )
    episode = matches[0]

    gt_payload = _read_gzip_json(
        resolved_gt_path, description="VLN-CE ground-truth trajectories"
    )
    if not isinstance(gt_payload, Mapping) or requested_id not in gt_payload:
        raise ValueError(
            f"VLN-CE GT episode {requested_id!r} was not found in "
            f"{resolved_gt_path}"
        )
    gt = gt_payload[requested_id]
    if not isinstance(gt, Mapping):
        raise ValueError(f"VLN-CE GT episode {requested_id!r} must be an object")

    instruction_payload = episode.get("instruction")
    if not isinstance(instruction_payload, Mapping):
        raise ValueError(f"VLN-CE episode {requested_id!r} has no instruction")
    instruction = instruction_payload.get("instruction_text")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            f"VLN-CE episode {requested_id!r} has invalid instruction text"
        )
    raw_tokens = instruction_payload.get("instruction_tokens", [])
    if not isinstance(raw_tokens, list):
        raise ValueError(
            f"VLN-CE episode {requested_id!r} has invalid instruction tokens"
        )

    goals = episode.get("goals")
    if not isinstance(goals, list) or len(goals) != 1:
        raise ValueError(
            f"VLN-CE episode {requested_id!r} must contain exactly one goal"
        )
    goal = goals[0]
    if not isinstance(goal, Mapping):
        raise ValueError(f"VLN-CE episode {requested_id!r} goal must be an object")

    raw_reference_path = episode.get("reference_path")
    raw_gt_locations = gt.get("locations")
    raw_gt_actions = gt.get("actions")
    if not isinstance(raw_reference_path, list) or not raw_reference_path:
        raise ValueError(
            f"VLN-CE episode {requested_id!r} has no reference path"
        )
    if not isinstance(raw_gt_locations, list) or not raw_gt_locations:
        raise ValueError(f"VLN-CE episode {requested_id!r} has no GT locations")
    if not isinstance(raw_gt_actions, list) or not raw_gt_actions:
        raise ValueError(f"VLN-CE episode {requested_id!r} has no GT actions")

    scene_id = episode.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError(f"VLN-CE episode {requested_id!r} has no scene_id")
    raw_rotation = episode.get("start_rotation")
    if not isinstance(raw_rotation, Sequence) or isinstance(
        raw_rotation, (str, bytes)
    ):
        raise ValueError(
            f"VLN-CE episode {requested_id!r} has invalid start_rotation"
        )
    rotation = tuple(float(value) for value in raw_rotation)
    habitat_heading_to_isaac(rotation)

    raw_info = episode.get("info", {})
    geodesic_distance = (
        float(raw_info["geodesic_distance"])
        if isinstance(raw_info, Mapping)
        and raw_info.get("geodesic_distance") is not None
        else None
    )
    return VLNCEEpisodeMetadata(
        episode_id=requested_id,
        trajectory_id=episode.get("trajectory_id"),
        scene_id=scene_id,
        instruction=instruction,
        instruction_tokens=tuple(int(token) for token in raw_tokens),
        start_position_habitat=_raw_position_tuple(
            episode.get("start_position"), field_name="start_position"
        ),
        start_rotation_habitat_xyzw=rotation,
        goal_position_habitat=_raw_position_tuple(
            goal.get("position"), field_name="goal position"
        ),
        goal_radius=float(goal.get("radius", 3.0)),
        reference_path_habitat=tuple(
            _raw_position_tuple(
                position, field_name=f"reference_path[{index}]"
            )
            for index, position in enumerate(raw_reference_path)
        ),
        geodesic_distance=geodesic_distance,
        gt_actions=tuple(int(action) for action in raw_gt_actions),
        gt_locations_habitat=tuple(
            _raw_position_tuple(
                position, field_name=f"GT locations[{index}]"
            )
            for index, position in enumerate(raw_gt_locations)
        ),
        gt_forward_steps=int(gt.get("forward_steps", 0)),
        metadata_path=resolved_metadata_path,
        gt_path=resolved_gt_path,
    )


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
    "DEFAULT_VLNCE_TRAIN_METADATA",
    "R2RReplayEpisode",
    "R2RReplayStep",
    "VLNCEEpisodeMetadata",
    "habitat_heading_to_isaac",
    "habitat_position_to_isaac",
    "load_r2r_replay_episode",
    "load_vlnce_episode_metadata",
]
