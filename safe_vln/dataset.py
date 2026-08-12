"""Atomic tar-shard storage for Safe-VLN visual transitions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any, Iterator, Mapping, Sequence
import uuid

from PIL import Image

from .live_render import LIVE_SCHEMA_VERSION
from .objective import SCHEMA_VERSION, validate_objective_config


@dataclass(frozen=True)
class SafeVLNSampleRef:
    """Metadata-only reference to one sample in a complete tar shard."""

    shard_path: Path
    metadata_name: str
    metadata: dict[str, Any]

    @property
    def key(self) -> str:
        return self.metadata_name[:-5]

    @property
    def identity(self) -> tuple[str, str]:
        return str(self.shard_path), self.metadata_name


def _encode_jpeg(frame: Image.Image, quality: int) -> bytes:
    """Encode without relying on Pillow plugins after Isaac Kit startup."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        buffer = BytesIO()
        frame.convert("RGB").save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim == 2:
        encoded_input = array
    elif array.ndim == 3 and array.shape[2] == 3:
        encoded_input = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        encoded_input = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(
            f"unsupported frame array shape for JPEG encoding: {array.shape}"
        )
    success, encoded = cv2.imencode(
        ".jpg",
        encoded_input,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not success:
        raise RuntimeError("OpenCV failed to encode a Safe-VLN JPEG frame")
    payload = encoded.tobytes()
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise RuntimeError("OpenCV returned data without valid JPEG markers")
    return payload


class SafeVLNShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        split: str = "train",
        samples_per_shard: int = 256,
        jpeg_quality: int = 90,
        schema_version: str | None = None,
        objective_config: Mapping[str, Any] | None = None,
    ) -> None:
        if samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.samples_per_shard = samples_per_shard
        self.jpeg_quality = jpeg_quality
        self.objective_config = (
            validate_objective_config(objective_config)
            if objective_config
            else {}
        )
        self.schema_version = str(
            schema_version
            or (SCHEMA_VERSION if self.objective_config else "safe-vln-go2-v1")
        )
        self.objective_fingerprint = self.objective_config.get("fingerprint")
        if (
            self.schema_version == LIVE_SCHEMA_VERSION
            and self.objective_config.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError(
                f"{LIVE_SCHEMA_VERSION} datasets require the current "
                f"{SCHEMA_VERSION} objective contract"
            )
        if self.schema_version == SCHEMA_VERSION and not self.objective_fingerprint:
            raise ValueError(
                "versioned Safe-VLN shards require an objective fingerprint"
            )
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != self.schema_version:
                raise ValueError(
                    "cannot append Safe-VLN data with a different schema version"
                )
            if manifest.get("objective_fingerprint") != self.objective_fingerprint:
                raise ValueError(
                    "cannot append Safe-VLN data with a different objective fingerprint"
                )
        self.shard_index = self._next_shard_index()
        self.sample_count = 0
        self.total_samples = 0
        self._keys = self._existing_sample_keys()
        self._tar: tarfile.TarFile | None = None
        self._temporary_path: Path | None = None

    def _next_shard_index(self) -> int:
        indices = []
        for path in self.output_dir.glob(f"{self.split}-*.tar"):
            try:
                indices.append(int(path.stem.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(indices, default=-1) + 1

    def _existing_sample_keys(self) -> set[str]:
        keys: set[str] = set()
        for shard_path in sorted(self.output_dir.glob(f"{self.split}-*.tar")):
            with tarfile.open(shard_path, "r") as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith(".json"):
                        key = member.name[:-5]
                        if key in keys:
                            raise RuntimeError(
                                f"existing Safe-VLN dataset has duplicate key: {key}"
                            )
                        keys.add(key)
        return keys

    def _open(self) -> None:
        name = f"{self.split}-{self.shard_index:06d}.tar"
        self._temporary_path = self.output_dir / f"{name}.incomplete"
        self._tar = tarfile.open(self._temporary_path, "w")

    def _add_bytes(self, name: str, payload: bytes) -> None:
        assert self._tar is not None
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        self._tar.addfile(info, BytesIO(payload))

    def add(self, key: str, frames: Sequence[Image.Image], metadata: Mapping[str, Any]) -> None:
        if len(frames) != 8:
            raise ValueError(f"Safe-VLN samples require exactly 8 frames, got {len(frames)}")
        safe_key = key.replace("..", "_").strip("/")
        if not safe_key:
            raise ValueError("sample key cannot be empty")
        if safe_key in self._keys:
            raise ValueError(f"duplicate sample key: {safe_key}")
        self._keys.add(safe_key)
        if self._tar is None:
            self._open()
        for index, frame in enumerate(frames):
            self._add_bytes(
                f"{safe_key}.{index}.jpg",
                _encode_jpeg(frame, self.jpeg_quality),
            )
        payload = json.dumps(
            dict(metadata),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        self._add_bytes(f"{safe_key}.json", payload)
        self.sample_count += 1
        self.total_samples += 1
        if self.sample_count >= self.samples_per_shard:
            self.close_shard()

    def close_shard(self) -> None:
        if self._tar is None:
            return
        self._tar.close()
        assert self._temporary_path is not None
        final_path = Path(str(self._temporary_path).removesuffix(".incomplete"))
        self._temporary_path.replace(final_path)
        self._tar = None
        self._temporary_path = None
        self.shard_index += 1
        self.sample_count = 0

    def close(self) -> None:
        self.close_shard()
        completed_shards = sorted(
            self.output_dir.glob(f"{self.split}-*.tar")
        )
        total_samples = 0
        for shard_path in completed_shards:
            with tarfile.open(shard_path, "r") as archive:
                total_samples += sum(
                    member.isfile() and member.name.endswith(".json")
                    for member in archive
                )
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "objective_fingerprint": self.objective_fingerprint,
                    "objective_config": self.objective_config or None,
                    "split": self.split,
                    "completed_shards": len(completed_shards),
                    "samples_written_this_run": self.total_samples,
                    "total_samples": total_samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        elif self._tar is not None:
            self._tar.close()
        return False


class SafeVLNEpisodeWriter:
    """Transactional episode writer used by strict live-render collection.

    Shards are first written beneath ``.incomplete`` and only become visible to
    readers after :meth:`commit` atomically publishes the complete episode.
    """

    def __init__(
        self,
        output_dir: str | Path,
        episode_id: str | int,
        *,
        dataset_role: str,
        split: str,
        schema_version: str,
        objective_config: Mapping[str, Any],
        samples_per_shard: int = 256,
    ) -> None:
        if dataset_role not in {"train", "eval"}:
            raise ValueError("dataset_role must be 'train' or 'eval'")
        self.output_dir = Path(output_dir)
        self.episode_id = str(episode_id)
        self.dataset_role = dataset_role
        self.split = split
        self.schema_version = schema_version
        self.objective_config = validate_objective_config(objective_config)
        self.objective_fingerprint = self.objective_config.get("fingerprint")
        root_manifest_path = self.output_dir / "manifest.json"
        if root_manifest_path.exists():
            root_manifest = json.loads(
                root_manifest_path.read_text(encoding="utf-8")
            )
            expected = (
                schema_version,
                self.objective_fingerprint,
                dataset_role,
                split,
            )
            actual = (
                root_manifest.get("schema_version"),
                root_manifest.get("objective_fingerprint"),
                root_manifest.get("dataset_role"),
                root_manifest.get("split"),
            )
            if actual != expected:
                raise ValueError(
                    "strict episode contract does not match the dataset manifest"
                )
        safe_episode = re.sub(r"[^A-Za-z0-9._-]+", "_", self.episode_id).strip("_")
        if not safe_episode:
            raise ValueError("episode_id has no filesystem-safe characters")
        self.safe_episode = safe_episode
        self.pending_dir = (
            self.output_dir
            / ".incomplete"
            / f"{safe_episode}-{uuid.uuid4().hex}"
        )
        self.final_dir = self.output_dir / "completed" / safe_episode
        if self.final_dir.exists():
            raise FileExistsError(
                f"strict episode output already exists: {self.final_dir}"
            )
        self.writer = SafeVLNShardWriter(
            self.pending_dir,
            split=split,
            samples_per_shard=samples_per_shard,
            schema_version=schema_version,
            objective_config=self.objective_config,
        )
        self._finished = False

    def add(
        self,
        key: str,
        frames: Sequence[Image.Image],
        metadata: Mapping[str, Any],
    ) -> None:
        if self._finished:
            raise RuntimeError("episode writer is already finished")
        self.writer.add(key, frames, metadata)

    def _write_episode_manifest(self) -> None:
        path = self.pending_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "dataset_role": self.dataset_role,
                "episode_id": self.episode_id,
                "transactional_episode": True,
            }
        )
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _update_root_manifest(self) -> None:
        manifests = []
        for path in sorted((self.output_dir / "completed").glob("*/manifest.json")):
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        if not manifests:
            return
        contracts = {
            (
                item.get("schema_version"),
                item.get("objective_fingerprint"),
                item.get("dataset_role"),
                item.get("split"),
            )
            for item in manifests
        }
        if len(contracts) != 1:
            raise RuntimeError(
                "completed strict episodes contain incompatible dataset contracts"
            )
        schema, fingerprint, role, split = next(iter(contracts))
        root_manifest = {
            "schema_version": schema,
            "objective_fingerprint": fingerprint,
            "objective_config": manifests[0].get("objective_config"),
            "dataset_role": role,
            "split": split,
            "transactional_episodes": True,
            "completed_episodes": len(manifests),
            "total_samples": sum(
                int(item.get("total_samples", 0)) for item in manifests
            ),
            "episode_ids": sorted(str(item["episode_id"]) for item in manifests),
        }
        temporary = self.output_dir / "manifest.json.incomplete"
        temporary.write_text(
            json.dumps(root_manifest, indent=2), encoding="utf-8"
        )
        temporary.replace(self.output_dir / "manifest.json")

    def commit(self) -> Path:
        if self._finished:
            raise RuntimeError("episode writer is already finished")
        if self.writer.total_samples <= 0:
            self.abort()
            raise RuntimeError("cannot commit an empty Safe-VLN episode")
        if self.final_dir.exists():
            self.abort()
            raise FileExistsError(
                f"strict episode output already exists: {self.final_dir}"
            )
        self.writer.close()
        self._write_episode_manifest()
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        self.pending_dir.replace(self.final_dir)
        try:
            self._update_root_manifest()
        except Exception:
            # Roll back visibility if publishing the root manifest fails.
            self.pending_dir.parent.mkdir(parents=True, exist_ok=True)
            self.final_dir.replace(self.pending_dir)
            self.abort()
            raise
        self._finished = True
        return self.final_dir

    def abort(self) -> None:
        if self._finished:
            return
        if self.writer._tar is not None:
            self.writer._tar.close()
            self.writer._tar = None
        shutil.rmtree(self.pending_dir, ignore_errors=True)
        self._finished = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        return False


def _shard_paths(dataset_dir: str | Path, split: str) -> list[Path]:
    root = Path(dataset_dir)
    paths = set(root.glob(f"{split}-*.tar"))
    paths.update(root.glob(f"completed/*/{split}-*.tar"))
    return sorted(paths)


def iter_metadata(dataset_dir: str | Path, split: str = "train") -> Iterator[dict[str, Any]]:
    for shard_path in _shard_paths(dataset_dir, split):
        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".json"):
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        yield json.loads(extracted.read().decode("utf-8"))


def iter_sample_refs(
    dataset_dir: str | Path,
    split: str = "train",
) -> Iterator[SafeVLNSampleRef]:
    """Yield sample references without decoding their eight RGB frames."""

    for shard_path in _shard_paths(dataset_dir, split):
        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                yield SafeVLNSampleRef(
                    shard_path=shard_path,
                    metadata_name=member.name,
                    metadata=json.loads(extracted.read().decode("utf-8")),
                )


def load_sample_refs(
    refs: Sequence[SafeVLNSampleRef],
) -> list[tuple[list[Image.Image], dict[str, Any]]]:
    """Decode selected references while opening each tar shard at most once."""

    grouped: dict[Path, list[tuple[int, SafeVLNSampleRef]]] = defaultdict(list)
    for output_index, ref in enumerate(refs):
        grouped[ref.shard_path].append((output_index, ref))

    loaded: list[tuple[list[Image.Image], dict[str, Any]] | None] = [
        None
    ] * len(refs)
    for shard_path, shard_refs in grouped.items():
        with tarfile.open(shard_path, "r") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile()
            }
            for output_index, ref in shard_refs:
                frames: list[Image.Image] = []
                for frame_index in range(8):
                    frame_name = f"{ref.key}.{frame_index}.jpg"
                    member = members.get(frame_name)
                    if member is None:
                        raise ValueError(
                            f"missing {frame_name} in {shard_path}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(
                            f"failed to extract {frame_name} from {shard_path}"
                        )
                    with Image.open(BytesIO(extracted.read())) as image:
                        frames.append(image.convert("RGB").copy())
                loaded[output_index] = frames, ref.metadata

    if any(sample is None for sample in loaded):
        raise RuntimeError("failed to decode one or more Safe-VLN samples")
    return [sample for sample in loaded if sample is not None]


def iter_samples(dataset_dir: str | Path, split: str = "train"):
    """Yield ``(frames, metadata)`` from complete shards."""
    for shard_path in _shard_paths(dataset_dir, split):
        with tarfile.open(shard_path, "r") as archive:
            members = {member.name: member for member in archive if member.isfile()}
            for metadata_name in sorted(name for name in members if name.endswith(".json")):
                prefix = metadata_name[:-5]
                metadata_file = archive.extractfile(members[metadata_name])
                if metadata_file is None:
                    continue
                metadata = json.loads(metadata_file.read().decode("utf-8"))
                frames = []
                for index in range(8):
                    image_file = archive.extractfile(members[f"{prefix}.{index}.jpg"])
                    if image_file is None:
                        raise ValueError(f"missing frame {index} for {prefix} in {shard_path}")
                    frames.append(Image.open(BytesIO(image_file.read())).convert("RGB"))
                yield frames, metadata


def write_episode_summary(output_dir: str | Path, episode: Mapping[str, Any]) -> Path:
    directory = Path(output_dir) / "episodes"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_{episode['episode_id']}.json"
    temporary = path.with_suffix(".json.incomplete")
    temporary.write_text(json.dumps(dict(episode), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path
