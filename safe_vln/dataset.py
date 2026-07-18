"""Atomic tar-shard storage for Safe-VLN visual transitions."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image


class SafeVLNShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        split: str = "train",
        samples_per_shard: int = 256,
        jpeg_quality: int = 90,
    ) -> None:
        if samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.samples_per_shard = samples_per_shard
        self.jpeg_quality = jpeg_quality
        self.shard_index = self._next_shard_index()
        self.sample_count = 0
        self.total_samples = 0
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
        if self._tar is None:
            self._open()
        safe_key = key.replace("..", "_").strip("/")
        if not safe_key:
            raise ValueError("sample key cannot be empty")
        for index, frame in enumerate(frames):
            buffer = BytesIO()
            frame.convert("RGB").save(buffer, format="JPEG", quality=self.jpeg_quality)
            self._add_bytes(f"{safe_key}.{index}.jpg", buffer.getvalue())
        payload = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True).encode("utf-8")
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
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "safe-vln-go2-v1",
                    "split": self.split,
                    "completed_shards": len(list(self.output_dir.glob(f"{self.split}-*.tar"))),
                    "samples_written_this_run": self.total_samples,
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


def iter_metadata(dataset_dir: str | Path, split: str = "train") -> Iterator[dict[str, Any]]:
    for shard_path in sorted(Path(dataset_dir).glob(f"{split}-*.tar")):
        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".json"):
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        yield json.loads(extracted.read().decode("utf-8"))


def iter_samples(dataset_dir: str | Path, split: str = "train"):
    """Yield ``(frames, metadata)`` from complete shards."""
    for shard_path in sorted(Path(dataset_dir).glob(f"{split}-*.tar")):
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
