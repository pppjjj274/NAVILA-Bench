"""Strict Go2-to-Habitat live-render protocol and pose utilities.

This module intentionally has no Habitat-Sim or Isaac-Sim dependency.  Both
simulators import it to share one wire contract and one coordinate transform.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import socket
from typing import Any, Mapping, Sequence

from PIL import Image

from .actions import action_from_id


LIVE_SCHEMA_VERSION = "safe-vln-go2-v5"
LEGACY_LIVE_SCHEMA_VERSIONS = frozenset(
    {"safe-vln-go2-v3", "safe-vln-go2-v4"}
)
LIVE_RENDER_PROTOCOL = "safe-vln-habitat-render-v2"
DEFAULT_RENDER_PORT = 54322
DEFAULT_RENDER_TIMEOUT_SECONDS = 10.0
DEFAULT_GO2_BASE_HEIGHT_M = 0.4
MAX_RENDER_MESSAGE_BYTES = 32 * 1024 * 1024
NAVILA_VIDEO_FRAMES = 8
NAVILA_HISTORY_SAMPLING_POLICY = "navila_uniform_full_history_v1"


def sample_navila_history(
    history: Sequence[Any],
    *,
    num_frames: int = NAVILA_VIDEO_FRAMES,
    padding_factory=None,
) -> list[Any]:
    """Match NaViLA official full-history sampling exactly."""
    if num_frames < 2:
        raise ValueError("NaViLA history sampling requires at least two frames")
    if not history:
        raise ValueError("NaViLA history cannot be empty")
    items = list(history)
    missing = num_frames - len(items)
    if missing > 0:
        if padding_factory is None:
            raise ValueError("short NaViLA histories require a padding factory")
        items = [padding_factory() for _ in range(missing)] + items
    last_index = len(items) - 1
    uniform_count = num_frames - 1
    indices = [
        (index * last_index) // uniform_count
        for index in range(uniform_count)
    ]
    return [items[index] for index in indices] + [items[-1]]



def wrap_angle_radians(value: float) -> float:
    """Normalize an angle to ``[-pi, pi)``."""
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def isaac_position_to_habitat(
    position: Sequence[float],
    *,
    base_height_m: float = DEFAULT_GO2_BASE_HEIGHT_M,
) -> tuple[float, float, float]:
    """Convert an Isaac Go2 root position to a Habitat navigation position.

    The Go2 root is nominally 0.4 m above the navigation floor.  Horizontal
    coordinates remain exact; the renderer later replaces the estimated floor
    height with the nearest point on the scene navmesh.
    """
    if len(position) != 3:
        raise ValueError("Isaac position must contain exactly three values")
    x, y, z = (float(value) for value in position)
    if not all(math.isfinite(value) for value in (x, y, z, base_height_m)):
        raise ValueError("Isaac position and base height must be finite")
    return (x, z - float(base_height_m), -y)


def habitat_position_to_isaac_root(
    position: Sequence[float],
    *,
    base_height_m: float = DEFAULT_GO2_BASE_HEIGHT_M,
) -> tuple[float, float, float]:
    """Inverse of :func:`isaac_position_to_habitat` for a nominal Go2 root."""
    if len(position) != 3:
        raise ValueError("Habitat position must contain exactly three values")
    x, y, z = (float(value) for value in position)
    if not all(math.isfinite(value) for value in (x, y, z, base_height_m)):
        raise ValueError("Habitat position and base height must be finite")
    return (x, -z, y + float(base_height_m))


def isaac_wxyz_to_yaw(rotation_wxyz: Sequence[float]) -> float:
    """Return Isaac's planar yaw from a normalized-or-unnormalized WXYZ quaternion."""
    if len(rotation_wxyz) != 4:
        raise ValueError("Isaac rotation must contain exactly four values")
    w, x, y, z = (float(value) for value in rotation_wxyz)
    if not all(math.isfinite(value) for value in (w, x, y, z)):
        raise ValueError("Isaac rotation must be finite")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise ValueError("Isaac rotation quaternion has zero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def isaac_yaw_to_habitat_yaw(isaac_yaw: float) -> float:
    return wrap_angle_radians(float(isaac_yaw) - math.pi / 2.0)


def habitat_yaw_to_isaac_yaw(habitat_yaw: float) -> float:
    return wrap_angle_radians(float(habitat_yaw) + math.pi / 2.0)


def navigation_alignment_error(
    isaac_position: Sequence[float],
    isaac_yaw: float,
    habitat_position: Sequence[float],
    habitat_yaw: float,
    *,
    base_height_m: float = DEFAULT_GO2_BASE_HEIGHT_M,
) -> tuple[float, float]:
    """Return horizontal round-trip error in metres and yaw error in radians."""
    converted = isaac_position_to_habitat(
        isaac_position, base_height_m=base_height_m
    )
    if len(habitat_position) != 3:
        raise ValueError("Habitat position must contain exactly three values")
    hx, _, hz = (float(value) for value in habitat_position)
    position_error = math.hypot(converted[0] - hx, converted[2] - hz)
    yaw_error = abs(
        wrap_angle_radians(
            isaac_yaw_to_habitat_yaw(isaac_yaw) - float(habitat_yaw)
        )
    )
    return position_error, yaw_error


def quantize_dynamic_oracle(
    *,
    geodesic_distance: float,
    relative_bearing_radians: float | None,
    forward_distance_m: float = 0.75,
    success_distance_m: float = 3.0,
) -> int | None:
    """Map a shortest-path bearing/lookahead to the ten NaViLA macro actions."""
    distance = float(geodesic_distance)
    if not math.isfinite(distance) or distance < 0:
        return None
    if distance <= success_distance_m:
        return 9
    if relative_bearing_radians is None or not math.isfinite(
        relative_bearing_radians
    ):
        return None
    bearing = wrap_angle_radians(relative_bearing_radians)
    threshold = math.radians(7.5)
    if abs(bearing) + 1e-12 >= threshold:
        magnitude = min(45, max(15, int(round(abs(math.degrees(bearing)) / 15.0)) * 15))
        index = {15: 0, 30: 1, 45: 2}[magnitude]
        return index if bearing > 0 else index + 3
    # Quantize only the distance that remains outside the success region.
    remaining_distance = max(0.0, distance - float(success_distance_m))
    safe_forward = min(
        max(0.0, float(forward_distance_m)),
        remaining_distance,
    )
    if safe_forward >= 0.75:
        return 8
    if safe_forward >= 0.50:
        return 7
    return 6


def oracle_payload(action_id: int | None) -> dict[str, Any]:
    if action_id is None:
        return {"oracle_valid": False, "dynamic_oracle_action": None}
    action = action_from_id(action_id)
    return {
        "oracle_valid": True,
        "dynamic_oracle_action": action.to_dict(),
    }


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("connection closed before the message completed")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_json_message(
    connection: socket.socket,
    *,
    max_bytes: int = MAX_RENDER_MESSAGE_BYTES,
) -> dict[str, Any]:
    size = int.from_bytes(recv_exact(connection, 8), "big")
    if size <= 0 or size > max_bytes:
        raise ValueError(f"invalid framed JSON size: {size}")
    payload = json.loads(recv_exact(connection, size).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("render protocol payload must be a JSON object")
    return payload


def send_json_message(
    connection: socket.socket,
    payload: Mapping[str, Any],
) -> None:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_RENDER_MESSAGE_BYTES:
        raise ValueError("render protocol response is too large")
    connection.sendall(len(encoded).to_bytes(8, "big"))
    connection.sendall(encoded)


@dataclass(frozen=True)
class RenderedFrame:
    image: Image.Image
    metadata: dict[str, Any]


class HabitatRenderClient:
    """Synchronous fail-closed client for the separate Habitat-Sim process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_RENDER_PORT,
        *,
        timeout_seconds: float = DEFAULT_RENDER_TIMEOUT_SECONDS,
    ) -> None:
        if not 0 < int(port) < 65536:
            raise ValueError("render port must be in [1, 65535]")
        if timeout_seconds <= 0:
            raise ValueError("render timeout must be positive")
        self.host = host
        self.port = int(port)
        self.timeout_seconds = float(timeout_seconds)

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = {
            "protocol_version": LIVE_RENDER_PROTOCOL,
            **dict(payload),
        }
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout_seconds
            ) as connection:
                connection.settimeout(self.timeout_seconds)
                send_json_message(connection, request)
                response = recv_json_message(connection)
        except (OSError, ConnectionError, TimeoutError, ValueError) as error:
            raise RuntimeError(
                f"Habitat render request failed at {self.host}:{self.port}"
            ) from error
        if response.get("protocol_version") != LIVE_RENDER_PROTOCOL:
            raise RuntimeError("Habitat renderer returned an incompatible protocol")
        if response.get("ok") is not True:
            raise RuntimeError(
                f"Habitat renderer rejected request: {response.get('error', 'unknown error')}"
            )
        return response

    def health(self) -> dict[str, Any]:
        return self._request({"operation": "health"})

    def render(self, request: Mapping[str, Any]) -> RenderedFrame:
        response = self._request({"operation": "render", **dict(request)})
        encoded = response.pop("image_png_base64", None)
        if not isinstance(encoded, str):
            raise RuntimeError("Habitat renderer response has no RGB image")
        try:
            raw_image = base64.b64decode(encoded, validate=True)
        except Exception as error:
            raise RuntimeError(
                "Habitat renderer returned invalid base64 image data"
            ) from error
        expected_size = response.get("image_byte_count")
        if not isinstance(expected_size, int) or expected_size != len(raw_image):
            raise RuntimeError(
                "Habitat renderer image length mismatch: "
                f"received={len(raw_image)} expected={expected_size!r}"
            )
        expected_digest = response.get("image_sha256")
        actual_digest = hashlib.sha256(raw_image).hexdigest()
        if not isinstance(expected_digest, str) or expected_digest != actual_digest:
            raise RuntimeError(
                "Habitat renderer image SHA-256 mismatch: "
                f"received={actual_digest} expected={expected_digest!r}"
            )
        if not raw_image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(
                "Habitat renderer returned data without a PNG signature: "
                f"bytes={len(raw_image)} prefix={raw_image[:8].hex()}"
            )
        try:
            opened = Image.open(BytesIO(raw_image))
            opened.load()
            image = opened.convert("RGB")
            response["client_image_decoder"] = "pillow"
        except Exception as pillow_error:
            # Isaac Kit can alter its embedded Python/plugin search paths after
            # startup.  Decode the already signature/length/SHA-verified bytes
            # with OpenCV when that makes Pillow's PNG plugin unavailable.
            try:
                import cv2
                import numpy as np

                decoded = cv2.imdecode(
                    np.frombuffer(raw_image, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if decoded is None:
                    raise ValueError("cv2.imdecode returned no image")
                image = Image.fromarray(
                    cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
                    mode="RGB",
                )
                response["client_image_decoder"] = "opencv"
                response["pillow_decode_fallback"] = type(pillow_error).__name__
            except Exception as opencv_error:
                raise RuntimeError(
                    "Habitat renderer returned an undecodable PNG after "
                    "Pillow and OpenCV: "
                    f"bytes={len(raw_image)} sha256={actual_digest}; "
                    f"pillow={type(pillow_error).__name__}; "
                    f"opencv={type(opencv_error).__name__}"
                ) from opencv_error
        return RenderedFrame(image=image, metadata=response)
