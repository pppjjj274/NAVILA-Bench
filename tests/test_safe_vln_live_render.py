import base64
import hashlib
from io import BytesIO
import math
import numpy as np
import pytest
from PIL import Image

from safe_vln.live_render import (
    HabitatRenderClient,
    LIVE_RENDER_PROTOCOL,
    habitat_position_to_isaac_root,
    habitat_yaw_to_isaac_yaw,
    isaac_position_to_habitat,
    isaac_wxyz_to_yaw,
    isaac_yaw_to_habitat_yaw,
    navigation_alignment_error,
    quantize_dynamic_oracle,
    recv_json_message,
    sample_navila_history,
    send_json_message,
)
from scripts.habitat_render_server import _install_numpy_legacy_aliases


def test_go2_habitat_navigation_pose_round_trip():
    isaac = (15.0, 4.5, 0.6)
    habitat = isaac_position_to_habitat(isaac)
    assert habitat == pytest.approx((15.0, 0.2, -4.5))
    assert habitat_position_to_isaac_root(habitat) == pytest.approx(isaac)

    isaac_yaw = math.radians(75)
    habitat_yaw = isaac_yaw_to_habitat_yaw(isaac_yaw)
    assert habitat_yaw == pytest.approx(math.radians(-15))
    assert habitat_yaw_to_isaac_yaw(habitat_yaw) == pytest.approx(isaac_yaw)


def test_navigation_alignment_ignores_navmesh_floor_snap_but_not_xy_or_yaw():
    position_error, yaw_error = navigation_alignment_error(
        (1.0, 2.0, 0.5),
        math.radians(90),
        (1.0, 7.5, -2.0),
        0.0,
    )
    assert position_error == pytest.approx(0.0)
    assert yaw_error == pytest.approx(0.0)


def test_isaac_quaternion_yaw_accepts_non_unit_input():
    assert isaac_wxyz_to_yaw((2.0, 0.0, 0.0, 2.0)) == pytest.approx(
        math.pi / 2
    )


@pytest.mark.parametrize(
    ("distance", "bearing", "lookahead", "expected"),
    [
        (3.0, None, 0.75, 9),
        (4.0, math.radians(7.5), 0.75, 0),
        (4.0, math.radians(23), 0.75, 1),
        (4.0, math.radians(80), 0.75, 2),
        (4.0, math.radians(-20), 0.75, 3),
        (4.0, 0.0, 0.75, 8),
        (4.0, 0.0, 0.50, 7),
        (4.0, 0.0, 0.20, 6),
        (math.inf, 0.0, 0.75, None),
    ],
)
def test_dynamic_oracle_macro_quantization(distance, bearing, lookahead, expected):
    assert (
        quantize_dynamic_oracle(
            geodesic_distance=distance,
            relative_bearing_radians=bearing,
            forward_distance_m=lookahead,
        )
        == expected
    )


def test_dynamic_oracle_uses_episode_specific_goal_radius():
    assert quantize_dynamic_oracle(
        geodesic_distance=2.0,
        relative_bearing_radians=0.0,
        success_distance_m=1.0,
    ) == 8
    assert quantize_dynamic_oracle(
        geodesic_distance=2.0,
        relative_bearing_radians=None,
        success_distance_m=3.0,
    ) == 9


def test_render_protocol_round_trip_uses_length_prefix():
    class MemoryConnection:
        def __init__(self):
            self.buffer = bytearray()

        def sendall(self, value):
            self.buffer.extend(value)

        def recv(self, size):
            value = bytes(self.buffer[:size])
            del self.buffer[:size]
            return value

    connection = MemoryConnection()
    payload = {
        "protocol_version": LIVE_RENDER_PROTOCOL,
        "operation": "health",
    }
    send_json_message(connection, payload)
    assert recv_json_message(connection) == payload


def test_habitat_017_numpy_compatibility_aliases():
    _install_numpy_legacy_aliases()
    assert np.__dict__["float"] is float
    assert np.__dict__["int"] is int
    assert np.__dict__["bool"] is bool


def test_render_client_verifies_png_integrity(monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), (1, 2, 3)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    response = {
        "image_png_base64": base64.b64encode(raw).decode("ascii"),
        "image_byte_count": len(raw),
        "image_sha256": hashlib.sha256(raw).hexdigest(),
    }
    client = HabitatRenderClient()
    monkeypatch.setattr(client, "_request", lambda payload: dict(response))
    assert client.render({}).image.size == (3, 2)

    response["image_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        client.render({})


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (1, ["pad"] * 7 + [0]),
        (7, ["pad", 0, 1, 2, 3, 4, 5, 6]),
        (8, list(range(8))),
        (20, [0, 2, 5, 8, 10, 13, 16, 19]),
    ],
)
def test_navila_full_history_sampling_matches_official_policy(length, expected):
    sampled = sample_navila_history(
        list(range(length)), padding_factory=lambda: "pad"
    )
    assert sampled == expected


def test_dynamic_oracle_quantizes_distance_outside_success_radius():
    assert quantize_dynamic_oracle(
        geodesic_distance=3.0,
        relative_bearing_radians=None,
    ) == 9
    assert quantize_dynamic_oracle(
        geodesic_distance=3.49,
        relative_bearing_radians=0.0,
    ) == 6
    assert quantize_dynamic_oracle(
        geodesic_distance=3.50,
        relative_bearing_radians=0.0,
    ) == 7
    assert quantize_dynamic_oracle(
        geodesic_distance=3.749,
        relative_bearing_radians=0.0,
    ) == 7
    assert quantize_dynamic_oracle(
        geodesic_distance=3.75,
        relative_bearing_radians=0.0,
    ) == 8
