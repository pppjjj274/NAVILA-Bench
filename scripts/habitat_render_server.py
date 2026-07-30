#!/usr/bin/env python3
"""Headless MP3D RGB/geodesic service for strict Go2 Safe-VLN alignment."""

from __future__ import annotations

import argparse
import base64
import hashlib
from io import BytesIO
import math
from pathlib import Path
import re
import socketserver
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from safe_vln.live_render import (
    LIVE_RENDER_PROTOCOL,
    isaac_position_to_habitat,
    isaac_wxyz_to_yaw,
    isaac_yaw_to_habitat_yaw,
    navigation_oracle_invalid_reason,
    oracle_payload,
    quantize_dynamic_oracle,
    recv_json_message,
    send_json_message,
    wrap_angle_radians,
)


_SCENE_ID = re.compile(r"^[A-Za-z0-9]+$")


def _install_numpy_legacy_aliases() -> None:
    """Restore aliases used by Habitat-Sim 0.1.7 under NumPy 1.24+."""
    for name, value in {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }.items():
        np.__dict__.setdefault(name, value)


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise ValueError(f"{name} must contain exactly {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _scene_name(value: Any) -> str:
    name = Path(str(value)).stem
    if _SCENE_ID.fullmatch(name) is None:
        raise ValueError(f"invalid MP3D scene id: {value!r}")
    return name


def _lookahead_point(points: Sequence[Sequence[float]], distance: float) -> np.ndarray:
    if not points:
        raise ValueError("shortest path has no points")
    current = np.asarray(points[0], dtype=np.float64)
    remaining = float(distance)
    for raw_next in points[1:]:
        next_point = np.asarray(raw_next, dtype=np.float64)
        segment = next_point - current
        horizontal = np.asarray([segment[0], 0.0, segment[2]])
        length = float(np.linalg.norm(horizontal))
        if length >= remaining and length > 1e-8:
            return current + horizontal * (remaining / length)
        remaining -= length
        current = next_point
    return np.asarray(points[-1], dtype=np.float64)


def _relative_path_bearing(
    start: Sequence[float],
    lookahead: Sequence[float],
    habitat_yaw: float,
) -> float | None:
    dx = float(lookahead[0]) - float(start[0])
    dz = float(lookahead[2]) - float(start[2])
    if math.hypot(dx, dz) <= 1e-6:
        return None
    # Habitat's yaw-zero forward direction is -Z.
    desired_yaw = math.atan2(-dx, -dz)
    return wrap_angle_radians(desired_yaw - habitat_yaw)


class HabitatRenderer:
    def __init__(self, args: argparse.Namespace) -> None:
        _install_numpy_legacy_aliases()
        try:
            import habitat_sim
            import quaternion  # noqa: F401  # registers np.quaternion
        except ImportError as error:
            raise RuntimeError(
                "Habitat-Sim 0.1.7 is required; install the local headless package "
                "in the vlnce3 environment"
            ) from error
        self.habitat_sim = habitat_sim
        self.args = args
        self.scenes_root = Path(args.scenes_root).expanduser().resolve()
        if not self.scenes_root.is_dir():
            raise FileNotFoundError(f"MP3D scenes root does not exist: {self.scenes_root}")
        self.simulator = None
        self.agent = None
        self.loaded_scene = None

    def close(self) -> None:
        if self.simulator is not None:
            self.simulator.close()
        self.simulator = None
        self.agent = None
        self.loaded_scene = None

    def _paths(self, scene_name: str) -> tuple[Path, Path]:
        scene_dir = (self.scenes_root / scene_name).resolve()
        if self.scenes_root not in scene_dir.parents:
            raise ValueError("scene path escapes the configured MP3D root")
        return scene_dir / f"{scene_name}.glb", scene_dir / f"{scene_name}.navmesh"

    def load_scene(self, scene_name: str) -> None:
        if self.loaded_scene == scene_name:
            return
        glb_path, navmesh_path = self._paths(scene_name)
        if not glb_path.is_file():
            raise FileNotFoundError(f"MP3D GLB does not exist: {glb_path}")
        if not navmesh_path.is_file():
            raise FileNotFoundError(
                f"MP3D navmesh does not exist: {navmesh_path}; run "
                "scripts/generate_mp3d_navmeshes.py first"
            )
        self.close()
        habitat_sim = self.habitat_sim
        simulator_cfg = habitat_sim.SimulatorConfiguration()
        simulator_cfg.scene_id = str(glb_path)
        simulator_cfg.gpu_device_id = int(self.args.gpu_device_id)
        simulator_cfg.create_renderer = True
        simulator_cfg.enable_physics = False

        sensor = habitat_sim.SensorSpec()
        sensor.uuid = "rgb"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor.resolution = [int(self.args.height), int(self.args.width)]
        sensor.position = [0.0, float(self.args.sensor_height), 0.0]
        sensor.orientation = [0.0, 0.0, 0.0]
        sensor.parameters["hfov"] = str(float(self.args.hfov))

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = float(self.args.agent_height)
        agent_cfg.radius = float(self.args.agent_radius)
        agent_cfg.sensor_specifications = [sensor]
        configuration = habitat_sim.Configuration(simulator_cfg, [agent_cfg])
        self.simulator = habitat_sim.Simulator(configuration)
        self.agent = self.simulator.get_agent(0)
        if not self.simulator.pathfinder.is_loaded:
            self.close()
            raise RuntimeError(f"failed to load navmesh: {navmesh_path}")
        self.loaded_scene = scene_name

    def _shortest_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[float, list[np.ndarray]]:
        path = self.habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if not self.simulator.pathfinder.find_path(path):
            return math.inf, []
        return float(path.geodesic_distance), list(path.points)

    def render(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        scene_name = _scene_name(request.get("scene_id"))
        request_id = str(request.get("request_id", ""))
        episode_id = str(request.get("episode_id", ""))
        if not request_id or not episode_id:
            raise ValueError("render request requires request_id and episode_id")
        position = np.asarray(
            _finite_vector(request.get("habitat_position"), 3, "habitat_position"),
            dtype=np.float32,
        )
        goal = np.asarray(
            _finite_vector(request.get("goal_position"), 3, "goal_position"),
            dtype=np.float32,
        )
        yaw = float(request.get("habitat_yaw"))
        if not math.isfinite(yaw):
            raise ValueError("habitat_yaw must be finite")
        success_distance_m = float(request.get("success_distance_m"))
        if not math.isfinite(success_distance_m) or success_distance_m <= 0:
            raise ValueError("success_distance_m must be finite and positive")
        physics_step = int(request.get("physics_step"))
        isaac_pose = request.get("isaac_pose")
        if not isinstance(isaac_pose, dict):
            raise ValueError("render request requires isaac_pose")
        isaac_position = _finite_vector(
            isaac_pose.get("position"), 3, "isaac_pose.position"
        )
        isaac_rotation = _finite_vector(
            isaac_pose.get("rotation_wxyz"), 4, "isaac_pose.rotation_wxyz"
        )
        expected_position = isaac_position_to_habitat(isaac_position)
        expected_yaw = isaac_yaw_to_habitat_yaw(
            isaac_wxyz_to_yaw(isaac_rotation)
        )
        coordinate_error = math.hypot(
            expected_position[0] - float(position[0]),
            expected_position[2] - float(position[2]),
        )
        coordinate_yaw_error = abs(
            wrap_angle_radians(expected_yaw - yaw)
        )
        if coordinate_error > 0.02 or coordinate_yaw_error > math.radians(1.0):
            raise ValueError(
                "Isaac/Habitat request poses are inconsistent: "
                f"{coordinate_error:.6f}m, "
                f"{math.degrees(coordinate_yaw_error):.4f}deg"
            )
        self.load_scene(scene_name)

        snapped = np.asarray(self.simulator.pathfinder.snap_point(position), dtype=np.float32)
        snap_valid = bool(np.all(np.isfinite(snapped)))
        render_position = position.copy()
        if snap_valid:
            # Preserve the actual Go2 horizontal location; use navmesh only for floor Y.
            render_position[1] = snapped[1]
        horizontal_snap_distance = (
            float(np.linalg.norm((snapped - position)[[0, 2]]))
            if snap_valid
            else math.inf
        )
        snapped_goal = np.asarray(self.simulator.pathfinder.snap_point(goal), dtype=np.float32)
        goal_valid = bool(np.all(np.isfinite(snapped_goal)))
        goal_snap_distance = (
            float(np.linalg.norm((snapped_goal - goal)[[0, 2]]))
            if goal_valid
            else math.inf
        )
        if snap_valid and goal_valid:
            geodesic_distance, path_points = self._shortest_path(snapped, snapped_goal)
        else:
            geodesic_distance, path_points = math.inf, []
        oracle_invalid_reason = navigation_oracle_invalid_reason(
            start_snap_valid=snap_valid,
            goal_snap_valid=goal_valid,
            start_snap_distance_m=(
                horizontal_snap_distance
                if math.isfinite(horizontal_snap_distance)
                else None
            ),
            goal_snap_distance_m=(
                goal_snap_distance if math.isfinite(goal_snap_distance) else None
            ),
            geodesic_distance_m=(
                geodesic_distance if math.isfinite(geodesic_distance) else None
            ),
            max_snap_distance_m=float(self.args.max_snap_distance),
        )
        reward_valid = oracle_invalid_reason is None

        state = self.agent.get_state()
        state.position = render_position
        state.rotation = np.quaternion(
            math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0
        )
        self.agent.set_state(state, reset_sensors=True)
        observation = np.asarray(self.simulator.get_sensor_observations()["rgb"])
        image = Image.fromarray(observation[:, :, :3].astype(np.uint8), mode="RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(
                "Pillow produced data without a PNG signature: "
                f"bytes={len(png_bytes)} prefix={png_bytes[:8].hex()}"
            )

        if reward_valid:
            lookahead = _lookahead_point(path_points, self.args.oracle_lookahead)
            relative_bearing = _relative_path_bearing(snapped, lookahead, yaw)
            action_id = quantize_dynamic_oracle(
                geodesic_distance=geodesic_distance,
                relative_bearing_radians=relative_bearing,
                forward_distance_m=self.args.oracle_lookahead,
                success_distance_m=success_distance_m,
            )
            if action_id is None:
                oracle_invalid_reason = "path_bearing_degenerate"
        else:
            relative_bearing = None
            action_id = None
        return {
            "request_id": request_id,
            "episode_id": episode_id,
            "scene_id": scene_name,
            "physics_step": physics_step,
            "image_encoding": "png-base64",
            "image_byte_count": len(png_bytes),
            "image_sha256": hashlib.sha256(png_bytes).hexdigest(),
            "image_png_base64": base64.b64encode(png_bytes).decode("ascii"),
            "applied_pose": {
                "position": render_position.astype(float).tolist(),
                "yaw": yaw,
            },
            "nearest_navmesh_point": (
                snapped.astype(float).tolist() if snap_valid else None
            ),
            "navmesh_snap_distance": (
                horizontal_snap_distance if math.isfinite(horizontal_snap_distance) else None
            ),
            "start_snap_valid": snap_valid,
            "goal_snap_valid": goal_valid,
            "start_snap_distance_m": (
                horizontal_snap_distance
                if math.isfinite(horizontal_snap_distance)
                else None
            ),
            "goal_snap_distance_m": (
                goal_snap_distance if math.isfinite(goal_snap_distance) else None
            ),
            "is_navigable": bool(
                snap_valid and self.simulator.pathfinder.is_navigable(snapped)
            ),
            "geodesic_distance": (
                geodesic_distance if math.isfinite(geodesic_distance) else None
            ),
            "navigation_reward_valid": reward_valid,
            "oracle_invalid_reason": oracle_invalid_reason,
            "success_distance_m": success_distance_m,
            "relative_path_bearing": relative_bearing,
            **oracle_payload(action_id),
            "render_latency_ms": (time.perf_counter() - started) * 1000.0,
            "camera": {
                "width": int(self.args.width),
                "height": int(self.args.height),
                "hfov": float(self.args.hfov),
                "sensor_height": float(self.args.sensor_height),
                "pose_policy": "navila_upright_1.25m",
            },
        }


class _RenderHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        response: dict[str, Any] = {
            "protocol_version": LIVE_RENDER_PROTOCOL,
            "ok": False,
        }
        try:
            request = recv_json_message(self.request)
            if request.get("protocol_version") != LIVE_RENDER_PROTOCOL:
                raise ValueError("incompatible render protocol")
            operation = request.get("operation")
            renderer: HabitatRenderer = self.server.renderer
            if operation == "health":
                response.update(
                    {
                        "ok": True,
                        "habitat_sim_version": getattr(
                            renderer.habitat_sim, "__version__", "unknown"
                        ),
                        "scenes_root": str(renderer.scenes_root),
                        "loaded_scene": renderer.loaded_scene,
                        "gpu_device_id": int(renderer.args.gpu_device_id),
                    }
                )
            elif operation == "render":
                response.update(renderer.render(request))
                response["ok"] = True
            else:
                raise ValueError(f"unknown operation: {operation!r}")
        except Exception as error:
            response["error"] = f"{type(error).__name__}: {error}"
        send_json_message(self.request, response)


class _RenderServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, address, handler, renderer):
        self.renderer = renderer
        super().__init__(address, handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54322)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--sensor-height", type=float, default=1.25)
    parser.add_argument("--agent-height", type=float, default=1.5)
    parser.add_argument("--agent-radius", type=float, default=0.1)
    parser.add_argument("--max-snap-distance", type=float, default=0.25)
    parser.add_argument("--oracle-lookahead", type=float, default=0.75)
    parser.add_argument("--success-distance", type=float, default=3.0)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be in [1, 65535]")
    for name in (
        "width",
        "height",
        "hfov",
        "sensor_height",
        "agent_height",
        "agent_radius",
        "max_snap_distance",
        "oracle_lookahead",
        "success_distance",
    ):
        if float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    renderer = HabitatRenderer(args)
    server = _RenderServer((args.host, args.port), _RenderHandler, renderer)
    print(
        f"[HABITAT-RENDER] listening on {args.host}:{args.port}; "
        f"scenes={renderer.scenes_root}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
