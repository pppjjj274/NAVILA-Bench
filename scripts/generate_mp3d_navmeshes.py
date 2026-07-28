#!/usr/bin/env python3
"""Generate and validate Habitat-Sim 0.1.7 navmeshes for MP3D GLBs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

if not hasattr(np, "float"):
    np.float = float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_name(scene_id: str) -> str:
    return Path(scene_id).stem


def _load_episodes(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"metadata has no episodes list: {path}")
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        by_scene.setdefault(_scene_name(str(episode["scene_id"])), []).append(episode)
    return by_scene


def _configuration(habitat_sim, glb: Path, args: argparse.Namespace):
    simulator_cfg = habitat_sim.SimulatorConfiguration()
    simulator_cfg.scene_id = str(glb)
    simulator_cfg.create_renderer = False
    simulator_cfg.requires_textures = False
    simulator_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = float(args.agent_height)
    agent_cfg.radius = float(args.agent_radius)
    return habitat_sim.Configuration(simulator_cfg, [agent_cfg])


def _validate_episode(pathfinder, habitat_sim, episode: dict[str, Any], max_snap: float):
    start = np.asarray(episode["start_position"], dtype=np.float32)
    goal = np.asarray(episode["goals"][0]["position"], dtype=np.float32)
    snapped_start = np.asarray(pathfinder.snap_point(start), dtype=np.float32)
    snapped_goal = np.asarray(pathfinder.snap_point(goal), dtype=np.float32)
    start_valid = bool(np.all(np.isfinite(snapped_start)))
    goal_valid = bool(np.all(np.isfinite(snapped_goal)))
    start_snap = (
        float(np.linalg.norm((snapped_start - start)[[0, 2]]))
        if start_valid
        else math.inf
    )
    goal_snap = (
        float(np.linalg.norm((snapped_goal - goal)[[0, 2]]))
        if goal_valid
        else math.inf
    )
    path = habitat_sim.ShortestPath()
    path.requested_start = snapped_start
    path.requested_end = snapped_goal
    finite_path = bool(
        start_valid
        and goal_valid
        and pathfinder.find_path(path)
        and math.isfinite(float(path.geodesic_distance))
    )
    valid = bool(
        finite_path and start_snap <= max_snap and goal_snap <= max_snap
    )
    return {
        "episode_id": str(episode.get("episode_id")),
        "valid": valid,
        "start_snap_distance": start_snap if math.isfinite(start_snap) else None,
        "goal_snap_distance": goal_snap if math.isfinite(goal_snap) else None,
        "geodesic_distance": (
            float(path.geodesic_distance) if finite_path else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--agent-height", type=float, default=1.5)
    parser.add_argument("--agent-radius", type=float, default=0.1)
    parser.add_argument("--max-snap-distance", type=float, default=0.25)
    parser.add_argument("--minimum-coverage", type=float, default=0.99)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.agent_height <= 0 or args.agent_radius <= 0:
        parser.error("agent dimensions must be positive")
    if args.max_snap_distance <= 0:
        parser.error("--max-snap-distance must be positive")
    if not 0 <= args.minimum_coverage <= 1:
        parser.error("--minimum-coverage must be in [0, 1]")
    return args


def main() -> int:
    args = parse_args()
    try:
        import habitat_sim
    except ImportError as error:
        raise RuntimeError(
            "install the local Habitat-Sim 0.1.7 headless package first"
        ) from error
    root = Path(args.scenes_root).expanduser().resolve()
    metadata_path = (
        Path(args.metadata).expanduser().resolve() if args.metadata else None
    )
    episodes_by_scene = _load_episodes(metadata_path)
    glbs = sorted(root.glob("*/*.glb"))
    if not glbs:
        raise RuntimeError(f"no MP3D GLBs found beneath {root}")

    manifest: dict[str, Any] = {
        "format": "safe-vln-mp3d-navmesh-v1",
        "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
        "scenes_root": str(root),
        "metadata": str(metadata_path) if metadata_path else None,
        "settings": {
            "agent_height": args.agent_height,
            "agent_radius": args.agent_radius,
            "max_snap_distance": args.max_snap_distance,
        },
        "navmesh_source": "locally_recomputed_habitat_sim_0.1.7",
        "scenes": [],
    }
    total_episodes = 0
    valid_episodes = 0
    for glb in glbs:
        scene_started = time.perf_counter()
        scene = glb.stem
        navmesh = glb.with_suffix(".navmesh")
        simulator = habitat_sim.Simulator(_configuration(habitat_sim, glb, args))
        try:
            if args.force or not navmesh.is_file():
                settings = habitat_sim.NavMeshSettings()
                settings.set_defaults()
                settings.agent_height = float(args.agent_height)
                settings.agent_radius = float(args.agent_radius)
                if not simulator.recompute_navmesh(simulator.pathfinder, settings):
                    raise RuntimeError(f"failed to recompute navmesh for {scene}")
                temporary = navmesh.with_suffix(".navmesh.incomplete")
                if not simulator.pathfinder.save_nav_mesh(str(temporary)):
                    raise RuntimeError(f"failed to save navmesh for {scene}")
                temporary.replace(navmesh)
            elif not simulator.pathfinder.load_nav_mesh(str(navmesh)):
                raise RuntimeError(f"failed to load existing navmesh for {scene}")

            validations = [
                _validate_episode(
                    simulator.pathfinder,
                    habitat_sim,
                    episode,
                    args.max_snap_distance,
                )
                for episode in episodes_by_scene.get(scene, [])
            ]
            scene_valid = sum(record["valid"] for record in validations)
            total_episodes += len(validations)
            valid_episodes += scene_valid
            manifest["scenes"].append(
                {
                    "scene_id": scene,
                    "glb": str(glb),
                    "navmesh": str(navmesh),
                    "glb_sha256": _sha256(glb),
                    "navmesh_sha256": _sha256(navmesh),
                    "navigable_area": float(simulator.pathfinder.navigable_area),
                    "episodes": len(validations),
                    "valid_episodes": scene_valid,
                    "coverage": (
                        scene_valid / len(validations) if validations else None
                    ),
                    "invalid_episode_ids": [
                        record["episode_id"]
                        for record in validations
                        if not record["valid"]
                    ],
                    "elapsed_seconds": time.perf_counter() - scene_started,
                }
            )
            print(
                f"[NAVMESH] {scene}: area={simulator.pathfinder.navigable_area:.2f} "
                f"coverage={scene_valid}/{len(validations)}",
                flush=True,
            )
        finally:
            simulator.close()

    coverage = valid_episodes / total_episodes if total_episodes else 1.0
    manifest.update(
        {
            "scene_count": len(manifest["scenes"]),
            "episode_count": total_episodes,
            "valid_episode_count": valid_episodes,
            "coverage": coverage,
            "minimum_coverage": args.minimum_coverage,
            "coverage_gate_passed": coverage >= args.minimum_coverage,
        }
    )
    output = Path(args.output_manifest).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({key: manifest[key] for key in (
        "scene_count",
        "episode_count",
        "valid_episode_count",
        "coverage",
        "coverage_gate_passed",
    )}, indent=2), flush=True)
    return 0 if manifest["coverage_gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
