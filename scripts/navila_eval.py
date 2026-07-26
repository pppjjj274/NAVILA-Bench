# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import json
import math
import re
import time
import base64
import io
import socket
import json


from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from safe_vln.objective import (
    build_objective_config,
    canonical_fingerprint,
    default_cost_profile,
    load_cost_profile,
    validate_cost_profile,
)
from safe_vln.replay import load_r2r_replay_episode

# isaaclab argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--visualize_path", action="store_true", default=False, help="Visualize the path in the simulator.")

# navila argparse arguments
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--vlm_host", type=str, default="localhost")
parser.add_argument("--vlm_port", type=int, default=54321)
parser.add_argument("--max_episode_seconds", type=float, default=None)
parser.add_argument("--max_vlm_calls", type=int, default=None)
parser.add_argument("--safe-vln", action="store_true", help="Enable Go2 CMDP safety evaluation and trajectory output.")
parser.add_argument("--safe-gamma", type=float, default=0.99)
parser.add_argument("--safe-cost-limit", type=float, default=None)
parser.add_argument("--safe-cost-profile", type=str, default=None)
parser.add_argument("--safe-calibration-file", type=str, default=None)
parser.add_argument("--safe-contact-threshold", type=float, default=1.0)
parser.add_argument("--safe-orientation-limit", type=float, default=0.8)
parser.add_argument("--safe-blocked-seconds", type=float, default=2.0)
parser.add_argument("--safe-blocked-distance", type=float, default=0.10)
parser.add_argument("--progress-reward-scale", type=float, default=1.0)
parser.add_argument("--success-reward", type=float, default=10.0)
parser.add_argument("--macro-step-penalty", type=float, default=-0.01)
parser.add_argument("--failed-stop-penalty", type=float, default=-1.0)
parser.add_argument("--safe-dataset-dir", type=str, default=None)
parser.add_argument(
    "--safe-replay",
    action="store_true",
    help="Use offline R2R frames for NaViLA while retaining Go2 physics and safety.",
)
parser.add_argument("--safe-replay-root", type=str, default=None)
parser.add_argument("--safe-replay-annotations", type=str, default=None)
parser.add_argument("--safe-replay-id", type=int, default=None)
parser.add_argument(
    "--safe-policy-tag",
    type=str,
    default=None,
    help="Filesystem-safe policy label added to Safe-Replay IDs and outputs.",
)


# r2r argparse arguments
parser.add_argument("--episode_idx", type=int, default=0)

# RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.max_episode_seconds is not None and args_cli.max_episode_seconds <= 0:
    parser.error("--max_episode_seconds must be positive")
if args_cli.max_vlm_calls is not None and args_cli.max_vlm_calls <= 0:
    parser.error("--max_vlm_calls must be positive")
if args_cli.safe_blocked_seconds <= 0:
    parser.error("--safe-blocked-seconds must be positive")
if args_cli.safe_blocked_distance <= 0:
    parser.error("--safe-blocked-distance must be positive")
if args_cli.safe_calibration_file and not args_cli.safe_vln:
    parser.error("--safe-calibration-file requires --safe-vln")
if args_cli.safe_policy_tag is not None:
    if not args_cli.safe_vln:
        parser.error("--safe-policy-tag requires --safe-vln")
    if re.fullmatch(r"[A-Za-z0-9._-]+", args_cli.safe_policy_tag) is None:
        parser.error(
            "--safe-policy-tag may contain only letters, digits, '.', '_' and '-'"
        )
if args_cli.safe_replay:
    if not args_cli.safe_vln:
        parser.error("--safe-replay requires --safe-vln")
    if args_cli.safe_replay_root is None:
        parser.error("--safe-replay requires --safe-replay-root")
    if args_cli.safe_replay_id is None:
        parser.error("--safe-replay requires --safe-replay-id")
    if getattr(args_cli, "enable_cameras", False):
        parser.error("--safe-replay cannot be combined with --enable_cameras")
    if not getattr(args_cli, "headless", False):
        parser.error("--safe-replay requires --headless")

replay_episode_preflight = None
if args_cli.safe_replay:
    replay_episode_preflight = load_r2r_replay_episode(
        args_cli.safe_replay_root,
        args_cli.safe_replay_id,
        annotations_path=args_cli.safe_replay_annotations,
    )

if args_cli.safe_cost_profile:
    safe_cost_profile = load_cost_profile(args_cli.safe_cost_profile)
else:
    profile_payload = default_cost_profile()
    profile_payload.pop("fingerprint", None)
    profile_payload["hard_thresholds"].update(
        {
            "contact_force_n": args_cli.safe_contact_threshold,
            "orientation_rad": args_cli.safe_orientation_limit,
            "blocked_seconds": args_cli.safe_blocked_seconds,
            "blocked_distance_m": args_cli.safe_blocked_distance,
        }
    )
    safe_cost_profile = validate_cost_profile(profile_payload)
if args_cli.safe_cost_limit is None:
    args_cli.safe_cost_limit = float(safe_cost_profile["cost_limit"])
else:
    safe_cost_profile = dict(safe_cost_profile)
    safe_cost_profile.pop("fingerprint", None)
    safe_cost_profile["cost_limit"] = float(args_cli.safe_cost_limit)
    safe_cost_profile = validate_cost_profile(safe_cost_profile)
safe_objective_config = build_objective_config(safe_cost_profile)
safe_objective_config["online_reward"].update(
    {
        "progress_scale": float(args_cli.progress_reward_scale),
        "macro_step_penalty": float(args_cli.macro_step_penalty),
        "success_reward": float(args_cli.success_reward),
        "failed_stop_penalty": float(args_cli.failed_stop_penalty),
    }
)
safe_objective_config.pop("fingerprint", None)
safe_objective_config["fingerprint"] = canonical_fingerprint(
    safe_objective_config
)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw

from safe_vln.actions import normalize_policy_response
from safe_vln.checkpoint import load_go2_inference_checkpoint
from safe_vln.dataset import SafeVLNShardWriter, write_episode_summary
from safe_vln.objective import graded_oracle_reward
from safe_vln.trajectory import SafeTrajectoryRecorder

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab.utils.io import load_yaml
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils import update_class_from_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
import omni.isaac.lab.sim as sim_utils

from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper
from omni.isaac.vlnce.utils.eval_utils import (
    get_vel_command, 
    read_episodes, 
    add_instruction_on_img,
    InstructionData, 
)
from omni.isaac.vlnce.utils.measures import PathLength, DistanceToGoal, Success, SPL, OracleNavigationError, OracleSuccess, MeasureManager


def quat2eulers(q0, q1, q2, q3):
    """
    Calculates the roll, pitch, and yaw angles from a quaternion.

    Args:
        q0: The scalar component of the quaternion.
        q1: The x-component of the quaternion.
        q2: The y-component of the quaternion.
        q3: The z-component of the quaternion.

    Returns:
        A tuple containing the roll, pitch, and yaw angles in radians.
    """

    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0**2 + q1**2 - q2**2 - q3**2)

    return roll, pitch, yaw


def define_markers() -> VisualizationMarkers:
    """Define path markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pathMarkers",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def reset_start_pos_rot(env_cfg, args_cli, episode):
    scene_id = os.path.splitext(os.path.basename(episode["scene_id"]))[0]
    env_cfg.scene.terrain.obj_filepath = os.path.join(ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")
    
    start_pos, start_rot, goal_pos = episode["start_position"], episode["start_rotation"], episode["reference_path"][-1]
    env_cfg.scene.robot.init_state.rot = start_rot

    if "go2" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+0.4)
    elif "h1" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+1.0)
    else:
        env_cfg.scene.robot.init_state.pos = (start_pos[0], start_pos[1], start_pos[2]+0.5)

    env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos

    env_cfg.scene.disk_1.init_state.pos = ([start_pos[0], start_pos[1], start_pos[2] + 2.5])
    env_cfg.scene.disk_2.init_state.pos = ([goal_pos[0], goal_pos[1], goal_pos[2] + 2.5])

    return env_cfg


def add_measurement(env, episode):
    measure_manager = MeasureManager()
    measure_names = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    for measure_name in measure_names:
        measure = eval(measure_name)(env, episode, measure_manager)
        measure_manager.register_measure(measure)
    
    env.measure_manager = measure_manager
    return


def sample_eight_images(image_list):
    if len(image_list) == 0:
        raise ValueError("Did not receive any images")
    if len(image_list) < 8:
        print("Not enough images received, padding.")
        image_list = image_list.copy()
        # append image value=0, in front of the existing images, image size equal to the last one
        for _ in range(8 - len(image_list)):
            image_list.insert(0, Image.new('RGB', image_list[-1].size, (0, 0, 0)))
    else:
        image_list = image_list.copy()
    num_images = len(image_list)
    indices = [int(i * (num_images - 1) / 7) for i in range(7)]
    sampled_images = [image_list[i] for i in indices]
    sampled_images.append(image_list[-1])
    return sampled_images


def sample_images_and_send_to_vlm(image_list, vlm_host, vlm_port, query, request_metadata=None):
    sampled_images = sample_eight_images(image_list)

    # save sampled images
    # time_stamp = time.strftime("%Y%m%d-%H%M%S")
    # if not os.path.exists("test_images"):
    #     os.makedirs("test_images")
    # for i, img in enumerate(sampled_images):
    #     # convert to PIL Image
    #     img = Image.fromarray(img)
    #     img.save(os.path.join("test_images", f"{time_stamp}_image_{i}.jpg"))

    # Convert images to base64 for transmission
    encoded_images = []
    for image in sampled_images:
        # Ensure PIL Image for JPEG encoding
        if isinstance(image, np.ndarray):
            array_image = image
            if array_image.dtype != np.uint8:
                # Convert to uint8. If values are 0-1, scale; otherwise clip to 0-255
                if array_image.max() <= 1.0:
                    array_image = (array_image * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    array_image = array_image.clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array_image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            # Fallback: try to construct a PIL image from whatever object is provided
            pil_image = Image.fromarray(np.array(image, dtype=np.uint8))

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # Prepare request data
    request_data = {
        'images': encoded_images,
        'query': query
    }
    if request_metadata:
        request_data.update(request_metadata)

    # Send to VLM server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((vlm_host, vlm_port))
        
        # Send data
        data_bytes = json.dumps(request_data).encode()
        s.sendall(len(data_bytes).to_bytes(8, 'big'))
        s.sendall(data_bytes)
        
        # Receive response
        size_data = s.recv(8)
        size = int.from_bytes(size_data, 'big')
        
        response_data = b''
        while len(response_data) < size:
            packet = s.recv(4096)
            if not packet:
                break
            response_data += packet
            
        response = json.loads(response_data.decode())
        return response


def _measurement_distance(infos):
    return float(infos["measurements"]["distance_to_goal"])


def _recorded_policy_version(recorder):
    versions = {
        transition.get("policy_version")
        for transition in recorder.transitions
        if transition.get("policy_version") is not None
    }
    if len(versions) > 1:
        raise RuntimeError(
            f"episode contains multiple policy versions: {sorted(versions)}"
        )
    return next(iter(versions), None)


def _replay_diagnostic_frame(frame, *, instruction, predicted, oracle, reward, cost, reason):
    image = frame.convert("RGB").resize((512, 512))
    canvas = Image.new("RGB", (1024, 512), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    lines = [
        "SAFE-REPLAY (offline visual / unpaired physics)",
        f"Instruction: {instruction}",
        f"Predicted: {predicted}",
        f"Oracle: {oracle}",
        f"Reward: {reward:.1f}    Cost: {cost:.1f}",
        f"Termination: {reason or '-'}",
    ]
    y = 20
    for line in lines:
        words = line.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate) > 475 and current:
                draw.text((532, y), current, fill="black")
                y += 24
                current = word
            else:
                current = candidate
        draw.text((532, y), current, fill="black")
        y += 30
    return np.asarray(canvas)


def run_safe_replay_episode(env, obs, infos, replay_episode, physical_episode):
    """Run offline R2R observations against live Go2 physics and safety."""
    replay_run_id = (
        f"replay{replay_episode.episode_id}_physical"
        f"{physical_episode['episode_id']}"
    )
    if args_cli.safe_policy_tag:
        replay_run_id = f"{replay_run_id}_{args_cli.safe_policy_tag}"
    recorder = SafeTrajectoryRecorder(
        episode_id=replay_run_id,
        scene_id=physical_episode["scene_id"],
        instruction=replay_episode.instruction,
        gamma=args_cli.safe_gamma,
        cost_limit=args_cli.safe_cost_limit,
        objective_config=safe_objective_config,
    )
    dataset_writer = (
        SafeVLNShardWriter(
            args_cli.safe_dataset_dir,
            objective_config=safe_objective_config,
        )
        if args_cli.safe_dataset_dir
        else None
    )
    dataset_samples = []
    diagnostic_frames = []

    step_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    max_steps = round(100 * 0.5 / step_dt)
    if args_cli.max_episode_seconds is not None:
        max_steps = min(max_steps, round(args_cli.max_episode_seconds / step_dt))

    num_steps = 0
    vlm_calls = 0
    terminal = False
    termination_reason = None
    try:
        for replay_index, replay_step in enumerate(replay_episode.steps):
            if terminal or not simulation_app.is_running() or num_steps >= max_steps:
                break
            if args_cli.max_vlm_calls is not None and vlm_calls >= args_cli.max_vlm_calls:
                termination_reason = "max_vlm_calls"
                break

            sampled_frames = replay_step.load_frames()
            response = sample_images_and_send_to_vlm(
                sampled_frames,
                args_cli.vlm_host,
                args_cli.vlm_port,
                replay_episode.instruction,
                request_metadata={
                    "protocol_version": "safe-vln-go2-v2",
                    "mode": "act",
                    "episode_id": replay_run_id,
                    "replay_episode_id": str(replay_episode.episode_id),
                    "replay_video_id": replay_step.video_id,
                    "transition_index": len(recorder.transitions),
                    "deterministic": True,
                },
            )
            vlm_calls += 1
            policy_output = normalize_policy_response(response)
            oracle_action = replay_step.oracle_action
            action_match = (
                not policy_output["invalid_action"]
                and policy_output["action_id"] == oracle_action.action_id
            )
            oracle_reward, oracle_reward_components = graded_oracle_reward(
                policy_output["action_id"],
                oracle_action.action_id,
                invalid_action=policy_output["invalid_action"],
            )
            recorder.begin(policy_output, _measurement_distance(infos))
            print(
                f"Safe-Replay output: {response}\n"
                f"Predicted {policy_output['action_id']}: {policy_output['text']}\n"
                f"Oracle {oracle_action.action_id}: {oracle_action.text}\n"
                f"Command: {policy_output['velocity_command']}, "
                f"duration: {policy_output['duration']:.2f}s\n",
                flush=True,
            )

            safety = infos.get("safety", {})
            if policy_output["action_id"] == 9:
                env.set_stop_called(True)
                env.measure_manager.update_measures()
                infos["measurements"] = env.measure_manager.get_measurements()
                termination_reason = "policy_stop"
                terminal = True
            else:
                requested_steps = max(1, round(policy_output["duration"] / step_dt))
                command = torch.tensor(
                    policy_output["velocity_command"], device=obs.device
                )
                env.begin_macro_action(command)
                for _ in range(requested_steps):
                    obs, _, done, infos = env.step(command)
                    num_steps += 1
                    safety = infos.get("safety", {})
                    recorder.record_env_step(safety)
                    if safety.get("hard_violation", False):
                        termination_reason = safety["termination_reason"]
                    elif done:
                        termination_reason = "environment_termination"
                    elif num_steps >= max_steps:
                        termination_reason = "max_episode_steps"
                    if termination_reason:
                        terminal = True
                        break

            if (
                not terminal
                and replay_index == len(replay_episode.steps) - 1
            ):
                termination_reason = "replay_exhausted"
                terminal = True

            recorder.finish(
                distance_after=_measurement_distance(infos),
                reward_override=oracle_reward,
                reward_components=oracle_reward_components,
                unsafe_contact=bool(safety.get("unsafe_contact", False)),
                fall=bool(safety.get("fall", False)),
                blocked=bool(safety.get("blocked", False)),
                safety_diagnostics=safety,
                terminated=terminal and termination_reason not in {
                    "max_episode_steps",
                    "max_vlm_calls",
                    "replay_exhausted",
                },
                truncated=termination_reason in {
                    "max_episode_steps",
                    "max_vlm_calls",
                    "replay_exhausted",
                },
                termination_reason=termination_reason,
            )
            transition = recorder.transitions[-1]
            transition.update(
                {
                    "episode_id": replay_run_id,
                    "physical_episode_id": str(physical_episode["episode_id"]),
                    "scene_id": physical_episode["scene_id"],
                    "instruction": replay_episode.instruction,
                    "replay_episode_id": str(replay_episode.episode_id),
                    "replay_video_id": replay_step.video_id,
                    "oracle_action_id": oracle_action.action_id,
                    "oracle_action_text": oracle_action.text,
                    "raw_oracle_action": replay_step.raw_oracle_action,
                    "action_match": action_match,
                    "reward_source": "graded_oracle_action",
                    "graded_oracle_reward": oracle_reward,
                    "observation_alignment": "offline_unpaired",
                    "navigation_metrics_aligned": False,
                    "policy_tag": args_cli.safe_policy_tag,
                }
            )
            transition["observation_key"] = (
                f"{replay_run_id}/state{transition['index']:06d}"
            )
            transition["next_observation_key"] = (
                None
                if transition["done"]
                else (
                    f"{replay_run_id}/state"
                    f"{transition['index'] + 1:06d}"
                )
            )
            diagnostic_frames.append(
                _replay_diagnostic_frame(
                    sampled_frames[-1],
                    instruction=replay_episode.instruction,
                    predicted=policy_output["text"],
                    oracle=oracle_action.text,
                    reward=transition["reward"],
                    cost=transition["cost"],
                    reason=termination_reason,
                )
            )
            if dataset_writer is not None:
                dataset_samples.append(
                    (transition["observation_key"], sampled_frames, transition)
                )
    finally:
        if (
            recorder.transitions
            and not recorder.transitions[-1].get("done")
            and termination_reason == "max_vlm_calls"
        ):
            recorder.transitions[-1]["done"] = True
            recorder.transitions[-1]["truncated"] = True
            recorder.transitions[-1]["termination_reason"] = termination_reason
        recorder.finalize()
        if dataset_writer is not None:
            for sample_key, frames, metadata in dataset_samples:
                dataset_writer.add(sample_key, frames, metadata)
            dataset_writer.close()
            episode_data = recorder.to_dict(infos.get("measurements", {}))
            episode_data.update(
                {
                    "replay_episode_id": str(replay_episode.episode_id),
                    "physical_episode_id": str(physical_episode["episode_id"]),
                    "observation_alignment": "offline_unpaired",
                    "navigation_metrics_aligned": False,
                    "reward_source": "graded_oracle_action",
                    "policy_tag": args_cli.safe_policy_tag,
                    "policy_version": _recorded_policy_version(recorder),
                }
            )
            write_episode_summary(args_cli.safe_dataset_dir, episode_data)
    return infos, diagnostic_frames, recorder


def run_safe_episode(env, obs, infos, image_observations, rgb_obses, instruction, episode):
    recorder = SafeTrajectoryRecorder(
        episode_id=episode["episode_id"],
        scene_id=episode["scene_id"],
        instruction=instruction.instruction_text,
        gamma=args_cli.safe_gamma,
        progress_scale=args_cli.progress_reward_scale,
        step_penalty=args_cli.macro_step_penalty,
        success_reward=args_cli.success_reward,
        failed_stop_penalty=args_cli.failed_stop_penalty,
        cost_limit=args_cli.safe_cost_limit,
        objective_config=safe_objective_config,
    )
    dataset_writer = None
    dataset_samples = []
    if args_cli.safe_dataset_dir:
        dataset_writer = SafeVLNShardWriter(
            args_cli.safe_dataset_dir,
            objective_config=safe_objective_config,
        )

    step_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    steps_per_image = max(1, round(0.5 / step_dt))
    steps_per_viz = max(1, round(0.1 / step_dt))
    max_steps = round(100 * 0.5 / step_dt)
    if args_cli.max_episode_seconds is not None:
        max_steps = min(max_steps, round(args_cli.max_episode_seconds / step_dt))

    num_steps = 0
    vlm_calls = 0
    terminal = False
    termination_reason = None
    try:
        while simulation_app.is_running() and not terminal and num_steps < max_steps:
            if args_cli.max_vlm_calls is not None and vlm_calls >= args_cli.max_vlm_calls:
                termination_reason = "max_vlm_calls"
                break

            sampled_frames = sample_eight_images(image_observations)
            response = sample_images_and_send_to_vlm(
                image_observations,
                args_cli.vlm_host,
                args_cli.vlm_port,
                instruction.instruction_text,
                request_metadata={
                    "protocol_version": "safe-vln-go2-v2",
                    "mode": "act",
                    "episode_id": str(episode["episode_id"]),
                    "transition_index": len(recorder.transitions),
                    "deterministic": True,
                },
            )
            vlm_calls += 1
            policy_output = normalize_policy_response(response)
            action_text = policy_output["text"]
            recorder.begin(policy_output, _measurement_distance(infos))
            print(
                f"Safe-VLN output: {response}\nAction {policy_output['action_id']}: {action_text}\n"
                f"Command: {policy_output['velocity_command']}, duration: {policy_output['duration']:.2f}s\n",
                flush=True,
            )

            if policy_output["action_id"] == 9:
                env.set_stop_called(True)
                env.measure_manager.update_measures()
                infos["measurements"] = env.measure_manager.get_measurements()
                success = bool(infos["measurements"]["success"])
                recorder.finish(
                    distance_after=_measurement_distance(infos),
                    success=success,
                    failed_stop=not success,
                    terminated=True,
                    termination_reason="success" if success else "failed_stop",
                )
                terminal = True
            else:
                requested_steps = max(1, round(policy_output["duration"] / step_dt))
                command = torch.tensor(policy_output["velocity_command"], device=obs.device)
                env.begin_macro_action(command)
                for _ in range(requested_steps):
                    obs, _, done, infos = env.step(command)
                    num_steps += 1
                    safety = infos.get("safety", {})
                    recorder.record_env_step(safety)

                    if num_steps % steps_per_image == 0:
                        frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
                        image_observations.append(Image.fromarray(frame))
                    if num_steps % steps_per_viz == 0:
                        camera_frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy().copy()
                        viz_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy().copy()
                        add_instruction_on_img(camera_frame, instruction.instruction_text)
                        add_instruction_on_img(viz_frame, action_text)
                        rgb_obses.append(np.concatenate([camera_frame, viz_frame], axis=1))

                    if safety.get("hard_violation", False):
                        termination_reason = safety["termination_reason"]
                    elif done:
                        termination_reason = "environment_termination"
                    elif num_steps >= max_steps:
                        termination_reason = "max_episode_steps"
                    if termination_reason:
                        terminal = True
                        break

                success = bool(infos["measurements"]["success"])
                safety = infos.get("safety", {})
                recorder.finish(
                    distance_after=_measurement_distance(infos),
                    success=success,
                    unsafe_contact=bool(safety.get("unsafe_contact", False)),
                    fall=bool(safety.get("fall", False)),
                    blocked=bool(safety.get("blocked", False)),
                    safety_diagnostics=safety,
                    terminated=terminal,
                    truncated=termination_reason in {"max_episode_steps", "max_vlm_calls"},
                    termination_reason=termination_reason,
                )

            transition = recorder.transitions[-1]
            transition["episode_id"] = str(episode["episode_id"])
            transition["scene_id"] = episode["scene_id"]
            transition["instruction"] = instruction.instruction_text
            transition["observation_key"] = f"episode{episode['episode_id']}/state{transition['index']:06d}"
            transition["next_observation_key"] = (
                None if transition["done"] else f"episode{episode['episode_id']}/state{transition['index'] + 1:06d}"
            )
            if dataset_writer is not None:
                dataset_samples.append((transition["observation_key"], sampled_frames, transition))
    finally:
        if recorder.transitions and not recorder.transitions[-1].get("done") and termination_reason == "max_vlm_calls":
            recorder.transitions[-1]["done"] = True
            recorder.transitions[-1]["truncated"] = True
            recorder.transitions[-1]["termination_reason"] = termination_reason
        recorder.finalize()
        if dataset_writer is not None:
            for sample_key, frames, metadata in dataset_samples:
                dataset_writer.add(sample_key, frames, metadata)
            dataset_writer.close()
            write_episode_summary(args_cli.safe_dataset_dir, recorder.to_dict(infos.get("measurements", {})))
    return infos, rgb_obses, recorder


def save_evaluation_outputs(
    env,
    episode,
    measurements,
    rgb_obses,
    trajectory=None,
    *,
    output_stem=None,
    video_fps=10,
):
    result_dir = f"eval_results/{args_cli.task}_loco_{args_cli.load_run}"
    measurement_dir = os.path.join(result_dir, "measurements")
    video_dir = os.path.join(result_dir, "videos")
    os.makedirs(measurement_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    output_index = (
        str(output_stem)
        if output_stem is not None
        else str(int(episode["episode_id"]) - 1)
    )
    with open(f"{measurement_dir}/{output_index}.json", "w") as file:
        json.dump(measurements, file, indent=4)
    if trajectory is not None:
        trajectory_dir = os.path.join(result_dir, "safe_trajectories")
        os.makedirs(trajectory_dir, exist_ok=True)
        with open(f"{trajectory_dir}/{output_index}.json", "w") as file:
            json.dump(trajectory, file, indent=2)
    writer = imageio.get_writer(
        f"{video_dir}/output_{output_index}.mp4", fps=video_fps
    )
    for frame in rgb_obses:
        writer.append_data(frame.astype(np.uint8))
    writer.close()


def main():
    """IsaacSim Evaluation using NaViLA and trained low-level policy."""

    if args_cli.safe_vln and args_cli.task != "go2_matterport_vision":
        raise ValueError("Safe-VLN supports only --task=go2_matterport_vision")

    # read R2R test episodes
    r2r_data_path = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    all_episodes = read_episodes(r2r_data_path)
    episode = all_episodes[args_cli.episode_idx]
    replay_episode = replay_episode_preflight
    if args_cli.safe_replay:
        print(
            f"[SAFE-REPLAY] loaded replay episode {replay_episode.episode_id} "
            f"with {len(replay_episode.steps)} action points; "
            f"physical Isaac episode={episode['episode_id']}",
            flush=True,
        )

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

    if args_cli.safe_vln:
        # ManagerBasedRLEnv resets terminated environments inside step(). The
        # high-level wrapper must observe and record the terminal Go2 pose first.
        env_cfg.terminations.base_contact = None
        env_cfg.terminations.bad_orientation = None
    if args_cli.safe_replay:
        # Offline replay replaces every high-level RGB observation. Removing
        # both sensors and their observation groups keeps Isaac camera/RTX
        # extensions out of the runtime while preserving LiDAR locomotion.
        env_cfg.scene.rgbd_camera = None
        env_cfg.scene.viz_rgb_camera = None
        env_cfg.observations.camera_obs = None
        env_cfg.observations.viz_camera_obs = None
        if hasattr(env_cfg.observations, "depth_obs"):
            env_cfg.observations.depth_obs = None

    # reset the position and rotation of the robot
    env_cfg = reset_start_pos_rot(env_cfg, args_cli, episode)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli, play=True
    )

    # specify directory for logging experiments
    log_root_path = os.path.join(os.path.dirname(__file__),"../logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(log_root_path, args_cli.load_run)
    print(f"[INFO] Loading run from directory: {log_dir}")

    # update agent config with the one from the loaded run
    log_agent_cfg_file_path = os.path.join(log_dir, "params", "agent.yaml")
    assert os.path.exists(log_agent_cfg_file_path), f"Agent config file not found: {log_agent_cfg_file_path}"
    log_agent_cfg_dict = load_yaml(log_agent_cfg_file_path)
    update_class_from_dict(agent_cfg, log_agent_cfg_dict)

    # specify directory for logging experiments
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    # wrap around environment for rsl-rl
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_go2_inference_checkpoint(
        ppo_runner, resume_path, map_location=agent_cfg.device
    )
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    env = VLNEnvWrapper(
                        env, policy, args_cli.task, episode,
                        high_level_obs_key=None if args_cli.safe_replay else "camera_obs",
                        measure_names=all_measures, safe_vln=args_cli.safe_vln,
                        contact_threshold=args_cli.safe_contact_threshold,
                        orientation_limit=args_cli.safe_orientation_limit,
                        blocked_seconds=args_cli.safe_blocked_seconds,
                        blocked_distance=args_cli.safe_blocked_distance,
                        cost_profile=safe_cost_profile,
                        calibration_file=args_cli.safe_calibration_file)
    
    if not args_cli.safe_replay:
        # set view pos and target
        robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
        cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
        cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
        # set the camera view
        env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)
    
    # step with zeros actions to get the initial frame
    obs, infos = env.reset()

    if args_cli.safe_replay:
        infos, rgb_obses, recorder = run_safe_replay_episode(
            env, obs, infos, replay_episode, episode
        )
        measurements = dict(infos["measurements"])
        measurements.update(recorder.summary(measurements))
        measurements.update(
            {
                "replay_episode_id": str(replay_episode.episode_id),
                "physical_episode_id": str(episode["episode_id"]),
                "observation_alignment": "offline_unpaired",
                "navigation_metrics_aligned": False,
                "reward_source": "graded_oracle_action",
                "policy_tag": args_cli.safe_policy_tag,
                "policy_version": _recorded_policy_version(recorder),
            }
        )
        trajectory = recorder.to_dict(measurements)
        trajectory.update(
            {
                "replay_episode_id": str(replay_episode.episode_id),
                "physical_episode_id": str(episode["episode_id"]),
                "observation_alignment": "offline_unpaired",
                "navigation_metrics_aligned": False,
                "reward_source": "graded_oracle_action",
                "policy_tag": args_cli.safe_policy_tag,
                "policy_version": _recorded_policy_version(recorder),
            }
        )
        replay_output_stem = (
            f"replay_{replay_episode.episode_id}_physical_"
            f"{episode['episode_id']}"
        )
        if args_cli.safe_policy_tag:
            replay_output_stem = (
                f"{replay_output_stem}_{args_cli.safe_policy_tag}"
            )
        save_evaluation_outputs(
            env,
            episode,
            measurements,
            rgb_obses,
            trajectory,
            output_stem=replay_output_stem,
            video_fps=2,
        )
        env.close()
        return

    # NaViLA training gets image observations each 0.5s, visualize every 0.1s
    steps_per_image = 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    steps_per_viz_image = 0.1 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)

    rgb_obs = infos["observations"]["camera_obs"]
    init_frame = rgb_obs[0, :, :, :3].cpu().numpy()
    # init_frame = cv2.rotate(init_frame, cv2.ROTATE_90_CLOCKWISE)
    instruction = InstructionData(**episode["instruction"])
    image_observations = []
    image_observations.append(Image.fromarray(init_frame))

    add_instruction_on_img(init_frame, instruction.instruction_text)
    vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
    # vis_frame = cv2.rotate(vis_frame, cv2.ROTATE_90_CLOCKWISE)
    add_instruction_on_img(vis_frame, "")
    rgb_obses = [np.concatenate([init_frame, vis_frame], axis=1)]

    if args_cli.safe_vln:
        infos, rgb_obses, recorder = run_safe_episode(
            env, obs, infos, image_observations, rgb_obses, instruction, episode
        )
        measurements = dict(infos["measurements"])
        measurements.update(recorder.summary(measurements))
        trajectory = recorder.to_dict(measurements)
        save_evaluation_outputs(env, episode, measurements, rgb_obses, trajectory)
        env.close()
        return

    num_steps = 0
    target_steps = 0
    same_pos_count = 0
    prev_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    max_episode_steps = 100 * 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    # visualizer = define_markers()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if num_steps == target_steps:
                stream_output = sample_images_and_send_to_vlm(image_observations, args_cli.vlm_host, args_cli.vlm_port, instruction.instruction_text)
                vlm_vel_commands, time_to_go = get_vel_command(stream_output)
                env_steps_to_go = int(time_to_go / (
                    env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
                ))
                target_steps = num_steps + env_steps_to_go
                print(f"VLM output: {stream_output}\nVel Command: {vlm_vel_commands}, Env Steps to go: {env_steps_to_go}\n")

        obs, _, done, infos = env.step(torch.tensor(vlm_vel_commands, device = obs.device))

        if done or env.is_stop_called or num_steps > max_episode_steps:
            break

        cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_vel = np.linalg.norm(env.unwrapped.scene["robot"].data.root_vel_w[0].detach().cpu().numpy())
        if np.linalg.norm(cur_pos - prev_pos) < 0.01 and robot_vel < 0.01:
            same_pos_count += 1
        else:
            same_pos_count = 0
        prev_pos = cur_pos

        # Break out of the loop if the robot has stayed in the same location for 500 steps
        if same_pos_count >= 1000:
            print("Robot has stayed in the same location for 1000 steps. Breaking out of the loop.")
            break

        if num_steps % steps_per_image == 0:
            curr_frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            image_observations.append(Image.fromarray(curr_frame))
            curr_frame_copy = curr_frame.copy()
            add_instruction_on_img(curr_frame_copy, instruction.instruction_text)
            
        if num_steps % steps_per_viz_image == 0:
            curr_vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
            add_instruction_on_img(curr_vis_frame, stream_output)
            rgb_obses.append(np.concatenate([curr_frame_copy, curr_vis_frame], axis=1))

        num_steps += 1
        if env_steps_to_go == 0:
            env.set_stop_called(True)

        # if args_cli.visualize_path:
        #     visualizer.visualize(reference_path_isaac)
    measurements = infos["measurements"]

    save_evaluation_outputs(env, episode, measurements, rgb_obses)

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
