# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib
import os
import json
import math
import re
import base64
import io
import socket
import uuid


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
from safe_vln.goal_stop import GOAL_STOP_MODES
from safe_vln.native_history import sample_native_history
from safe_vln.vlnce_dataset import load_isaac_vlnce_payload
from safe_vln.replay import (
    DEFAULT_VLNCE_TRAIN_METADATA,
    load_r2r_replay_episode,
    load_vlnce_episode_metadata,
)
from safe_vln.live_render import (
    DEFAULT_RENDER_PORT,
    DEFAULT_RENDER_TIMEOUT_SECONDS,
    HabitatRenderClient,
    LIVE_SCHEMA_VERSION,
    NAVILA_HISTORY_SAMPLING_POLICY,
    NAVILA_VIDEO_FRAMES,
    isaac_position_to_habitat,
    isaac_wxyz_to_yaw,
    isaac_yaw_to_habitat_yaw,
    navigation_alignment_error,
    sample_navila_history,
)

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
parser.add_argument("--vlm-timeout-seconds", type=float, default=300.0)
parser.add_argument("--max_episode_seconds", type=float, default=None)
parser.add_argument("--max_vlm_calls", type=int, default=None)
parser.add_argument(
    "--r2r-data-path",
    type=str,
    default="isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz",
)
parser.add_argument("--safe-vln", action="store_true", help="Enable Go2 CMDP safety evaluation and trajectory output.")
parser.add_argument("--safe-gamma", type=float, default=0.99)
parser.add_argument("--safe-cost-limit", type=float, default=None)
parser.add_argument("--safe-cost-profile", type=str, default=None)
parser.add_argument("--safe-calibration-file", type=str, default=None)
parser.add_argument("--safe-contact-threshold", type=float, default=1.0)
parser.add_argument("--safe-orientation-limit", type=float, default=0.8)
parser.add_argument("--safe-blocked-seconds", type=float, default=2.0)
parser.add_argument("--safe-blocked-distance", type=float, default=0.10)
parser.add_argument(
    "--safe-turn-min-expected-angle",
    type=float,
    default=0.18,
    help=("Minimum requested yaw (rad) before proportional turn-execution "
        "checking is enabled; the achieved-angle threshold is ratio-based."),
)
parser.add_argument(
    "--safe-turn-min-achieved-ratio",
    type=float,
    default=0.25,
    help="Minimum achieved/requested yaw ratio for a completed turn.",
)
parser.add_argument("--progress-reward-scale", type=float, default=1.0)
parser.add_argument("--success-reward", type=float, default=10.0)
parser.add_argument("--macro-step-penalty", type=float, default=-0.01)
parser.add_argument("--failed-stop-penalty", type=float, default=-1.0)
parser.add_argument("--missed-stop-penalty", type=float, default=-0.5)
parser.add_argument("--missed-stop-patience", type=int, default=3)
parser.add_argument(
    "--goal-stop-mode",
    choices=GOAL_STOP_MODES,
    default="policy",
    help="Raw policy execution, goal shield, or authoritative sensor gate.",
)
parser.add_argument(
    "--collection-policy",
    choices=("vlm", "oracle"),
    default="vlm",
    help="Use the VLM policy or the live dynamic oracle for data collection.",
)
parser.add_argument(
    "--allow-online-oracle",
    action="store_true",
    help="Explicitly allow the experimental live navmesh oracle for ablations.",
)
parser.add_argument("--safe-dataset-dir", type=str, default=None)
parser.add_argument("--online-round", type=int, default=None)
parser.add_argument(
    "--safe-replay",
    action="store_true",
    help="Use offline R2R frames for NaViLA while retaining Go2 physics and safety.",
)
parser.add_argument("--safe-replay-root", type=str, default=None)
parser.add_argument("--safe-replay-annotations", type=str, default=None)
parser.add_argument("--safe-replay-id", type=int, default=None)
parser.add_argument(
    "--safe-replay-vlnce-metadata",
    type=str,
    default=str(DEFAULT_VLNCE_TRAIN_METADATA),
    help=(
        "Original VLN-CE split JSON.GZ used to align the replay ID with its "
        "Matterport scene, start pose, goal, and reference path."
    ),
)
parser.add_argument(
    "--safe-replay-vlnce-gt",
    type=str,
    default=None,
    help="Optional VLN-CE split_gt.json.gz override (defaults beside metadata).",
)
parser.add_argument(
    "--safe-replay-legacy-unpaired",
    action="store_true",
    help=(
        "Retain the old independent --episode_idx physical episode instead "
        "of loading the matching original VLN-CE episode."
    ),
)
parser.add_argument(
    "--safe-policy-tag",
    type=str,
    default=None,
    help="Filesystem-safe policy label added to Safe-Replay IDs and outputs.",
)
parser.add_argument(
    "--safe-live-render",
    action="store_true",
    help=(
        "Render NaViLA RGB observations from the live Go2 pose in a separate "
        "headless Habitat-Sim process."
    ),
)
parser.add_argument("--render-host", type=str, default="127.0.0.1")
parser.add_argument("--render-port", type=int, default=DEFAULT_RENDER_PORT)
parser.add_argument(
    "--render-timeout-seconds",
    type=float,
    default=DEFAULT_RENDER_TIMEOUT_SECONDS,
)
parser.add_argument("--mp3d-scenes-root", type=str, default=None)
parser.add_argument("--vlnce-episode-id", type=int, default=None)
parser.add_argument("--vlnce-metadata", type=str, default=None)
parser.add_argument("--vlnce-gt", type=str, default=None)
parser.add_argument(
    "--dataset-role",
    choices=("train", "eval"),
    default="train",
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
if args_cli.safe_turn_min_expected_angle < 0:
    parser.error("--safe-turn-min-expected-angle must be non-negative")
if not 0 <= args_cli.safe_turn_min_achieved_ratio <= 1:
    parser.error("--safe-turn-min-achieved-ratio must be in [0, 1]")
if args_cli.missed_stop_patience <= 0:
    parser.error("--missed-stop-patience must be positive")
if args_cli.safe_calibration_file and not args_cli.safe_vln:
    parser.error("--safe-calibration-file requires --safe-vln")
if args_cli.safe_policy_tag is not None:
    if not args_cli.safe_vln:
        parser.error("--safe-policy-tag requires --safe-vln")
    if re.fullmatch(r"[A-Za-z0-9._-]+", args_cli.safe_policy_tag) is None:
        parser.error(
            "--safe-policy-tag may contain only letters, digits, '.', '_' and '-'"
        )
if args_cli.safe_live_render:
    if not args_cli.safe_vln:
        parser.error("--safe-live-render requires --safe-vln")
    if args_cli.safe_replay:
        parser.error("--safe-live-render cannot be combined with --safe-replay")
    if getattr(args_cli, "enable_cameras", False):
        parser.error("--safe-live-render cannot be combined with --enable_cameras")
    if not getattr(args_cli, "headless", False):
        parser.error("--safe-live-render requires --headless")
    if args_cli.vlnce_episode_id is None or args_cli.vlnce_metadata is None:
        parser.error(
            "--safe-live-render requires --vlnce-episode-id and --vlnce-metadata"
        )
    if args_cli.mp3d_scenes_root is None:
        parser.error("--safe-live-render requires --mp3d-scenes-root")
    if not 0 < args_cli.render_port < 65536:
        parser.error("--render-port must be in [1, 65535]")
    if args_cli.render_timeout_seconds <= 0:
        parser.error("--render-timeout-seconds must be positive")
    metadata_split = os.path.basename(
        os.path.dirname(os.path.abspath(args_cli.vlnce_metadata))
    )
    if metadata_split == "val_unseen" and args_cli.dataset_role != "eval":
        parser.error("val_unseen live-render data requires --dataset-role=eval")
    if args_cli.collection_policy == "oracle" and args_cli.dataset_role != "train":
        parser.error("--collection-policy=oracle is allowed only for train data")
    if args_cli.collection_policy == "oracle" and not args_cli.allow_online_oracle:
        parser.error(
            "live dynamic Oracle is paused because it is not closed-loop executable; "
            "use offline paired labels or pass --allow-online-oracle for an ablation"
        )
    if (
        args_cli.collection_policy == "oracle"
        and args_cli.goal_stop_mode != "policy"
    ):
        parser.error("--collection-policy=oracle requires --goal-stop-mode=policy")
elif (
    args_cli.vlnce_episode_id is not None
    or args_cli.vlnce_metadata is not None
    or args_cli.vlnce_gt is not None
):
    parser.error("VLN-CE live metadata arguments require --safe-live-render")
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
replay_vlnce_metadata_preflight = None
live_vlnce_metadata_preflight = None
native_vlnce_payload_preflight = None
if args_cli.safe_replay:
    replay_episode_preflight = load_r2r_replay_episode(
        args_cli.safe_replay_root,
        args_cli.safe_replay_id,
        annotations_path=args_cli.safe_replay_annotations,
    )
    if not args_cli.safe_replay_legacy_unpaired:
        replay_vlnce_metadata_preflight = load_vlnce_episode_metadata(
            args_cli.safe_replay_vlnce_metadata,
            args_cli.safe_replay_id,
            gt_path=args_cli.safe_replay_vlnce_gt,
        )
        if (
            replay_episode_preflight.instruction.strip()
            != replay_vlnce_metadata_preflight.instruction.strip()
        ):
            parser.error(
                "Safe-Replay annotation instruction does not match the "
                "original VLN-CE episode metadata"
            )
if args_cli.safe_live_render:
    live_vlnce_metadata_preflight = load_vlnce_episode_metadata(
        args_cli.vlnce_metadata,
        args_cli.vlnce_episode_id,
        gt_path=args_cli.vlnce_gt,
    )
    scenes_root = os.path.abspath(os.path.expanduser(args_cli.mp3d_scenes_root))
    live_scene_glb = os.path.join(
        scenes_root,
        live_vlnce_metadata_preflight.scene_name,
        f"{live_vlnce_metadata_preflight.scene_name}.glb",
    )
    live_scene_navmesh = os.path.splitext(live_scene_glb)[0] + ".navmesh"
    if not os.path.isfile(live_scene_glb):
        parser.error(f"live-render MP3D GLB does not exist: {live_scene_glb}")
    if not os.path.isfile(live_scene_navmesh):
        parser.error(f"live-render MP3D navmesh does not exist: {live_scene_navmesh}")
if args_cli.safe_vln and not args_cli.safe_replay and not args_cli.safe_live_render:
    try:
        native_vlnce_payload_preflight = load_isaac_vlnce_payload(
            args_cli.r2r_data_path,
            expected_role=args_cli.dataset_role,
            expected_scene_count=61 if args_cli.dataset_role == "train" else None,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

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
        "missed_stop_penalty": float(args_cli.missed_stop_penalty),
        "missed_stop_patience": int(args_cli.missed_stop_patience),
    }
)
safe_objective_config.pop("fingerprint", None)
safe_objective_config["fingerprint"] = canonical_fingerprint(
    safe_objective_config
)

# launch omniverse app
# Keep Isaac Lab's headless-rendering experience GPU configuration intact.
# In Isaac Sim 4.1, forcing ``multi_gpu=False`` at the launcher overrides the
# experience's Vulkan/offscreen setup and can segfault while Hydra creates the
# first camera viewport on A800 nodes.  Slurm isolates workers at the node/GPU
# level; the experience itself already caps rendering to one GPU.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw

from safe_vln.actions import (
    action_from_id,
    has_valid_policy_statistics,
    normalize_policy_response,
)
from safe_vln.alignment import requires_strict_start_alignment
from safe_vln.checkpoint import (
    POLICY_INTERFACE_NAVILA_GREEDY,
    load_go2_inference_checkpoint,
)
from safe_vln.dataset import (
    SafeVLNEpisodeWriter,
    SafeVLNShardWriter,
    write_episode_summary,
)
from safe_vln.objective import graded_oracle_reward
from safe_vln.goal_stop import GoalStopController
from safe_vln.trajectory import SafeTrajectoryRecorder
from safe_vln.rpc import raise_for_remote_error, recv_json, send_json

from rsl_rl.runners import OnPolicyRunner

# Import for Gym task-registration side effects after SimulationApp starts.
importlib.import_module("omni.isaac.lab_tasks")
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils import update_class_from_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
import omni.isaac.lab.sim as sim_utils

# Import for local Gym task-registration side effects.
importlib.import_module("omni.isaac.vlnce.config")
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper
from omni.isaac.vlnce.utils.eval_utils import (
    get_vel_command, 
    read_episodes, 
    add_instruction_on_img,
    InstructionData, 
)


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
    scene_path = os.path.join(
        ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd"
    )
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(
            f"Matterport USD scene for episode {episode['episode_id']} "
            f"does not exist: {scene_path}"
        )
    env_cfg.scene.terrain.obj_filepath = scene_path
    
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


def sample_eight_images(image_list):
    if len(image_list) == 0:
        raise ValueError("Did not receive any images")
    if len(image_list) < 8:
        print("Not enough images received; repeating the first valid frame.")
        image_list = image_list.copy()
        first = image_list[0]
        for _ in range(8 - len(image_list)):
            image_list.insert(0, first.copy())
    else:
        image_list = image_list.copy()
    num_images = len(image_list)
    indices = [int(i * (num_images - 1) / 7) for i in range(7)]
    sampled_images = [image_list[i] for i in indices]
    sampled_images.append(image_list[-1])
    return sampled_images


def sample_images_and_send_to_vlm(
    image_list,
    vlm_host,
    vlm_port,
    query,
    request_metadata=None,
    sampled_images=None,
    timeout_seconds=300.0,
):
    """Send exactly one selected eight-frame history to the VLM.

    Callers that already selected a history pass ``sampled_images`` so the
    native-camera path does not pad/resample the same history twice.
    """
    sampled_images = (
        list(sampled_images)
        if sampled_images is not None
        else sample_eight_images(image_list)
    )
    if len(sampled_images) != NAVILA_VIDEO_FRAMES:
        raise ValueError(
            f"VLM requests require exactly {NAVILA_VIDEO_FRAMES} frames, "
            f"got {len(sampled_images)}"
        )

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
        s.settimeout(float(timeout_seconds))
        s.connect((vlm_host, vlm_port))
        send_json(s, request_data)
        return raise_for_remote_error(recv_json(s))


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


def _robot_pose(env):
    robot = env.unwrapped.scene["robot"].data
    position = robot.root_pos_w[0].detach().cpu().tolist()
    rotation = robot.root_quat_w[0].detach().cpu().tolist()
    yaw = isaac_wxyz_to_yaw(rotation)
    return {
        "position": [float(value) for value in position],
        "rotation_wxyz": [float(value) for value in rotation],
        "yaw": float(yaw),
    }


def _native_camera_pose(env):
    """Return the world pose that produced the current native RGB frame."""

    camera = env.unwrapped.scene.sensors["rgbd_camera"].data
    position = camera.pos_w[0].detach().cpu().tolist()
    rotation = camera.quat_w_world[0].detach().cpu().tolist()
    return {
        "position": [float(value) for value in position],
        "rotation_wxyz": [float(value) for value in rotation],
    }


def _sample_live_history(history):
    if not history:
        raise RuntimeError("strict live-render history is empty")
    width, height = history[-1]["image"].size

    def repeat_first_padding():
        return {
            "image": history[0]["image"].copy(),
            "metadata": {
                "history_padding": True,
                "padding_policy": "repeat_first",
                "strict_observation_state_alignment": False,
                "physics_step": None,
            },
        }

    return sample_navila_history(
        history,
        num_frames=NAVILA_VIDEO_FRAMES,
        padding_factory=repeat_first_padding,
    )


def _render_live_frame(
    env,
    render_client,
    vlnce_metadata,
    *,
    physics_step,
    transition_index,
    frame_index,
):
    before = _robot_pose(env)
    habitat_position = isaac_position_to_habitat(before["position"])
    habitat_yaw = isaac_yaw_to_habitat_yaw(before["yaw"])
    expected_start_yaw = isaac_yaw_to_habitat_yaw(
        isaac_wxyz_to_yaw(vlnce_metadata.start_rotation_isaac_wxyz)
    )
    start_position_error, start_yaw_error = navigation_alignment_error(
        before["position"],
        before["yaw"],
        vlnce_metadata.start_position_habitat,
        expected_start_yaw,
    )
    if transition_index == 0 and frame_index == 0 and (
        start_position_error > 0.02
        or start_yaw_error > math.radians(1.0)
    ):
        diagnostic = {
            "event": "episode_start_alignment_failed",
            "episode_id": str(vlnce_metadata.episode_id),
            "scene_id": vlnce_metadata.scene_name,
            "position_error_m": start_position_error,
            "yaw_error_rad": start_yaw_error,
            "isaac_pose": before,
            "expected_habitat_position": list(
                vlnce_metadata.start_position_habitat
            ),
            "expected_habitat_yaw": expected_start_yaw,
        }
        print("[SAFE-LIVE][START-INVALID] " + json.dumps(diagnostic), flush=True)
        raise RuntimeError(
            "strict live-render episode start alignment failed: "
            f"position={start_position_error:.6f}m "
            f"yaw={math.degrees(start_yaw_error):.4f}deg"
        )
    request_id = (
        f"{vlnce_metadata.episode_id}-{transition_index}-{frame_index}-"
        f"{uuid.uuid4().hex}"
    )
    rendered = render_client.render(
        {
            "request_id": request_id,
            "episode_id": str(vlnce_metadata.episode_id),
            "scene_id": vlnce_metadata.scene_name,
            "physics_step": int(physics_step),
            "isaac_pose": before,
            "habitat_position": list(habitat_position),
            "habitat_yaw": habitat_yaw,
            "goal_position": list(vlnce_metadata.goal_position_habitat),
            "success_distance_m": float(vlnce_metadata.goal_radius),
        }
    )
    after = _robot_pose(env)
    response = rendered.metadata
    if response.get("request_id") != request_id:
        raise RuntimeError("Habitat renderer request_id mismatch")
    if str(response.get("episode_id")) != str(vlnce_metadata.episode_id):
        raise RuntimeError("Habitat renderer episode_id mismatch")
    if response.get("scene_id") != vlnce_metadata.scene_name:
        raise RuntimeError("Habitat renderer scene_id mismatch")
    if int(response.get("physics_step", -1)) != int(physics_step):
        raise RuntimeError("Habitat renderer physics_step mismatch")
    returned_success_distance = response.get("success_distance_m")
    if (
        returned_success_distance is None
        or not math.isclose(
            float(returned_success_distance),
            float(vlnce_metadata.goal_radius),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError("Habitat renderer goal radius mismatch")
    applied_pose = response.get("applied_pose")
    if not isinstance(applied_pose, dict):
        raise RuntimeError("Habitat renderer did not report its applied pose")
    position_error, yaw_error = navigation_alignment_error(
        before["position"],
        before["yaw"],
        applied_pose["position"],
        applied_pose["yaw"],
    )
    paused_position_error = float(
        np.linalg.norm(
            np.asarray(before["position"], dtype=np.float64)
            - np.asarray(after["position"], dtype=np.float64)
        )
    )
    paused_yaw_error = abs(
        math.atan2(
            math.sin(before["yaw"] - after["yaw"]),
            math.cos(before["yaw"] - after["yaw"]),
        )
    )
    strict = bool(
        position_error <= 0.02
        and yaw_error <= math.radians(1.0)
        and paused_position_error <= 1e-6
        and paused_yaw_error <= 1e-6
    )
    if not strict:
        raise RuntimeError(
            "strict live-render alignment failed: "
            f"position={position_error:.6f}m yaw={math.degrees(yaw_error):.4f}deg "
            f"paused_position={paused_position_error:.8f}m "
            f"paused_yaw={math.degrees(paused_yaw_error):.6f}deg"
        )
    include_online_oracle = bool(args_cli.allow_online_oracle)
    metadata = {
        "request_id": request_id,
        "physics_step": int(physics_step),
        "isaac_pose": before,
        "habitat_requested_pose": {
            "position": list(habitat_position),
            "yaw": habitat_yaw,
        },
        "habitat_applied_pose": applied_pose,
        "horizontal_position_error_m": position_error,
        "yaw_error_rad": yaw_error,
        "physics_paused_position_error_m": paused_position_error,
        "physics_paused_yaw_error_rad": paused_yaw_error,
        "episode_start_position_error_m": start_position_error,
        "episode_start_yaw_error_rad": start_yaw_error,
        "strict_observation_state_alignment": True,
        "nearest_navmesh_point": response.get("nearest_navmesh_point"),
        "navmesh_snap_distance": response.get("navmesh_snap_distance"),
        "start_snap_valid": bool(response.get("start_snap_valid", False)),
        "goal_snap_valid": bool(response.get("goal_snap_valid", False)),
        "start_snap_distance_m": response.get("start_snap_distance_m"),
        "goal_snap_distance_m": response.get("goal_snap_distance_m"),
        "is_navigable": bool(response.get("is_navigable", False)),
        "geodesic_distance": response.get("geodesic_distance"),
        "navigation_reward_valid": bool(
            response.get("navigation_reward_valid", False)
        ),
        "success_distance_m": float(returned_success_distance),
        "dynamic_oracle_action": (
            response.get("dynamic_oracle_action")
            if include_online_oracle
            else None
        ),
        "oracle_valid": bool(
            include_online_oracle and response.get("oracle_valid", False)
        ),
        "oracle_invalid_reason": (
            response.get("oracle_invalid_reason")
            if include_online_oracle
            else "online_oracle_disabled"
        ),
        "render_latency_ms": response.get("render_latency_ms"),
        "camera": response.get("camera"),
    }
    return {"image": rendered.image, "metadata": metadata}


def _live_output_metadata(vlnce_metadata, *, policy_version):
    return {
        "schema_version": LIVE_SCHEMA_VERSION,
        "vlnce_episode_id": str(vlnce_metadata.episode_id),
        "vlnce_scene_id": vlnce_metadata.scene_name,
        "episode_metadata_aligned": True,
        "observation_alignment": "live_habitat_from_go2_pose",
        "navigation_metrics_aligned": True,
        "strict_observation_state_alignment": True,
        "alignment_scope": "navigation_pose_xy_yaw",
        "camera_pose_policy": "navila_upright_1.25m",
        "history_sampling_policy": NAVILA_HISTORY_SAMPLING_POLICY,
        "history_padding_policy": "repeat_first",
        "history_num_frames": NAVILA_VIDEO_FRAMES,
        "physical_episode_source": "original_vlnce_episode",
        "reward_source": "live_habitat_geodesic_progress",
        "navmesh_source": "locally_recomputed_habitat_sim_0.1.7",
        "dataset_role": args_cli.dataset_role,
        "policy_tag": args_cli.safe_policy_tag,
        "policy_version": policy_version,
        "online_round": args_cli.online_round,
        "goal_radius_m": float(vlnce_metadata.goal_radius),
        "goal_stop_mode": args_cli.goal_stop_mode,
        "collection_policy": args_cli.collection_policy,
        "online_oracle_enabled": bool(args_cli.allow_online_oracle),
        "missed_stop_patience": int(args_cli.missed_stop_patience),
        "vlnce_alignment": vlnce_metadata.alignment_record(),
    }


def _online_recovery_metadata(transitions, oracle_action, policy_action, executed_action):
    """Tag closed-loop errors without adding privileged inputs to the policy."""
    turn_streak = 0
    for transition in reversed(transitions):
        previous = transition.get("executed_action_id")
        if not isinstance(previous, int) or not 0 <= previous <= 5:
            break
        turn_streak += 1
    if 0 <= executed_action <= 5:
        turn_streak += 1
    if not isinstance(oracle_action, int):
        category = "oracle_invalid"
    elif 6 <= oracle_action <= 8 and 0 <= policy_action <= 5:
        category = "forward_after_turn"
    elif policy_action != oracle_action:
        category = "action_mismatch"
    else:
        category = "action_match"
    return {
        "recovery_category": category,
        "consecutive_turn_actions": turn_streak,
        "online_dagger_eligible": bool(
            isinstance(oracle_action, int)
            and 0 <= oracle_action <= 9
            and isinstance(policy_action, int)
            and 0 <= policy_action <= 9
        ),
    }


def run_safe_live_render_episode(
    env,
    obs,
    infos,
    physical_episode,
    vlnce_metadata,
):
    """Run Go2 physics with RGB/geodesics rendered from each actual pose."""
    render_client = HabitatRenderClient(
        args_cli.render_host,
        args_cli.render_port,
        timeout_seconds=args_cli.render_timeout_seconds,
    )
    health = render_client.health()
    print(
        f"[SAFE-LIVE] renderer ready: habitat={health.get('habitat_sim_version')} "
        f"root={health.get('scenes_root')}",
        flush=True,
    )
    episode_id = str(vlnce_metadata.episode_id)
    recorder = SafeTrajectoryRecorder(
        episode_id=episode_id,
        scene_id=physical_episode["scene_id"],
        instruction=vlnce_metadata.instruction,
        gamma=args_cli.safe_gamma,
        progress_scale=args_cli.progress_reward_scale,
        step_penalty=args_cli.macro_step_penalty,
        success_reward=args_cli.success_reward,
        failed_stop_penalty=args_cli.failed_stop_penalty,
        missed_stop_penalty=args_cli.missed_stop_penalty,
        cost_limit=args_cli.safe_cost_limit,
        objective_config=safe_objective_config,
        schema_version=LIVE_SCHEMA_VERSION,
    )
    goal_stop = GoalStopController(
        goal_radius=vlnce_metadata.goal_radius,
        mode=args_cli.goal_stop_mode,
        dataset_role=args_cli.dataset_role,
        missed_stop_patience=args_cli.missed_stop_patience,
    )
    dataset_samples = []
    diagnostic_frames = []
    live_history = []
    step_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    render_interval_steps = max(1, round(0.5 / step_dt))
    max_steps = round(100 * 0.5 / step_dt)
    if args_cli.max_episode_seconds is not None:
        max_steps = min(max_steps, round(args_cli.max_episode_seconds / step_dt))
    num_steps = 0
    vlm_calls = 0
    terminal = False
    termination_reason = None

    initial = _render_live_frame(
        env,
        render_client,
        vlnce_metadata,
        physics_step=num_steps,
        transition_index=0,
        frame_index=0,
    )
    live_history.append(initial)
    initial_geodesic_distance = initial["metadata"].get("geodesic_distance")
    try:
        while simulation_app.is_running() and not terminal and num_steps < max_steps:
            if args_cli.max_vlm_calls is not None and vlm_calls >= args_cli.max_vlm_calls:
                termination_reason = "max_vlm_calls"
                break
            sampled_entries = _sample_live_history(live_history)
            sampled_frames = [entry["image"] for entry in sampled_entries]
            current_frame_metadata = live_history[-1]["metadata"]
            current_distance = current_frame_metadata.get("geodesic_distance")
            reward_valid_before = bool(
                current_frame_metadata["navigation_reward_valid"]
                and current_distance is not None
            )
            distance_before = (
                float(current_distance)
                if reward_valid_before
                else _measurement_distance(infos)
            )
            raw_oracle = current_frame_metadata.get("dynamic_oracle_action")
            # The live shortest-path oracle is privileged.  Keep it entirely
            # out of mainline VLM rollouts unless this process was explicitly
            # launched as the online-Oracle ablation.
            oracle = raw_oracle if args_cli.allow_online_oracle else None
            if args_cli.collection_policy == "oracle":
                if not (
                    current_frame_metadata.get("oracle_valid", False)
                    and isinstance(oracle, dict)
                ):
                    diagnostic = {
                        "event": "invalid_dynamic_oracle",
                        "episode_id": str(vlnce_metadata.episode_id),
                        "scene_id": vlnce_metadata.scene_name,
                        "transition_index": len(recorder.transitions),
                        "oracle_invalid_reason": current_frame_metadata.get(
                            "oracle_invalid_reason"
                        ),
                        "start_snap_valid": current_frame_metadata.get("start_snap_valid"),
                        "goal_snap_valid": current_frame_metadata.get("goal_snap_valid"),
                        "start_snap_distance_m": current_frame_metadata.get("start_snap_distance_m"),
                        "goal_snap_distance_m": current_frame_metadata.get("goal_snap_distance_m"),
                        "geodesic_distance_m": current_frame_metadata.get("geodesic_distance"),
                        "episode_start_position_error_m": current_frame_metadata.get(
                            "episode_start_position_error_m"
                        ),
                        "episode_start_yaw_error_rad": current_frame_metadata.get(
                            "episode_start_yaw_error_rad"
                        ),
                        "isaac_pose": current_frame_metadata.get("isaac_pose"),
                    }
                    print("[SAFE-LIVE][ORACLE-INVALID] " + json.dumps(diagnostic), flush=True)
                    raise RuntimeError(
                        "oracle collection encountered an invalid dynamic oracle: "
                        f"{diagnostic['oracle_invalid_reason']}"
                    )
                response = {
                    "protocol_version": LIVE_SCHEMA_VERSION,
                    "action_id": int(oracle["action_id"]),
                    "action": oracle["text"],
                    "collection_policy": "oracle",
                }
            else:
                response = sample_images_and_send_to_vlm(
                    sampled_frames,
                    args_cli.vlm_host,
                    args_cli.vlm_port,
                    vlnce_metadata.instruction,
                    request_metadata={
                        "protocol_version": LIVE_SCHEMA_VERSION,
                        "mode": "act",
                        "episode_id": episode_id,
                        "scene_id": vlnce_metadata.scene_name,
                        "transition_index": len(recorder.transitions),
                        "strict_observation_state_alignment": True,
                        "history_sampling_policy": (
                            NAVILA_HISTORY_SAMPLING_POLICY
                        ),
                        "deterministic": True,
                    },
                    timeout_seconds=args_cli.vlm_timeout_seconds,
                )
            vlm_calls += 1
            policy_output = normalize_policy_response(response)
            stop_decision = goal_stop.resolve(
                policy_output["action_id"],
                distance_before,
                navigation_reward_valid=reward_valid_before,
                action_probabilities=policy_output.get("action_probabilities"),
            )
            executed_action = action_from_id(stop_decision.executed_action_id)
            recorder.begin(policy_output, distance_before)
            pose_before = _robot_pose(env)
            print(
                f"Safe-Live output: {response}\n"
                f"Policy action {policy_output['action_id']}: "
                f"{policy_output['text']}\n"
                f"Executed action {executed_action.action_id}: "
                f"{executed_action.text}"
                + (
                    " [GOAL SHIELD]\n"
                    if stop_decision.shield_intervened
                    else "\n"
                )
                + f"Command: {list(executed_action.velocity_command)}, "
                f"duration: {executed_action.duration:.2f}s\n",
                flush=True,
            )
            safety = infos.get("safety", {})
            if stop_decision.immediate_terminal:
                success = stop_decision.success
                env.set_stop_called(True)
                termination_reason = stop_decision.termination_reason
                terminal = True
                distance_after = distance_before
                reward_valid_after = reward_valid_before
            else:
                requested_steps = max(
                    1, round(executed_action.duration / step_dt)
                )
                command = torch.tensor(
                    executed_action.velocity_command, device=obs.device
                )
                env.begin_macro_action(
                    command,
                    duration=executed_action.duration,
                    action_id=executed_action.action_id,
                )
                for macro_step in range(1, requested_steps + 1):
                    obs, _, done, infos = env.step(command)
                    num_steps += 1
                    safety = infos.get("safety", {})
                    recorder.record_env_step(safety)
                    should_render = bool(
                        macro_step % render_interval_steps == 0
                        or macro_step == requested_steps
                        or safety.get("hard_violation", False)
                        or done
                    )
                    if should_render:
                        live_history.append(
                            _render_live_frame(
                                env,
                                render_client,
                                vlnce_metadata,
                                physics_step=num_steps,
                                transition_index=len(recorder.transitions),
                                frame_index=len(live_history),
                            )
                        )
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
                    and stop_decision.terminate_after_execution
                ):
                    terminal = True
                    termination_reason = stop_decision.termination_reason
                latest = live_history[-1]["metadata"]
                distance_after_value = latest.get("geodesic_distance")
                reward_valid_after = bool(
                    latest["navigation_reward_valid"]
                    and distance_after_value is not None
                )
                distance_after = (
                    float(distance_after_value)
                    if reward_valid_after
                    else _measurement_distance(infos)
                )
                success = False

            navigation_reward_valid = bool(
                reward_valid_before and reward_valid_after
            )
            recorder.finish(
                distance_after=distance_after,
                reward_override=None if navigation_reward_valid else 0.0,
                reward_components=(
                    None
                    if navigation_reward_valid
                    else {"invalid_navigation_reward": 0.0}
                ),
                success=success,
                failed_stop=stop_decision.failed_stop,
                missed_stop=stop_decision.missed_stop,
                unsafe_contact=bool(safety.get("unsafe_contact", False)),
                fall=bool(safety.get("fall", False)),
                blocked=bool(safety.get("blocked", False)),
                safety_diagnostics=safety,
                terminated=terminal and termination_reason not in {
                    "max_episode_steps",
                    "max_vlm_calls",
                },
                truncated=termination_reason in {
                    "max_episode_steps",
                    "max_vlm_calls",
                },
                termination_reason=termination_reason,
            )
            transition = recorder.transitions[-1]
            policy_statistics_valid = has_valid_policy_statistics(
                policy_output,
                objective_fingerprint=safe_objective_config.get("fingerprint"),
            )
            ppo_eligible = bool(
                args_cli.dataset_role == "train"
                and args_cli.collection_policy == "vlm"
                and navigation_reward_valid
                and policy_statistics_valid
                and not stop_decision.shield_intervened
            )
            actor_distillation_eligible = bool(
                args_cli.dataset_role == "train"
                and args_cli.collection_policy == "vlm"
                and policy_output.get("policy_interface")
                == POLICY_INTERFACE_NAVILA_GREEDY
                and not policy_output.get("invalid_action", False)
            )
            transition.update(
                {
                    "schema_version": LIVE_SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "physical_episode_id": episode_id,
                    "scene_id": physical_episode["scene_id"],
                    "instruction": vlnce_metadata.instruction,
                    "dataset_role": args_cli.dataset_role,
                    "observation_alignment": "live_habitat_from_go2_pose",
                    "alignment_scope": "navigation_pose_xy_yaw",
                    "camera_pose_policy": "navila_upright_1.25m",
                    "history_sampling_policy": (
                        NAVILA_HISTORY_SAMPLING_POLICY
                    ),
                    "history_padding_policy": "repeat_first",
                    "history_num_frames": NAVILA_VIDEO_FRAMES,
                    # The current frame is pose-aligned; repeated history
                    # entries are explicit padding, not synthetic black data.
                    "strict_observation_state_alignment": True,
                    "navigation_reward_valid": navigation_reward_valid,
                    "policy_statistics_valid": policy_statistics_valid,
                    "ppo_eligible": ppo_eligible,
                    "actor_eligible": ppo_eligible,
                    "actor_distillation_eligible": actor_distillation_eligible,
                    "actor_teacher_action_id": stop_decision.policy_action_id,
                    "actor_teacher_interface": policy_output.get(
                        "policy_interface"
                    ),
                    "reward_critic_eligible": navigation_reward_valid,
                    "cost_critic_eligible": True,
                    "oracle_valid": bool(
                        args_cli.allow_online_oracle
                        and current_frame_metadata.get("oracle_valid", False)
                    ),
                    # The live navmesh oracle is diagnostic only.  It is not
                    # an executable teacher because its action is not checked
                    # against the Go2 controller outcome.
                    "oracle_eligible": bool(
                        args_cli.allow_online_oracle
                        and current_frame_metadata.get("oracle_valid", False)
                    ),
                    "dynamic_oracle_action": oracle,
                    "oracle_action_id": (
                        oracle.get("action_id") if isinstance(oracle, dict) else None
                    ),
                    "action_match": bool(
                        isinstance(oracle, dict)
                        and oracle.get("action_id") == policy_output["action_id"]
                    ),
                    "policy_action_id": stop_decision.policy_action_id,
                    "policy_action_text": policy_output["text"],
                    "executed_action_id": stop_decision.executed_action_id,
                    "executed_action_text": executed_action.text,
                    "executed_velocity_command": list(
                        executed_action.velocity_command
                    ),
                    "executed_duration": executed_action.duration,
                    "goal_radius_m": float(vlnce_metadata.goal_radius),
                    "goal_stop_mode": args_cli.goal_stop_mode,
                    "collection_policy": args_cli.collection_policy,
                    "online_oracle_enabled": bool(args_cli.allow_online_oracle),
                    "in_goal_radius": stop_decision.in_goal_radius,
                    "goal_distance_valid": stop_decision.goal_distance_valid,
                    "goal_gate_intervened": stop_decision.shield_intervened,
                    "goal_gate_reason": stop_decision.goal_gate_reason,
                    "missed_stop": stop_decision.missed_stop,
                    "consecutive_missed_stops": (
                        stop_decision.consecutive_missed_stops
                    ),
                    "shield_intervened": stop_decision.shield_intervened,
                    "policy_success": bool(
                        stop_decision.policy_action_id == 9
                        and stop_decision.success
                    ),
                    "system_success": stop_decision.success,
                    "isaac_pose_before": pose_before,
                    "isaac_pose_after": _robot_pose(env),
                    "frame_alignment": [
                        entry["metadata"] for entry in sampled_entries
                    ],
                    "reward_source": "live_habitat_geodesic_progress",
                    "navmesh_source": "locally_recomputed_habitat_sim_0.1.7",
                    "policy_tag": args_cli.safe_policy_tag,
                    "online_round": args_cli.online_round,
                    **_online_recovery_metadata(
                        recorder.transitions[:-1],
                        oracle.get("action_id") if isinstance(oracle, dict) else None,
                        stop_decision.policy_action_id,
                        stop_decision.executed_action_id,
                    ),
                }
            )
            transition["observation_key"] = (
                f"episode{episode_id}/state{transition['index']:06d}"
            )
            transition["next_observation_key"] = (
                None
                if transition["done"]
                else f"episode{episode_id}/state{transition['index'] + 1:06d}"
            )
            dataset_samples.append(
                (transition["observation_key"], sampled_frames, transition)
            )
            diagnostic_frames.append(
                _replay_diagnostic_frame(
                    sampled_frames[-1],
                    instruction=vlnce_metadata.instruction,
                    predicted=policy_output["text"],
                    oracle=(
                        oracle.get("text")
                        if isinstance(oracle, dict)
                        else "invalid"
                    ),
                    reward=transition["reward"],
                    cost=transition["cost"],
                    reason=termination_reason,
                    alignment="strict_live_habitat",
                )
            )
        if (
            termination_reason is None
            and recorder.transitions
            and not terminal
        ):
            termination_reason = "simulation_stopped"
        if termination_reason == "simulation_stopped":
            raise RuntimeError(
                "Isaac simulation stopped before the live-render episode completed"
            )
        if termination_reason == "max_vlm_calls" and recorder.transitions:
            recorder.truncate_last(termination_reason)
        recorder.finalize()
    except Exception:
        # No dataset writer has been opened yet, so the whole episode is
        # automatically quarantined on any renderer/alignment failure.
        raise

    latest_distance = live_history[-1]["metadata"].get("geodesic_distance")
    if latest_distance is not None:
        infos["measurements"]["distance_to_goal"] = float(latest_distance)
    infos["measurements"]["success"] = bool(
        recorder.transitions and recorder.transitions[-1].get("system_success")
    )
    if (
        initial_geodesic_distance is not None
        and infos["measurements"]["success"]
    ):
        path_length = float(infos["measurements"].get("path_length", 0.0) or 0.0)
        initial_distance = float(initial_geodesic_distance)
        infos["measurements"]["spl"] = initial_distance / max(
            initial_distance, path_length, 1e-8
        )
    else:
        infos["measurements"]["spl"] = 0.0

    if args_cli.safe_dataset_dir:
        split = os.path.basename(
            os.path.dirname(os.path.abspath(args_cli.vlnce_metadata))
        )
        with SafeVLNEpisodeWriter(
            args_cli.safe_dataset_dir,
            episode_id,
            dataset_role=args_cli.dataset_role,
            split=split,
            schema_version=LIVE_SCHEMA_VERSION,
            objective_config=safe_objective_config,
        ) as writer:
            for sample_key, frames, metadata in dataset_samples:
                writer.add(sample_key, frames, metadata)
        episode_data = recorder.to_dict(infos.get("measurements", {}))
        episode_data.update(
            _live_output_metadata(
                vlnce_metadata,
                policy_version=_recorded_policy_version(recorder),
            )
        )
        write_episode_summary(args_cli.safe_dataset_dir, episode_data)
    return infos, diagnostic_frames, recorder


def _replay_alignment_flags(vlnce_metadata):
    if vlnce_metadata is None:
        return {
            "episode_metadata_aligned": False,
            "observation_alignment": "offline_unpaired",
            "navigation_metrics_aligned": False,
            "strict_observation_state_alignment": False,
            "physical_episode_source": "legacy_isaac_episode_idx",
        }
    return {
        "episode_metadata_aligned": True,
        "observation_alignment": "offline_reference_same_episode",
        "navigation_metrics_aligned": True,
        "strict_observation_state_alignment": False,
        "physical_episode_source": "original_vlnce_episode",
    }


def _replay_output_metadata(
    replay_episode,
    physical_episode,
    vlnce_metadata,
    *,
    policy_version,
):
    payload = {
        "replay_episode_id": str(replay_episode.episode_id),
        "physical_episode_id": str(physical_episode["episode_id"]),
        "reward_source": "graded_oracle_action",
        "policy_tag": args_cli.safe_policy_tag,
        "policy_version": policy_version,
        **_replay_alignment_flags(vlnce_metadata),
    }
    if vlnce_metadata is not None:
        payload["vlnce_alignment"] = vlnce_metadata.alignment_record()
    return payload


def _replay_diagnostic_frame(
    frame,
    *,
    instruction,
    predicted,
    oracle,
    reward,
    cost,
    reason,
    alignment,
):
    image = frame.convert("RGB").resize((512, 512))
    canvas = Image.new("RGB", (1024, 512), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    lines = [
        f"SAFE-REPLAY ({alignment})",
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


def run_safe_replay_episode(
    env,
    obs,
    infos,
    replay_episode,
    physical_episode,
    vlnce_metadata,
):
    """Run offline R2R observations against live Go2 physics and safety."""
    alignment_flags = _replay_alignment_flags(vlnce_metadata)
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
                    "episode_metadata_aligned": alignment_flags[
                        "episode_metadata_aligned"
                    ],
                    "observation_alignment": alignment_flags[
                        "observation_alignment"
                    ],
                    "transition_index": len(recorder.transitions),
                    "deterministic": True,
                },
                timeout_seconds=args_cli.vlm_timeout_seconds,
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
                env.begin_macro_action(
                    command,
                    duration=policy_output["duration"],
                    action_id=policy_output["action_id"],
                )
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
                    "policy_tag": args_cli.safe_policy_tag,
                    **alignment_flags,
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
                    alignment=alignment_flags["observation_alignment"],
                )
            )
            if dataset_writer is not None:
                dataset_samples.append(
                    (transition["observation_key"], sampled_frames, transition)
                )
    except Exception:
        # Samples are buffered in memory until the whole replay completes.
        # Never publish the prefix of a failed physics/VLM run.
        dataset_samples.clear()
        raise
    else:
        if not recorder.transitions:
            raise RuntimeError("Safe-Replay episode produced no transitions")
        if not recorder.transitions[-1].get("done"):
            if termination_reason == "max_vlm_calls":
                recorder.truncate_last(termination_reason)
            elif not simulation_app.is_running():
                raise RuntimeError(
                    "Isaac simulation stopped before Safe-Replay completed"
                )
            else:
                raise RuntimeError(
                    "Safe-Replay episode ended without a terminal transition"
                )
        recorder.finalize()
        if dataset_writer is not None:
            for sample_key, frames, metadata in dataset_samples:
                dataset_writer.add(sample_key, frames, metadata)
            dataset_writer.close()
            episode_data = recorder.to_dict(infos.get("measurements", {}))
            episode_data.update(
                _replay_output_metadata(
                    replay_episode,
                    physical_episode,
                    vlnce_metadata,
                    policy_version=_recorded_policy_version(recorder),
                )
            )
            write_episode_summary(args_cli.safe_dataset_dir, episode_data)
    return infos, diagnostic_frames, recorder


def run_safe_episode(env, obs, infos, image_observations, rgb_obses, instruction, episode):
    """Run the native Isaac-camera Safe-VLN episode.

    The native-camera path uses the same goal-stop contract as live rendering:
    the policy action is recorded, but sensor-gated fallback actions are what
    the Go2 actually executes.  This keeps premature STOP decisions observable
    without truncating every rollout at the first invalid stop.
    """
    recorder = SafeTrajectoryRecorder(
        episode_id=episode["episode_id"],
        scene_id=episode["scene_id"],
        instruction=instruction.instruction_text,
        gamma=args_cli.safe_gamma,
        progress_scale=args_cli.progress_reward_scale,
        step_penalty=args_cli.macro_step_penalty,
        success_reward=args_cli.success_reward,
        failed_stop_penalty=args_cli.failed_stop_penalty,
        missed_stop_penalty=args_cli.missed_stop_penalty,
        cost_limit=args_cli.safe_cost_limit,
        objective_config=safe_objective_config,
        schema_version=LIVE_SCHEMA_VERSION,
    )
    goal_stop = GoalStopController(
        goal_radius=float(episode["goals"][0]["radius"]),
        mode=args_cli.goal_stop_mode,
        dataset_role=args_cli.dataset_role,
        missed_stop_patience=args_cli.missed_stop_patience,
    )
    write_dataset = bool(args_cli.safe_dataset_dir)
    dataset_samples = []
    native_provenance = dict(
        native_vlnce_payload_preflight["safe_vln_conversion"]
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

            # Select once; send and persist this exact eight-frame history.
            if len(image_observations) < NAVILA_VIDEO_FRAMES:
                print(
                    "Not enough images received; repeating the first valid frame.",
                    flush=True,
                )
            # Native-camera and Habitat live-render inputs must share the
            # official NaViLA full-history sampler before they are combined
            # for Actor training.  The old linear sampler is kept only for
            # legacy replay data and is rejected by the v5 Actor audit.
            sampled_entries = sample_native_history(
                image_observations,
                num_frames=NAVILA_VIDEO_FRAMES,
            )
            sampled_frames = [entry["image"] for entry in sampled_entries]
            response = sample_images_and_send_to_vlm(
                image_observations,
                args_cli.vlm_host,
                args_cli.vlm_port,
                instruction.instruction_text,
                request_metadata={
                    "protocol_version": "safe-vln-go2-v5",
                    "mode": "act",
                    "episode_id": str(episode["episode_id"]),
                    "transition_index": len(recorder.transitions),
                    "deterministic": True,
                },
                sampled_images=sampled_frames,
                timeout_seconds=args_cli.vlm_timeout_seconds,
            )
            vlm_calls += 1
            policy_output = normalize_policy_response(response)
            distance_before = _measurement_distance(infos)
            navigation_reward_valid = bool(
                math.isfinite(distance_before) and distance_before >= 0.0
            )
            stop_decision = goal_stop.resolve(
                policy_output["action_id"],
                distance_before,
                navigation_reward_valid=navigation_reward_valid,
                action_probabilities=policy_output.get("action_probabilities"),
            )
            executed_action = action_from_id(stop_decision.executed_action_id)
            recorder.begin(policy_output, distance_before)
            pose_before = _robot_pose(env)
            print(
                f"Safe-VLN output: {response}\n"
                f"Policy action {policy_output['action_id']}: {policy_output['text']}\n"
                f"Executed action {executed_action.action_id}: {executed_action.text}"
                + (" [GOAL GATE]\n" if stop_decision.shield_intervened else "\n")
                + f"Command: {list(executed_action.velocity_command)}, "
                f"duration: {executed_action.duration:.2f}s\n",
                flush=True,
            )

            safety = infos.get("safety", {})
            if stop_decision.immediate_terminal:
                if stop_decision.success:
                    env.set_stop_called(True)
                    env.measure_manager.update_measures()
                    infos["measurements"] = env.measure_manager.get_measurements()
                success = bool(stop_decision.success)
                termination_reason = stop_decision.termination_reason or (
                    "success" if success else "failed_stop"
                )
                terminal = True
            else:
                requested_steps = max(1, round(executed_action.duration / step_dt))
                command = torch.tensor(
                    executed_action.velocity_command, device=obs.device
                )
                env.begin_macro_action(
                    command,
                    duration=executed_action.duration,
                    action_id=executed_action.action_id,
                )
                for _ in range(requested_steps):
                    obs, _, done, infos = env.step(command)
                    num_steps += 1
                    safety = infos.get("safety", {})
                    recorder.record_env_step(safety)

                    if num_steps % steps_per_image == 0:
                        frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
                        image_observations.append(
                            {
                                "image": Image.fromarray(frame),
                                "metadata": {
                                    "frame_index": len(image_observations),
                                    "physics_step": num_steps,
                                    "isaac_pose": _robot_pose(env),
                                    "camera_pose": _native_camera_pose(env),
                                    "history_padding": False,
                                    "strict_observation_state_alignment": True,
                                },
                            }
                        )
                    if num_steps % steps_per_viz == 0:
                        camera_frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy().copy()
                        viz_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy().copy()
                        add_instruction_on_img(camera_frame, instruction.instruction_text)
                        add_instruction_on_img(viz_frame, executed_action.text)
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

                if not terminal and stop_decision.terminate_after_execution:
                    terminal = True
                    termination_reason = stop_decision.termination_reason
                success = bool(infos["measurements"]["success"])
                safety = infos.get("safety", {})
                recorder.finish(
                    distance_after=_measurement_distance(infos),
                    success=success,
                    missed_stop=bool(stop_decision.missed_stop),
                    unsafe_contact=bool(safety.get("unsafe_contact", False)),
                    fall=bool(safety.get("fall", False)),
                    blocked=bool(safety.get("blocked", False)),
                    safety_diagnostics=safety,
                    terminated=terminal,
                    truncated=termination_reason in {"max_episode_steps", "max_vlm_calls"},
                    termination_reason=termination_reason,
                )

            if stop_decision.immediate_terminal:
                recorder.finish(
                    distance_after=_measurement_distance(infos),
                    success=success,
                    failed_stop=bool(stop_decision.failed_stop),
                    missed_stop=bool(stop_decision.missed_stop),
                    unsafe_contact=bool(safety.get("unsafe_contact", False)),
                    fall=bool(safety.get("fall", False)),
                    blocked=bool(safety.get("blocked", False)),
                    safety_diagnostics=safety,
                    terminated=True,
                    termination_reason=termination_reason,
                )

            transition = recorder.transitions[-1]
            policy_statistics_valid = has_valid_policy_statistics(
                policy_output,
                objective_fingerprint=safe_objective_config.get("fingerprint"),
            )
            ppo_eligible = bool(
                args_cli.dataset_role == "train"
                and args_cli.collection_policy == "vlm"
                and navigation_reward_valid
                and policy_statistics_valid
                and not stop_decision.shield_intervened
            )
            actor_distillation_eligible = bool(
                args_cli.dataset_role == "train"
                and args_cli.collection_policy == "vlm"
                and policy_output.get("policy_interface")
                == POLICY_INTERFACE_NAVILA_GREEDY
                and not policy_output.get("invalid_action", False)
            )
            transition.update(
                {
                    "schema_version": LIVE_SCHEMA_VERSION,
                    "dataset_role": args_cli.dataset_role,
                    "vlnce_source_split": native_provenance["source_split"],
                    "vlnce_source_dataset_role": native_provenance[
                        "dataset_role"
                    ],
                    "vlnce_coordinate_system": native_provenance[
                        "coordinate_system"
                    ],
                    "vlnce_source_metadata_sha256": native_provenance.get(
                        "source_metadata_sha256"
                    ),
                    "vlnce_source_gt_sha256": native_provenance.get(
                        "source_gt_sha256"
                    ),
                    "observation_alignment": "native_isaac_camera",
                    "alignment_scope": "go2_native_camera_pose",
                    "camera_pose_policy": "isaac_native_attached_camera",
                    "history_sampling_policy": NAVILA_HISTORY_SAMPLING_POLICY,
                    "history_padding_policy": "repeat_first",
                    "history_num_frames": NAVILA_VIDEO_FRAMES,
                    "history_padding_count": sum(
                        bool(entry["metadata"].get("history_padding", False))
                        for entry in sampled_entries
                    ),
                    "strict_observation_state_alignment": True,
                    "frame_alignment": [
                        entry["metadata"] for entry in sampled_entries
                    ],
                    "isaac_pose_before": pose_before,
                    "isaac_pose_after": _robot_pose(env),
                    "episode_id": str(episode["episode_id"]),
                    "scene_id": episode["scene_id"],
                    "instruction": instruction.instruction_text,
                    "collection_policy": args_cli.collection_policy,
                    "policy_tag": args_cli.safe_policy_tag,
                    "online_round": args_cli.online_round,
                    "online_oracle_enabled": False,
                    "oracle_valid": False,
                    "oracle_eligible": False,
                    "oracle_action_id": None,
                    "policy_action_id": stop_decision.policy_action_id,
                    "policy_action_text": policy_output["text"],
                    "executed_action_id": stop_decision.executed_action_id,
                    "executed_action_text": executed_action.text,
                    "executed_velocity_command": list(executed_action.velocity_command),
                    "executed_duration": executed_action.duration,
                    "goal_radius_m": float(episode["goals"][0]["radius"]),
                    "goal_stop_mode": args_cli.goal_stop_mode,
                    "in_goal_radius": stop_decision.in_goal_radius,
                    "goal_distance_valid": stop_decision.goal_distance_valid,
                    "goal_gate_intervened": stop_decision.shield_intervened,
                    "shield_intervened": stop_decision.shield_intervened,
                    "goal_gate_reason": stop_decision.goal_gate_reason,
                    "missed_stop": stop_decision.missed_stop,
                    "consecutive_missed_stops": stop_decision.consecutive_missed_stops,
                    "policy_success": bool(
                        stop_decision.policy_action_id == 9 and stop_decision.success
                    ),
                    "system_success": bool(success),
                    "navigation_reward_valid": navigation_reward_valid,
                    "policy_statistics_valid": policy_statistics_valid,
                    "ppo_eligible": ppo_eligible,
                    "actor_eligible": ppo_eligible,
                    "actor_distillation_eligible": actor_distillation_eligible,
                    "actor_teacher_action_id": stop_decision.policy_action_id,
                    "actor_teacher_interface": policy_output.get(
                        "policy_interface"
                    ),
                    "reward_critic_eligible": navigation_reward_valid,
                    "cost_critic_eligible": True,
                }
            )
            transition["observation_key"] = (
                f"episode{episode['episode_id']}/state{transition['index']:06d}"
            )
            transition["next_observation_key"] = (
                None
                if transition["done"]
                else f"episode{episode['episode_id']}/state{transition['index'] + 1:06d}"
            )
            if write_dataset:
                dataset_samples.append(
                    (transition["observation_key"], sampled_frames, transition)
                )
    except Exception:
        # Never publish a partial episode when rendering, VLM inference, or
        # physics raises.  The old flat writer committed whatever had been
        # buffered from the failed run and made the dataset look complete.
        dataset_samples.clear()
        raise
    if recorder.transitions and not recorder.transitions[-1].get("done"):
        if termination_reason == "max_vlm_calls":
            recorder.truncate_last(termination_reason)
        elif not simulation_app.is_running():
            raise RuntimeError(
                "Isaac simulation stopped before the native-camera episode completed"
            )
        else:
            raise RuntimeError(
                "native-camera episode ended without a terminal transition"
            )
    recorder.finalize()
    if write_dataset:
        if not dataset_samples:
            raise RuntimeError(
                "native-camera episode produced no aligned transitions"
            )
        with SafeVLNEpisodeWriter(
            args_cli.safe_dataset_dir,
            episode["episode_id"],
            dataset_role=args_cli.dataset_role,
            split=args_cli.dataset_role,
            schema_version=LIVE_SCHEMA_VERSION,
            objective_config=safe_objective_config,
        ) as writer:
            for sample_key, frames, metadata in dataset_samples:
                writer.add(sample_key, frames, metadata)
        write_episode_summary(
            args_cli.safe_dataset_dir,
            recorder.to_dict(infos.get("measurements", {})),
        )
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

    replay_episode = replay_episode_preflight
    replay_vlnce_metadata = replay_vlnce_metadata_preflight
    live_vlnce_metadata = live_vlnce_metadata_preflight
    if args_cli.safe_replay:
        if replay_vlnce_metadata is not None:
            episode = replay_vlnce_metadata.to_isaac_episode()
            physical_description = (
                f"matching VLN-CE episode={episode['episode_id']} "
                f"scene={replay_vlnce_metadata.scene_name} "
                f"start={episode['start_position']} "
                f"goal={episode['goals'][0]['position']}"
            )
        else:
            r2r_data_path = os.path.join(
                ASSETS_DIR, "vln_ce_isaac_v1.json.gz"
            )
            all_episodes = read_episodes(r2r_data_path)
            episode = all_episodes[args_cli.episode_idx]
            physical_description = (
                f"legacy physical Isaac episode={episode['episode_id']}"
            )
        print(
            f"[SAFE-REPLAY] loaded replay episode {replay_episode.episode_id} "
            f"with {len(replay_episode.steps)} action points; "
            f"{physical_description}",
            flush=True,
        )
    elif args_cli.safe_live_render:
        episode = live_vlnce_metadata.to_isaac_episode()
        print(
            f"[SAFE-LIVE] loaded VLN-CE episode={episode['episode_id']} "
            f"scene={live_vlnce_metadata.scene_name} "
            f"start={episode['start_position']} "
            f"goal={episode['goals'][0]['position']}",
            flush=True,
        )
    else:
        if args_cli.safe_vln:
            all_episodes = native_vlnce_payload_preflight["episodes"]
        else:
            all_episodes = read_episodes(args_cli.r2r_data_path)
        episode = all_episodes[args_cli.episode_idx]

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

    if args_cli.safe_vln:
        # ManagerBasedRLEnv resets terminated environments inside step(). The
        # high-level wrapper must observe and record the terminal Go2 pose first.
        env_cfg.terminations.base_contact = None
        env_cfg.terminations.bad_orientation = None
        # The stock Go2 locomotion config disables PhysX contact processing for
        # speed.  That makes a zero collision rate meaningless for Safe-VLN.
        env_cfg.sim.disable_contact_processing = False
    if args_cli.safe_replay or args_cli.safe_live_render:
        # Offline replay and live Habitat rendering replace every high-level
        # Isaac RGB observation. Removing both sensors keeps RTX/viewport
        # extensions out of the A800 runtime while preserving LiDAR locomotion.
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
                        high_level_obs_key=(
                            None
                            if args_cli.safe_replay or args_cli.safe_live_render
                            else "camera_obs"
                        ),
                        measure_names=all_measures, safe_vln=args_cli.safe_vln,
                        contact_threshold=args_cli.safe_contact_threshold,
                        orientation_limit=args_cli.safe_orientation_limit,
                        blocked_seconds=args_cli.safe_blocked_seconds,
                        blocked_distance=args_cli.safe_blocked_distance,
                        turn_min_expected_angle=args_cli.safe_turn_min_expected_angle,
                        turn_min_achieved_ratio=args_cli.safe_turn_min_achieved_ratio,
                        cost_profile=safe_cost_profile,
                        calibration_file=args_cli.safe_calibration_file,
                        strict_start_alignment=requires_strict_start_alignment(
                            safe_vln=args_cli.safe_vln,
                            safe_replay=args_cli.safe_replay,
                        ))
    
    if not args_cli.safe_replay and not args_cli.safe_live_render:
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
            env,
            obs,
            infos,
            replay_episode,
            episode,
            replay_vlnce_metadata,
        )
        measurements = dict(infos["measurements"])
        measurements.update(recorder.summary(measurements))
        measurements.update(
            _replay_output_metadata(
                replay_episode,
                episode,
                replay_vlnce_metadata,
                policy_version=_recorded_policy_version(recorder),
            )
        )
        trajectory = recorder.to_dict(measurements)
        trajectory.update(
            _replay_output_metadata(
                replay_episode,
                episode,
                replay_vlnce_metadata,
                policy_version=_recorded_policy_version(recorder),
            )
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
    if args_cli.safe_live_render:
        infos, rgb_obses, recorder = run_safe_live_render_episode(
            env,
            obs,
            infos,
            episode,
            live_vlnce_metadata,
        )
        measurements = dict(infos["measurements"])
        measurements.update(recorder.summary(measurements))
        measurements.update(
            _live_output_metadata(
                live_vlnce_metadata,
                policy_version=_recorded_policy_version(recorder),
            )
        )
        trajectory = recorder.to_dict(measurements)
        trajectory.update(
            _live_output_metadata(
                live_vlnce_metadata,
                policy_version=_recorded_policy_version(recorder),
            )
        )
        output_stem = f"live_{live_vlnce_metadata.episode_id}"
        if args_cli.safe_policy_tag:
            output_stem = f"{output_stem}_{args_cli.safe_policy_tag}"
        save_evaluation_outputs(
            env,
            episode,
            measurements,
            rgb_obses,
            trajectory,
            output_stem=output_stem,
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
    if args_cli.safe_vln:
        image_observations.append(
            {
                "image": Image.fromarray(init_frame),
                "metadata": {
                    "frame_index": 0,
                    "physics_step": 0,
                    "isaac_pose": _robot_pose(env),
                    "camera_pose": _native_camera_pose(env),
                    "history_padding": False,
                    "strict_observation_state_alignment": True,
                },
            }
        )
    else:
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
                stream_output = sample_images_and_send_to_vlm(
                    image_observations,
                    args_cli.vlm_host,
                    args_cli.vlm_port,
                    instruction.instruction_text,
                    timeout_seconds=args_cli.vlm_timeout_seconds,
                )
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
