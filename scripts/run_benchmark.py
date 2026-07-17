import os
import argparse
import shlex
import subprocess
import sys
import time
import gzip
import json


def read_episodes(file_path):
    """Read episode list from the compressed VLN-CE Isaac dataset."""
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data["episodes"]


def clean_kit_state():
    """Remove stale kit cache/lock files left by a crashed Isaac Sim process."""
    paths = [
        os.path.expanduser("~/.cache/omni"),
        os.path.expanduser("~/.nv/Omniverse"),
    ]
    tmp_kit = os.environ.get("OMNI_USER_CACHE_DIR", "")
    if tmp_kit:
        paths.append(tmp_kit)
    for p in paths:
        if p and os.path.isdir(p):
            # Only remove lock/cache files, not the whole directory
            for root, _dirs, files in os.walk(p, topdown=False):
                for name in files:
                    if name.endswith(".lock") or name.endswith(".lck"):
                        os.remove(os.path.join(root, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2r-data-path", type=str,
                        default="isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz")
    parser.add_argument("--navila-model-path", type=str,
                        default="/home/zhaojing/mnt/legged_nav/NaVILA/NaVILA-llama3-8B-8f-scanqa-rxr")
    parser.add_argument("--task", type=str, default="go2_matterport_vision")
    parser.add_argument("--low_level_policy_dir", type=str, default="2024-09-25_23-22-02")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Exclusive episode index. Defaults to the end of the dataset.")
    parser.add_argument("--vlm_host", type=str, default="localhost")
    parser.add_argument("--vlm_port", type=int, default=54321)
    parser.add_argument("--max_episode_seconds", type=float, default=None)
    parser.add_argument("--max_vlm_calls", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop the benchmark when one episode subprocess exits non-zero.")
    parser.add_argument("--clean-kit-state", action="store_true",
                        help="Clean stale kit cache/lock files between episodes.")
    args = parser.parse_args()

    eval_args = [f"--task={args.task}", "--num_envs=1",
                 f"--load_run={args.low_level_policy_dir}",
                 "--headless", "--enable_cameras",
                 f"--vlm_host={args.vlm_host}",
                 f"--vlm_port={args.vlm_port}",
                 ]
    if args.max_episode_seconds is not None:
        eval_args.append(f"--max_episode_seconds={args.max_episode_seconds}")
    if args.max_vlm_calls is not None:
        eval_args.append(f"--max_vlm_calls={args.max_vlm_calls}")

    if args.task == "go2_matterport_vision":
        eval_args.append("--history_length=9")

    episodes = read_episodes(args.r2r_data_path)
    end_idx = len(episodes) if args.end_idx is None else min(args.end_idx, len(episodes))

    if args.start_idx < 0 or args.start_idx >= len(episodes):
        raise ValueError(f"start_idx={args.start_idx} is outside [0, {len(episodes) - 1}]")
    if end_idx <= args.start_idx:
        raise ValueError(f"end_idx={end_idx} must be greater than start_idx={args.start_idx}")

    for i in range(args.start_idx, end_idx):
        episode = episodes[i]
        print("Episode id: ", episode['episode_id'])

        msg = f"\n======================= Running Evaluation of Episode {i} ======================="
        msg += f"\nScene: {episodes[i]['scene_id']}"
        msg += f"\nStart Position: {episodes[i]['start_position']}"
        msg += f"\nStart Rotation: {episodes[i]['start_rotation']}"
        msg += f"\nInstruction: {episodes[i]['instruction']['instruction_text']}\n"
        print(msg)

        if args.clean_kit_state:
            clean_kit_state()

        episode_eval_args = eval_args + [f"--episode_idx={i}"]
        cmd = [sys.executable, 'scripts/navila_eval.py'] + episode_eval_args
        print(f"Running: {shlex.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + "/..")

        if completed.returncode != 0:
            print(f"Episode {i} failed with return code {completed.returncode}", flush=True)
            print(f"Failed command: {shlex.join(cmd)}", flush=True)
            if args.stop_on_error:
                sys.exit(completed.returncode)

        # Brief pause to let GPU memory / kit locks settle between episodes.
        time.sleep(2.0)
