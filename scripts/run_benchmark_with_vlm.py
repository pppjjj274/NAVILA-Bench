"""Run the NaVILA benchmark with an embedded VLM socket server.

This merges the original two-terminal workflow:

1. ``scripts/vlm_server.py`` loads the NaVILA VLM and serves TCP requests.
2. ``scripts/run_benchmark.py`` launches ``scripts/navila_eval.py`` per episode.

The VLM still uses the same local length-prefixed JSON protocol, but the server
is started in a daemon thread so a batch job only needs one Python entry point.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import signal
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NAVILA_EVAL_SCRIPT = REPO_ROOT / "scripts" / "navila_eval.py"

torch = None
Image = None
tqdm = None
AutoConfig = None
AutoTokenizer = None
IMAGE_TOKEN_INDEX = None
SeparatorStyle = None
conv_templates = None
KeywordsStoppingCriteria = None
get_model_name_from_path = None
process_image = None
tokenizer_image_token = None
load_pretrained_model = None


def load_vlm_dependencies() -> None:
    """Import heavy VLM dependencies only when the server is actually started."""
    global torch
    global Image
    global tqdm
    global AutoConfig
    global AutoTokenizer
    global IMAGE_TOKEN_INDEX
    global SeparatorStyle
    global conv_templates
    global KeywordsStoppingCriteria
    global get_model_name_from_path
    global process_image
    global tokenizer_image_token
    global load_pretrained_model

    if torch is not None:
        return

    import torch as torch_module
    from PIL import Image as image_module
    from tqdm import tqdm as tqdm_function
    from transformers import AutoConfig as auto_config_class
    from transformers import AutoTokenizer as auto_tokenizer_class

    from llava.constants import IMAGE_TOKEN_INDEX as image_token_index
    from llava.conversation import SeparatorStyle as separator_style_class
    from llava.conversation import conv_templates as conversation_templates
    from llava.mm_utils import KeywordsStoppingCriteria as stopping_criteria_class
    from llava.mm_utils import get_model_name_from_path as model_name_from_path
    from llava.mm_utils import process_image as process_single_image
    from llava.mm_utils import tokenizer_image_token as tokenize_image_token
    from llava.model.builder import load_pretrained_model as load_model

    torch = torch_module
    Image = image_module
    tqdm = tqdm_function
    AutoConfig = auto_config_class
    AutoTokenizer = auto_tokenizer_class
    IMAGE_TOKEN_INDEX = image_token_index
    SeparatorStyle = separator_style_class
    conv_templates = conversation_templates
    KeywordsStoppingCriteria = stopping_criteria_class
    get_model_name_from_path = model_name_from_path
    process_image = process_single_image
    tokenizer_image_token = tokenize_image_token
    load_pretrained_model = load_model


def resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


class ExternalVLMProcess:
    """Handle an externally launched VLM server subprocess."""

    def __init__(self, process: subprocess.Popen, log_file):
        self.process = process
        self.log_file = log_file

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10.0)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10.0)

        if self.log_file is not None:
            self.log_file.close()


@dataclass(frozen=True)
class VLMServerArgs:
    model_path: str
    precision: str = "W16A16"
    conv_mode: str = "llama_3"
    device: str = "cuda"
    num_video_frames: int = 8


class VLMServer:
    """Small TCP server that wraps NaVILA inference for IsaacLab evaluation."""

    def __init__(self, args: VLMServerArgs):
        load_vlm_dependencies()
        self.args = args
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self.setup()

    def setup(self) -> None:
        """Load tokenizer/model once, then keep them resident for all episodes."""
        self._disable_initializers()
        self._initialize_tokenizer_and_model()

        if self.args.precision == "W16A16":
            self._load_checkpoint_w16a16()
        else:
            raise ValueError(f"Precision {self.args.precision} not supported")

    @staticmethod
    def _disable_initializers() -> None:
        # Loading checkpoint weights does not need random parameter
        # initialization. Disabling it saves startup time and avoids a memory
        # spike while the large VLM is being constructed.
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None

    def _initialize_tokenizer_and_model(self) -> None:
        # The checkpoint stores the tokenizer under the llm/ subdirectory while
        # the full multimodal config lives at model_path.
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.args.model_path, "llm"), use_fast=False
        )
        _ = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)

    def _load_checkpoint_w16a16(self) -> None:
        """Load the full-precision/fp16 NaVILA checkpoint used by the paper."""
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint shards")
        for _ in pbar:
            model_name = get_model_name_from_path(self.args.model_path)
            tokenizer, model, image_processor, _context_len = load_pretrained_model(
                self.args.model_path, model_name, None
            )
            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor

        self.model = self.model.to(self.args.device)

    def stop(self) -> None:
        """Ask the server loop to stop and unblock accept() if needed."""
        self._stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass

    def start_server(
        self,
        host: str = "localhost",
        port: int = 54321,
        ready_event: threading.Event | None = None,
    ) -> None:
        """Serve one request per TCP connection using a length-prefixed JSON protocol."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            self._server_socket = server_socket
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)
            print(f"VLM Server listening on {host}:{port}", flush=True)

            if ready_event is not None:
                ready_event.set()

            while not self._stop_event.is_set():
                try:
                    conn, addr = server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise

                with conn:
                    self._handle_connection(conn, addr)

    def _handle_connection(self, conn: socket.socket, addr) -> None:
        try:
            size_data = recv_exact(conn, 8)
            size = int.from_bytes(size_data, "big")
            data = recv_exact(conn, size)

            request = json.loads(data.decode())
            images = request["images"]
            query = request["query"]

            response = self.process_request(images, query)
            response_bytes = json.dumps(response).encode()

            try:
                conn.sendall(len(response_bytes).to_bytes(8, "big"))
                conn.sendall(response_bytes)
            except BrokenPipeError:
                print(f"Client {addr} disconnected while sending response", flush=True)
            except Exception as exc:
                print(f"Error sending response to {addr}: {exc}", flush=True)
        except Exception as exc:
            print(f"Error processing request from {addr}: {exc}", flush=True)

    def process_request(self, images, query: str) -> str:
        """Run NaVILA on sampled history/current frames and one instruction."""
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.args.device, dtype=torch.float16)

        conv = conv_templates[self.args.conv_mode].copy()
        image_token = "<image>\n"
        qs = (
            "Imagine you are a robot programmed for navigation tasks. You have "
            f"been given a video of historical observations "
            f"{image_token * (self.args.num_video_frames - 1)}, and current "
            f'observation <image>\n. Your assigned task is: "{query}" '
            "Analyze this series of images to decide your next action, which "
            "could be turning left or right by a specific degree, moving "
            "forward a certain distance, or stop if the task is completed."
        )
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.args.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            start_time = time.time()
            output_ids = self.model.generate(
                input_ids,
                images=[image_tensor],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            generation_time = time.time() - start_time
            print(f"Model generation took {generation_time:.2f} seconds", flush=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return outputs.strip()


def recv_exact(conn: socket.socket, num_bytes: int) -> bytes:
    """Receive exactly num_bytes bytes or raise if the peer disconnects."""
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed before the full payload was received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def process_images(images, image_processor, model_cfg):
    """Process a list of images, either PIL Images or base64 strings."""
    if not images:
        raise ValueError("At least one image is required for VLM inference")

    model_cfg.image_processor = image_processor
    processed_images = []

    for image in images:
        if isinstance(image, str):
            try:
                image = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
            except Exception as exc:
                print(f"Error decoding base64 image: {exc}", flush=True)
                image = Image.new("RGB", (224, 224), (0, 0, 0))

        processed_images.append(process_image(image, model_cfg, None))

    if all(x.shape == processed_images[0].shape for x in processed_images):
        return torch.stack(processed_images, dim=0)
    return processed_images


def read_episodes(file_path: str):
    """Read episode list from the compressed VLN-CE Isaac dataset."""
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    return data["episodes"]


def build_eval_args(args: argparse.Namespace) -> list[str]:
    eval_args = [
        f"--task={args.task}",
        "--num_envs=1",
        f"--load_run={args.low_level_policy_dir}",
        "--headless",
        "--enable_cameras",
        f"--vlm_host={args.vlm_host}",
        f"--vlm_port={args.vlm_port}",
    ]
    if args.isaac_device_id is not None:
        eval_args.append(f"--device_id={args.isaac_device_id}")
    if args.max_episode_seconds is not None:
        eval_args.append(f"--max_episode_seconds={args.max_episode_seconds}")
    if args.max_vlm_calls is not None:
        eval_args.append(f"--max_vlm_calls={args.max_vlm_calls}")

    if args.task == "go2_matterport_vision":
        # The released Go2 policy was trained with 9 proprioception history frames.
        eval_args.append("--history_length=9")

    return eval_args


def wait_for_vlm_port(
    host: str,
    port: int,
    timeout: float,
    process: subprocess.Popen | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"VLM server exited early with code {process.returncode}")

        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)

    raise TimeoutError(
        f"VLM server on {host}:{port} was not ready within {timeout} seconds"
    ) from last_error


def start_vlm_server(args: argparse.Namespace) -> tuple[object | None, threading.Thread | None]:
    if args.no_start_vlm_server:
        print(
            "Using an already-started VLM server; skipping TCP readiness probe "
            "because vlm_server.py treats empty probe connections as requests.",
            flush=True,
        )
        return None, None

    if args.vlm_launch_command:
        log_path = Path(args.vlm_log_file or f"vlm_server_{os.getpid()}.log")
        if not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            args.vlm_launch_command,
            cwd=REPO_ROOT,
            executable="/bin/bash",
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle = ExternalVLMProcess(process, log_file)
        try:
            wait_for_vlm_port(
                args.vlm_host,
                args.vlm_port,
                args.server_startup_timeout,
                process=process,
            )
        except Exception:
            handle.stop()
            raise
        return handle, None

    server_args = VLMServerArgs(
        model_path=resolve_repo_path(args.navila_model_path),
        precision=args.vlm_precision,
        conv_mode=args.conv_mode,
        device=args.vlm_device,
        num_video_frames=args.num_video_frames,
    )
    server = VLMServer(server_args)
    ready_event = threading.Event()
    startup_state: dict[str, BaseException | None] = {"error": None}

    def serve() -> None:
        try:
            server.start_server(args.vlm_host, args.vlm_port, ready_event=ready_event)
        except BaseException as exc:
            startup_state["error"] = exc
            ready_event.set()
            raise

    thread = threading.Thread(target=serve, name="embedded-vlm-server", daemon=True)
    thread.start()

    if not ready_event.wait(timeout=args.server_startup_timeout):
        server.stop()
        raise TimeoutError(
            f"VLM server did not start within {args.server_startup_timeout} seconds"
        )
    if startup_state["error"] is not None:
        raise RuntimeError("VLM server failed to start") from startup_state["error"]

    return server, thread


def run_benchmark(args: argparse.Namespace) -> int:
    eval_args = build_eval_args(args)
    episodes = read_episodes(resolve_repo_path(args.r2r_data_path))
    end_idx = len(episodes) if args.end_idx is None else min(args.end_idx, len(episodes))

    if args.start_idx < 0 or args.start_idx >= len(episodes):
        raise ValueError(f"start_idx={args.start_idx} is outside [0, {len(episodes) - 1}]")
    if end_idx <= args.start_idx:
        raise ValueError(f"end_idx={end_idx} must be greater than start_idx={args.start_idx}")

    for i in range(args.start_idx, end_idx):
        episode = episodes[i]
        print("Episode id: ", episode["episode_id"], flush=True)

        msg = f"\n======================= Running Evaluation of Episode {i} ======================="
        msg += f"\nScene: {episode['scene_id']}"
        msg += f"\nStart Position: {episode['start_position']}"
        msg += f"\nStart Rotation: {episode['start_rotation']}"
        msg += f"\nInstruction: {episode['instruction']['instruction_text']}\n"
        print(msg, flush=True)

        episode_eval_args = eval_args + [f"--episode_idx={i}"]
        cmd = ['python', 'scripts/navila_eval.py', *episode_eval_args]
        print(f"Running command: {shlex.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)

        if completed.returncode != 0:
            print(
                f"Episode {i} failed with return code {completed.returncode}",
                flush=True,
            )
            print(f"Failed command: {shlex.join(cmd)}", flush=True)
            if args.stop_on_error:
                return completed.returncode

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NaVILA benchmark with an embedded local VLM server."
    )
    parser.add_argument(
        "--r2r-data-path",
        type=str,
        default="isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz",
    )
    parser.add_argument(
        "--navila-model-path",
        "--model_path",
        dest="navila_model_path",
        type=str,
        default="/home/zhaojing/mnt/legged_nav/NaVILA/NaVILA-llama3-8B-8f-scanqa-rxr",
        help="Path to the NaVILA/LLaVA checkpoint root.",
    )
    parser.add_argument("--task", type=str, default="go2_matterport_vision")
    parser.add_argument(
        "--low_level_policy_dir",
        "--low-level-policy-dir",
        dest="low_level_policy_dir",
        type=str,
        default="2024-09-25_23-22-02",
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Exclusive episode index. Defaults to the end of the dataset.",
    )
    parser.add_argument(
        "--vlm_host",
        "--vlm-host",
        dest="vlm_host",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--vlm_port",
        "--vlm-port",
        dest="vlm_port",
        type=int,
        default=54321,
    )
    parser.add_argument("--max_episode_seconds", type=float, default=None)
    parser.add_argument("--max_vlm_calls", type=int, default=None)
    parser.add_argument("--vlm-precision", type=str, default="W16A16")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--vlm-device", type=str, default="cuda")
    parser.add_argument("--isaac-device-id", type=int, default=None)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--server-startup-timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-start-vlm-server",
        action="store_true",
        help="Assume a VLM server is already running.",
    )
    parser.add_argument(
        "--vlm-launch-command",
        type=str,
        default=None,
        help="Shell command used to launch vlm_server.py in a separate environment.",
    )
    parser.add_argument(
        "--vlm-log-file",
        type=str,
        default=None,
        help="Log file for --vlm-launch-command. Defaults to vlm_server_<pid>.log.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the benchmark when one episode subprocess exits non-zero.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server_handle, thread = start_vlm_server(args)
    try:
        return run_benchmark(args)
    finally:
        if server_handle is not None:
            server_handle.stop()
        if thread is not None:
            thread.join(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
