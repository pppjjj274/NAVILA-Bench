#!/usr/bin/env python3
"""
Entry point for the full VLN pipeline (upper VLM + low-level policy).

Currently only --dry-run is implemented.  Real policy integration is stubbed
out in run_policy_control_loop() with TODO markers.
"""
import argparse
import sys

from action_parser import parse_action
from command_mapper import map_action_to_command
from vlm_agent import VLMNavigationAgent


def run_policy_control_loop(
    instruction: str,
    checkpoint: str,
    num_steps: int = 100,
):
    """Placeholder for the full VLM + policy control loop.

    TODO items (do NOT execute — just documentation):
        - 参考 play.py 创建 env
        - 加载 checkpoint
        - 得到 policy
        - reset 得到 obs
        - 优先用 command_manager 或 env.update_command 注入 vx,vy,wz
        - obs[:, 6:9] 只是临时假设，后续必须确认 observation 结构
        - 按 duration/dt 循环 policy(obs), env.step(actions)
    """
    raise NotImplementedError(
        "run_policy_control_loop is a placeholder. "
        "Use --dry-run to test the upper pipeline."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Upper-level VLN demo with optional policy integration."
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="go forward",
        help="Navigation instruction in natural language.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock VLM (default, only mode available).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Only run the upper pipeline (VLM → parse → map) and print.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint (required when NOT in --dry-run).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Max control steps for the policy loop.",
    )

    args = parser.parse_args()

    # --- VLM ---
    agent = VLMNavigationAgent(mode="mock")
    raw_action = agent.predict(args.instruction)

    # --- Parse ---
    parsed = parse_action(raw_action)

    # --- Map to low-level command ---
    command = map_action_to_command(parsed)

    # --- Output ---
    if args.dry_run:
        print(f"Instruction    : {args.instruction}")
        print(f"VLM raw action : {raw_action}")
        print(f"Parsed action  : {parsed}")
        print(f"Mapped command : vx={command['vx']:.3f}, "
              f"vy={command['vy']:.3f}, wz={command['wz']:.3f}, "
              f"duration={command['duration']:.3f}")
        print("Dry-run only. Low-level policy not executed.")
    else:
        if not args.checkpoint:
            print(
                "ERROR: --checkpoint is required when NOT in --dry-run mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        run_policy_control_loop(
            instruction=args.instruction,
            checkpoint=args.checkpoint,
            num_steps=args.num_steps,
        )


if __name__ == "__main__":
    main()
