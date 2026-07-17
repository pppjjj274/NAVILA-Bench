#!/usr/bin/env python3
"""
Dry-run demo of the upper-level VLN pipeline:
    instruction → VLMNavigationAgent → parse_action → map_action_to_command

Prints every stage so you can trace the full chain without Isaac or a policy.
"""
import sys

from action_parser import parse_action
from command_mapper import map_action_to_command
from vlm_agent import VLMNavigationAgent


def main():
    instruction = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "go forward"

    agent = VLMNavigationAgent(mode="mock")
    raw_action = agent.predict(instruction)

    parsed = parse_action(raw_action)
    command = map_action_to_command(parsed)

    print(f"Instruction        : {instruction}")
    print(f"VLM raw action     : {raw_action}")
    print(f"Parsed action      : {parsed}")
    print(f"Mapped command     : vx={command['vx']:.3f}, "
          f"vy={command['vy']:.3f}, wz={command['wz']:.3f}, "
          f"duration={command['duration']:.3f}")


if __name__ == "__main__":
    main()
