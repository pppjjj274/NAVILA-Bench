"""
VLM Navigation Agent — mock version.

Currently uses keyword-based mock rules to simulate a VLM.
TODO: Replace with Qwen3-VL-8B-Instruct inference when available.
"""


class VLMNavigationAgent:
    """Mock VLM agent for Vision-Language Navigation.

    Keyword-based mock rules:
        - "stop", "停", "reached", "到达"  →  STOP
        - "left", "左"                     →  TURN_LEFT 30
        - "right", "右"                    →  TURN_RIGHT 30
        - default                           →  MOVE_FORWARD 0.5
    """

    def __init__(self, mode: str = "mock"):
        """Initialise the agent.

        Args:
            mode: "mock" (default) for keyword-based rules.
                  Reserved for future modes: "qwen3vl", etc.
        """
        self.mode = mode
        if mode != "mock":
            raise ValueError(
                f"Unsupported mode '{mode}'. Only 'mock' is available for now."
            )

    def predict(self, instruction: str, image=None) -> str:
        """Return a raw action string for the given instruction.

        Args:
            instruction: Natural-language navigation instruction.
            image: (unused in mock mode) Observation image.

        Returns:
            Raw action string, e.g. "MOVE_FORWARD 0.5".
        """
        if self.mode == "mock":
            return self._mock_predict(instruction)
        # TODO: Qwen3-VL-8B-Instruct inference path
        raise NotImplementedError("Only mock mode is supported.")

    def _mock_predict(self, instruction: str) -> str:
        lower = instruction.lower()

        # Check for stop-related keywords
        if any(kw in lower for kw in ("stop", "停", "reached", "到达")):
            return "STOP"

        # Check for left turn
        if any(kw in lower for kw in ("left", "左")):
            return "TURN_LEFT 30"

        # Check for right turn
        if any(kw in lower for kw in ("right", "右")):
            return "TURN_RIGHT 30"

        # Default: move forward
        return "MOVE_FORWARD 0.5"
