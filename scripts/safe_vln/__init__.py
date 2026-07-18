"""Expose the repository Safe-VLN package to directly executed scripts."""

from pathlib import Path


# ``python scripts/<entrypoint>.py`` places ``scripts`` (not the repository
# root) on sys.path. Point this package at the implementation in the root so
# existing benchmark launch commands keep working without PYTHONPATH changes.
__path__ = [str(Path(__file__).resolve().parents[2] / "safe_vln")]

from .actions import ACTIONS, SafeAction
from .cmdp import LagrangeController, compute_gae, compute_returns, safe_advantage
from .trajectory import SafeTrajectoryRecorder

__all__ = [
    "ACTIONS",
    "SafeAction",
    "SafeTrajectoryRecorder",
    "LagrangeController",
    "compute_gae",
    "compute_returns",
    "safe_advantage",
]
