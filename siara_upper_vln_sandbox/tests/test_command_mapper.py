"""Tests for command_mapper.map_action_to_command()."""

import math

import pytest

from command_mapper import map_action_to_command


class TestMapActionToCommand:
    def test_move_forward(self):
        cmd = map_action_to_command({"action": "MOVE_FORWARD", "value": 1.0})
        assert cmd["vx"] == 0.5
        assert cmd["vy"] == 0.0
        assert cmd["wz"] == 0.0
        assert cmd["duration"] == pytest.approx(2.0)  # 1.0 / 0.5

    def test_turn_left(self):
        cmd = map_action_to_command({"action": "TURN_LEFT", "value": 30.0})
        assert cmd["vx"] == 0.0
        assert cmd["vy"] == 0.0
        assert cmd["wz"] == pytest.approx(math.pi / 6.0)
        expected_duration = math.radians(30.0) / (math.pi / 6.0)
        assert cmd["duration"] == pytest.approx(expected_duration)

    def test_turn_right(self):
        cmd = map_action_to_command({"action": "TURN_RIGHT", "value": 45.0})
        assert cmd["vx"] == 0.0
        assert cmd["vy"] == 0.0
        assert cmd["wz"] == pytest.approx(-math.pi / 6.0)
        expected_duration = math.radians(45.0) / (math.pi / 6.0)
        assert cmd["duration"] == pytest.approx(expected_duration)

    def test_stop(self):
        cmd = map_action_to_command({"action": "STOP", "value": 0.0})
        assert cmd == {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 1.0}

    def test_unknown_action_fallback(self):
        cmd = map_action_to_command({"action": "UNKNOWN", "value": 99.0})
        assert cmd == {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 1.0}

    def test_missing_action_key(self):
        cmd = map_action_to_command({"value": 0.5})
        assert cmd == {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 1.0}
