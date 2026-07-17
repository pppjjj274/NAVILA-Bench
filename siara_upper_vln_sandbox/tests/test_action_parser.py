"""Tests for action_parser.parse_action()."""

import pytest

from action_parser import parse_action


class TestParseAction:
    def test_move_forward_valid(self):
        result = parse_action("MOVE_FORWARD 0.5")
        assert result == {"action": "MOVE_FORWARD", "value": 0.5}

    def test_turn_left_valid(self):
        result = parse_action("TURN_LEFT 30")
        assert result == {"action": "TURN_LEFT", "value": 30.0}

    def test_turn_right_valid(self):
        result = parse_action("TURN_RIGHT 45")
        assert result == {"action": "TURN_RIGHT", "value": 45.0}

    def test_stop(self):
        result = parse_action("STOP")
        assert result == {"action": "STOP", "value": 0.0}

    def test_case_insensitive(self):
        result = parse_action("move_forward 1.0")
        assert result == {"action": "MOVE_FORWARD", "value": 1.0}

    def test_extra_whitespace(self):
        result = parse_action("  MOVE_FORWARD   0.5  ")
        assert result == {"action": "MOVE_FORWARD", "value": 0.5}

    def test_unknown_action_fallback(self):
        result = parse_action("JUMP 5")
        assert result == {"action": "STOP", "value": 0.0}

    def test_empty_string_fallback(self):
        result = parse_action("")
        assert result == {"action": "STOP", "value": 0.0}

    def test_none_fallback(self):
        result = parse_action(None)
        assert result == {"action": "STOP", "value": 0.0}

    def test_no_value_fallback(self):
        result = parse_action("MOVE_FORWARD")
        assert result == {"action": "STOP", "value": 0.0}
