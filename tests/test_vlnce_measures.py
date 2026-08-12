from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


MEASURES_PATH = (
    Path(__file__).parents[1]
    / "isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/measures.py"
)
SPEC = spec_from_file_location("tested_vlnce_measures", MEASURES_PATH)
MEASURES = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MEASURES)


class ConstantDistance:
    def __init__(self, value):
        self.value = value

    def get_metric(self):
        return self.value


def test_success_reset_clears_previous_episode_stop_before_measuring():
    env = SimpleNamespace(is_stop_called=True)
    manager = MEASURES.MeasureManager()
    manager.measures[MEASURES.DistanceToGoal.cls_uuid] = ConstantDistance(0.0)
    measure = MEASURES.Success(
        env,
        {"goals": [{"radius": 3.0}]},
        manager,
    )

    measure.reset_metric()

    assert env.is_stop_called is False
    assert measure.get_metric() == 0.0


def test_measure_factory_rejects_unknown_names_without_eval():
    with pytest.raises(ValueError, match="unknown navigation measure"):
        MEASURES.add_measurement(None, {}, ["__import__('os').system('false')"])
