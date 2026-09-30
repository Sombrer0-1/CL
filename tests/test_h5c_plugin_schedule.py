"""H5c scripted plugin lifecycle helpers. Does not claim formal H5."""

from types import SimpleNamespace

from orion_repro.runner.loop import _release_optional_plugin_state, _scheduled_plugin_mode


def test_scheduled_plugin_mode_none_without_schedule():
    assert _scheduled_plugin_mode({"controller": {}}, 0) is None


def test_scheduled_plugin_mode_reads_index():
    spec = {"controller": {"plugin_schedule": ["advanced", "default", "advanced"]}}
    assert _scheduled_plugin_mode(spec, 0) == "advanced"
    assert _scheduled_plugin_mode(spec, 1) == "default"
    assert _scheduled_plugin_mode(spec, 2) == "advanced"


def test_scheduled_plugin_mode_rejects_bad_value():
    spec = {"controller": {"plugin_schedule": ["always_on"]}}
    try:
        _scheduled_plugin_mode(spec, 0)
    except RuntimeError as exc:
        assert "illegal plugin_schedule" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


class SparseGEMPlugin:
    def __init__(self) -> None:
        self.memory_x = {"0": object()}
        self.memory_y = {"0": object()}
        self.memory_tid = {"0": object()}


class EWCPlugin:
    def __init__(self) -> None:
        self.saved_params = {"0": {"w": object()}}
        self.importances = {"0": {"w": object()}}


class OtherPlugin:
    keep = 1


def test_release_optional_plugin_state_clears_gem_ewc_keeps_other():
    gem = SparseGEMPlugin()
    ewc = EWCPlugin()
    other = OtherPlugin()
    strategy = SimpleNamespace(plugins=[gem, ewc, other])
    released = _release_optional_plugin_state(strategy)
    assert released["SparseGEMPlugin"] == "cleared_memory_xy"
    assert released["EWCPlugin"] == "cleared_fisher"
    assert gem.memory_x == {}
    assert gem.memory_y == {}
    assert gem.memory_tid == {}
    assert ewc.saved_params == {}
    assert ewc.importances == {}
    assert other.keep == 1
