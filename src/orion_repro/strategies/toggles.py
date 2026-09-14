"""Enable/disable optional GEM/EWC without duplicating GEM (A10)."""

from __future__ import annotations

from avalanche.core import SupervisedPlugin

_HOOKS = (
    "before_training",
    "after_training",
    "before_training_exp",
    "after_training_exp",
    "before_train_dataset_adaptation",
    "after_train_dataset_adaptation",
    "before_training_epoch",
    "after_training_epoch",
    "before_training_iteration",
    "after_training_iteration",
    "before_forward",
    "after_forward",
    "before_backward",
    "after_backward",
    "before_update",
    "after_update",
)


class TogglePlugin(SupervisedPlugin):
    """Keep plugin state in memory while skipping compute when disabled.

    Closing mode is pause-compute-keep-state (PLAN §7.2). Bytes still count.
    """

    def __init__(self, inner: SupervisedPlugin, *, enabled: bool = True, name: str = "") -> None:
        super().__init__()
        self.inner = inner
        self.enabled = enabled
        self.name = name or type(inner).__name__

    def __getattr__(self, item: str):
        return getattr(self.inner, item)


def _make_hook(hook: str):
    def _fn(self, strategy, *args, **kwargs):
        if not self.enabled:
            return None
        fn = getattr(self.inner, hook, None)
        if fn is None:
            return None
        return fn(strategy, *args, **kwargs)

    _fn.__name__ = hook
    return _fn


for _hook in _HOOKS:
    setattr(TogglePlugin, _hook, _make_hook(_hook))
