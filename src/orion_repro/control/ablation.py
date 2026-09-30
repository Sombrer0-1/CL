"""Explicit plugin-switch ablation; URGE's raw proposals are never overwritten."""

PLUGIN_POLICIES = (
    "adaptive",
    "fixed_default",
    "fixed_advanced",
    "scripted_pause_keep_state",
    "scripted_release_rebuild",
)


def applied_optimizer_mode(controller, proposed):
    mode = controller.get("plugin_policy", "adaptive")
    if mode == "adaptive":
        return proposed
    if mode == "fixed_default":
        return "default"
    if mode == "fixed_advanced":
        return "advanced"
    if mode in {"scripted_pause_keep_state", "scripted_release_rebuild"}:
        return proposed
    raise ValueError(f"unknown controller.plugin_policy: {mode}")


def initial_optimizer_mode(spec):
    """k=0 plugin mode. F11 starts advanced; R11 starts default; O11 follows optional_start_enabled."""
    controller = spec.get("controller") or {}
    policy = controller.get("plugin_policy", "adaptive")
    start_enabled = bool((spec.get("algorithm") or {}).get("optional_start_enabled", False))
    sched = controller.get("plugin_schedule") or []
    if policy in {"scripted_pause_keep_state", "scripted_release_rebuild"} and sched:
        return str(sched[0])
    if policy == "fixed_advanced":
        return "advanced"
    if policy == "fixed_default":
        return "default"
    return "advanced" if start_enabled else "default"
