"""Explicit plugin-switch ablation; URGE's raw proposals are never overwritten."""

PLUGIN_POLICIES = ("adaptive", "fixed_default", "fixed_advanced")


def applied_optimizer_mode(controller, proposed):
    mode = controller.get("plugin_policy", "adaptive")
    if mode == "adaptive":
        return proposed
    if mode == "fixed_default":
        return "default"
    if mode == "fixed_advanced":
        return "advanced"
    raise ValueError(f"unknown controller.plugin_policy: {mode}")


def initial_optimizer_mode(spec):
    """k=0 plugin mode. F11 starts advanced; R11 starts default; O11 follows optional_start_enabled."""
    policy = (spec.get("controller") or {}).get("plugin_policy", "adaptive")
    start_enabled = bool((spec.get("algorithm") or {}).get("optional_start_enabled", False))
    if policy == "fixed_advanced":
        return "advanced"
    if policy == "fixed_default":
        return "default"
    return "advanced" if start_enabled else "default"
