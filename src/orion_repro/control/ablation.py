"""Explicit plugin-switch ablation; URGE's raw proposals are never overwritten."""


def applied_optimizer_mode(controller, proposed):
    mode = controller.get('plugin_policy', 'adaptive')
    if mode == 'adaptive':
        return proposed
    if mode == 'fixed_default':
        return 'default'
    raise ValueError(f'unknown controller.plugin_policy: {mode}')
