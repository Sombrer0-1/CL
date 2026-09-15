"""Failure context must distinguish completed training from completed evaluation."""


def failure_metadata(spec, *, phase, experience, trained, evaluated):
    return {
        'failure_phase': phase,
        'failure_experience': experience,
        'n_experiences_trained': trained,
        'n_experiences_run': evaluated,
        'phase': spec['phase'],
        'dataset': spec['dataset']['name'],
        'method_id': spec['method_id'],
        'eval_batch': spec['training']['eval_batch'],
        'budget_enforcement': spec['budget']['enforcement'],
        'budget_limit_bytes': spec['budget'].get('limit_bytes'),
    }
