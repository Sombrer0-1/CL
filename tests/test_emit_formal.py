from orion_repro.emit_formal import _apply_method_on_base, LR_SKIP_BASES, METHOD_TEMPLATES, E03_CONTEXTS, E04_ALGO_TEMPLATES, SEEDS


def test_lr_on_projection_algorithms_is_refused_not_silenced():
    dummy = {"algorithm": {"base": "gem", "optional_plugins": "none"}, "controller": {}, "training": {}, "replay": {}}
    assert _apply_method_on_base(dummy, "lr_reconstructed", base="gem") is None
    assert _apply_method_on_base(dummy, "lr_reconstructed", base="agem") is None
    assert _apply_method_on_base(dummy, "lr_reconstructed", base="gss") is None
    kept = _apply_method_on_base(
        {
            "algorithm": {"base": "er", "optional_plugins": "none"},
            "controller": {},
            "training": {"new_batch": 16, "replay_batch": 16},
            "replay": {"capacity": 200},
        },
        "lr_reconstructed",
        base="er",
    )
    assert kept is not None
    assert kept["algorithm"]["base"] == "lr"


def test_historical_full_scope_counts_preserved_for_reference():
    n_e03 = len(E03_CONTEXTS) * len(METHOD_TEMPLATES) * len(SEEDS)
    n_e04 = 0
    n_skip = 0
    for base in E04_ALGO_TEMPLATES:
        for method_id in METHOD_TEMPLATES:
            for _ in SEEDS:
                if method_id == "lr_reconstructed" and base in LR_SKIP_BASES:
                    n_skip += 1
                else:
                    n_e04 += 1
    assert n_e03 == 60
    assert n_e04 == 27
    assert n_skip == 9
    # Historical full-scope PLAN 120 = 96 online + 24 oracle selected. E04 LR cells are 9 of the 96.
    planned_online = 8 * 4 * 3
    assert planned_online == 96
    assert n_e03 + n_e04 + n_skip == planned_online
