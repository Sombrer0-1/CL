"""H7 deterministic flag wiring. Does not claim prefetch benefit."""

from __future__ import annotations

import os

from orion_repro.runner.loop import configure_torch


def test_configure_torch_sets_cublas_when_deterministic(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    configure_torch(
        {
            "training": {
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "cudnn_benchmark": False,
            }
        }
    )
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_configure_torch_does_not_force_cublas_when_not_deterministic(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    configure_torch(
        {
            "training": {
                "deterministic_algorithms": False,
                "allow_tf32": False,
                "cudnn_benchmark": False,
            }
        }
    )
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
