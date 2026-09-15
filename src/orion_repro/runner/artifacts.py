"""Artifact writers for a single run directory (PLAN §11.3)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


class RunArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events = (self.run_dir / "events.jsonl").open("a", encoding="utf-8")
        self._control = (self.run_dir / "control_trace.jsonl").open("a", encoding="utf-8")
        self._resource = self.run_dir / "resource_trace.csv"
        self._metrics = self.run_dir / "experience_metrics.csv"
        self._matrix = self.run_dir / "accuracy_matrix.csv"
        self._phases = self.run_dir / "phase_trace.csv"
        self._plugin_activity = self.run_dir / "plugin_activity.jsonl"
        self._consumption = self.run_dir / "consumption.jsonl"

    def append_jsonl(self, path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def close(self) -> None:
        self._events.close()
        self._control.close()

    def write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def event(self, payload: Mapping[str, Any]) -> None:
        self._events.write(json.dumps(payload, default=str) + "\n")
        self._events.flush()

    def control(self, payload: Mapping[str, Any]) -> None:
        self._control.write(json.dumps(payload, default=str) + "\n")
        self._control.flush()

    def append_csv(self, path: Path, row: Mapping[str, Any], fieldnames: list[str] | None = None) -> None:
        fields = fieldnames or list(row.keys())
        new_file = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
