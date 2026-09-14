import json

from orion_repro.report import summarize_registry


def test_inventory_counts_runs_not_events_and_preserves_failures(tmp_path):
    path = tmp_path / "registry.jsonl"
    events = [
        {"run_id": "a", "status": "running", "phase": "development"},
        {"run_id": "a", "status": "completed"},
        {"run_id": "a", "status": "completed"},
        {"run_id": "b", "status": "running"},
        {"run_id": "b", "status": "cuda_oom"},
        {"run_id": "c", "status": "running"},
        {"run_id": "c", "status": "interrupted", "cause": "unknown"},
        {"run_id": "d", "status": "planned"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in events))
    result = summarize_registry(path)
    assert result["n_records"] == 8
    assert result["n_runs"] == 4
    assert result["status_counts"] == {"completed": 1, "cuda_oom": 1, "interrupted": 1, "planned": 1}
    assert result["completed_runs"][0]["phase"] == "development"
    assert {r["run_id"] for r in result["failed_runs"]} == {"b", "c"}
    assert result["pending_runs"] == [{"run_id": "d", "status": "planned"}]


def test_empty_inventory_does_not_invent_results(tmp_path):
    path = tmp_path / "registry.jsonl"
    path.write_text("")
    result = summarize_registry(path)
    assert result["n_runs"] == 0
    assert result["status_counts"] == {}
    assert result["completed_runs"] == []
