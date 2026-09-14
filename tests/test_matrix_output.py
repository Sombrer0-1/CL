import json
from orion_repro.run_matrix import parse_child_summary


def test_child_summary_after_logs_and_pretty_printed():
    expected={"run_id":"example", "status":"completed", "nested":{"a":1}}
    output="loading dataset\n"+json.dumps(expected,indent=2)+"\n"
    assert parse_child_summary(output)==expected


def test_unrelated_json_is_not_a_run_summary():
    assert parse_child_summary('log\n{"status":"completed"}') is None
