"""Smoke tests for the task-builder demo web app.

Not wired into the main pytest suite (this is an experiment/demo tool, not
core library code) -- run directly: python tools/task_builder_web/test_app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as app_module  # noqa: E402


def test_json_re_extracts_cost_monitor_line():
    line = (
        "2026-07-30 10:00:00.000 | INFO | x:y:1 | [CostMonitorHookJSON] "
        '{"mutant": 3, "predicted_tokens": 12345, "token_ci_low": 10000, '
        '"token_ci_high": 15000, "predicted_duration_s": 120.5, '
        '"ci_low_s": 100.0, "ci_high_s": 140.0}'
    )
    m = app_module._JSON_RE.search(line)
    assert m is not None
    import json

    d = json.loads(m.group(1))
    assert d["mutant"] == 3
    assert d["predicted_tokens"] == 12345


def test_llm_call_re_extracts_tokens():
    line = (
        "2026-07-30 10:00:00.000 | INFO | x:y:1 | [LLM_CALL] "
        '{"event": "LLM_CALL", "tokens_in": 100, "tokens_out": 50, "ok": true}'
    )
    m = app_module._LLM_CALL_RE.search(line)
    assert m is not None
    import json

    d = json.loads(m.group(1))
    assert d["tokens_in"] + d["tokens_out"] == 150


def test_next_redis_db_cycles_through_pool():
    app_module._db_cursor = 0
    seen = [
        app_module._next_redis_db() for _ in range(len(app_module.REDIS_DB_POOL) + 1)
    ]
    assert seen[0] == app_module.REDIS_DB_POOL[0]
    assert seen[-1] == seen[0]  # wraps around


def test_job_public_dict_hides_private_fields_and_serializes_points():
    job = app_module.Job(id="abc", request="do a thing")
    job.points.append(
        app_module.MutantPoint(
            mutant=1,
            predicted_tokens=100,
            token_ci_low=80,
            token_ci_high=120,
            predicted_duration_s=10.0,
            ci_low_s=8.0,
            ci_high_s=12.0,
            elapsed_s=5.0,
            actual_tokens_so_far=90,
        )
    )
    d = job.public_dict()
    assert "_proc" not in d
    assert d["points"][0]["predicted_tokens"] == 100


def test_routes_registered():
    paths = {r.path for r in app_module.app.routes}
    assert "/api/jobs" in paths
    assert "/api/jobs/{job_id}" in paths
    assert "/api/jobs/{job_id}/download" in paths


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} tests passed")
