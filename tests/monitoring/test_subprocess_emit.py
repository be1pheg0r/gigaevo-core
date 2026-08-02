"""Tests for `emit_llm_call_from_subprocess`.

This is the emission seam for LLM clients that run inside
`CallValidatorFunction`'s isolated subprocess (chain-eval and prompt-eval
validators). Their stdout is a binary length-prefixed protocol and their
stderr/loguru output never reaches the parent process on the success path,
so this helper appends a plain `[LLM_CALL] {json}` line directly to the
run's log file (path threaded in via `GIGAEVO_LOG_FILE`) instead of going
through loguru.
"""

from __future__ import annotations

import json
import re

from gigaevo.monitoring.subprocess_emit import emit_llm_call_from_subprocess

_LINE_RE = re.compile(r"\[LLM_CALL\]\s+(\{.*\})\s*$")


def test_noop_when_log_file_env_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GIGAEVO_LOG_FILE", raising=False)
    emit_llm_call_from_subprocess(
        stage="ChainValidatorLLM",
        model="Qwen/Qwen3-8B",
        endpoint="http://localhost:8000/v1",
        attempt=1,
        ok=True,
        latency_ms=12.3,
        tokens_in=10,
        tokens_out=20,
        error_type=None,
    )
    # Nothing should have been created anywhere under tmp_path either.
    assert not list(tmp_path.iterdir())


def test_appends_llm_call_line_with_expected_fields(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "evolution_test.log"
    monkeypatch.setenv("GIGAEVO_LOG_FILE", str(log_file))
    monkeypatch.setenv("GIGAEVO_PROGRAM_ID", "prog-123")

    emit_llm_call_from_subprocess(
        stage="ChainValidatorLLM",
        model="Qwen/Qwen3-8B",
        endpoint="http://localhost:8000/v1",
        attempt=2,
        ok=False,
        latency_ms=456.7,
        tokens_in=111,
        tokens_out=222,
        error_type="RuntimeError",
    )

    text = log_file.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if "[LLM_CALL]" in line]
    assert len(lines) == 1, f"expected exactly one [LLM_CALL] line, got {lines}"

    match = _LINE_RE.search(lines[0])
    assert match, f"line does not match [LLM_CALL] {{json}} shape: {lines[0]!r}"
    body = json.loads(match.group(1))

    assert body["event"] == "LLM_CALL"
    assert body["stage"] == "ChainValidatorLLM"
    assert body["program_id"] == "prog-123"
    assert body["model"] == "Qwen/Qwen3-8B"
    assert body["attempt"] == 2
    assert body["ok"] is False
    assert body["latency_ms"] == 456.7
    assert body["tokens_in"] == 111
    assert body["tokens_out"] == 222
    assert body["error_type"] == "RuntimeError"


def test_appends_multiple_calls_across_invocations(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "evolution_test.log"
    monkeypatch.setenv("GIGAEVO_LOG_FILE", str(log_file))
    monkeypatch.delenv("GIGAEVO_PROGRAM_ID", raising=False)

    for attempt in (1, 2):
        emit_llm_call_from_subprocess(
            stage="PromptValidatorLLM",
            model="qwen/qwen3-8b",
            endpoint="https://openrouter.ai/api/v1",
            attempt=attempt,
            ok=attempt == 2,
            latency_ms=1.0,
            tokens_in=1,
            tokens_out=1,
            error_type=None if attempt == 2 else "TimeoutError",
        )

    lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    bodies = [json.loads(_LINE_RE.search(line).group(1)) for line in lines]
    assert [b["attempt"] for b in bodies] == [1, 2]
    assert bodies[0]["program_id"] is None


def test_never_raises_on_unwritable_log_file(monkeypatch, tmp_path) -> None:
    # Point at a path inside a nonexistent directory — open() will raise.
    monkeypatch.setenv("GIGAEVO_LOG_FILE", str(tmp_path / "missing_dir" / "x.log"))
    emit_llm_call_from_subprocess(
        stage="ChainValidatorLLM",
        model="m",
        endpoint="e",
        attempt=1,
        ok=True,
        latency_ms=0.0,
        tokens_in=0,
        tokens_out=0,
        error_type=None,
    )  # must not raise
