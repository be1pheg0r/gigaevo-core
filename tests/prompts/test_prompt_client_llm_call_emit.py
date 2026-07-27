"""Tests for cost/time tracking and LLM_CALL emission in the prompt-eval LLMClient.

Mirrors `tests/problems/test_chain_client_llm_call_emit.py` for
`problems/prompts/client.py::LLMClient` — the analogous client used by
prompt-coevolution validators, which previously left the same kind of gap
(no `[LLM_CALL]` trace for validator LLM calls).
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import tenacity

from problems.prompts.client import LLMClient

_LINE_RE = re.compile(r"\[LLM_CALL\]\s+(\{.*\})\s*$")


def _fake_response(prompt_tokens: int, completion_tokens: int, content: str):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def _read_llm_call_bodies(log_file) -> list[dict]:
    text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    bodies = []
    for line in text.splitlines():
        m = _LINE_RE.search(line)
        if m:
            bodies.append(json.loads(m.group(1)))
    return bodies


@pytest.fixture(autouse=True)
def _log_file_env(monkeypatch, tmp_path):
    log_file = tmp_path / "evolution_test.log"
    monkeypatch.setenv("GIGAEVO_LOG_FILE", str(log_file))
    monkeypatch.setenv("GIGAEVO_PROGRAM_ID", "prog-xyz")
    return log_file


def _make_client() -> LLMClient:
    return LLMClient(
        model="qwen/qwen3-8b",
        client_kwargs={"api_key": "sk-test", "base_url": "http://x/v1"},
    )


class TestSuccessfulCall:
    async def test_records_call_log_with_timing_and_emits_llm_call(
        self, _log_file_env
    ) -> None:
        client = _make_client()
        client.client.chat.completions.create = AsyncMock(
            return_value=_fake_response(5, 7, "hi")
        )

        result = await client("some prompt")

        assert result == "hi"
        assert len(client.call_logs) == 1
        log = client.call_logs[0]
        assert log.ok is True
        assert log.error_type is None
        assert log.duration_ms >= 0.0

        bodies = _read_llm_call_bodies(_log_file_env)
        assert len(bodies) == 1
        assert bodies[0]["stage"] == "PromptValidatorLLM"
        assert bodies[0]["program_id"] == "prog-xyz"
        assert bodies[0]["ok"] is True


class TestBudgetExceeded:
    async def test_budget_exceeded_still_logs_ok_true_call(self, _log_file_env) -> None:
        client = _make_client()
        client.max_cost = 1e-12  # any nonzero cost blows the budget immediately
        client.client.chat.completions.create = AsyncMock(
            return_value=_fake_response(1000, 1000, "big")
        )

        # The budget-exceeded ValueError is itself retried (tenacity's default
        # retry predicate matches any Exception) — pre-existing behavior,
        # unchanged by this refactor. After 3 attempts it surfaces as
        # RetryError, same as any other exhausted-retry failure.
        with pytest.raises(tenacity.RetryError):
            await client("some prompt")

        # Every attempt's API call itself succeeded — only the manual budget
        # check (which runs after logging) raised.
        assert len(client.call_logs) == 3
        assert all(log.ok for log in client.call_logs)

        bodies = _read_llm_call_bodies(_log_file_env)
        assert all(b["ok"] is True for b in bodies)


class TestFailingCall:
    async def test_retries_and_emits_one_llm_call_per_attempt(
        self, _log_file_env
    ) -> None:
        client = _make_client()
        client.client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        with pytest.raises(tenacity.RetryError):
            await client("some prompt")

        assert len(client.call_logs) == 3
        assert all(not log.ok for log in client.call_logs)

        bodies = _read_llm_call_bodies(_log_file_env)
        assert [b["attempt"] for b in bodies] == [1, 2, 3]
        assert all(b["ok"] is False for b in bodies)
