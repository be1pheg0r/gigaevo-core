"""Test cost telemetry emitted by the chain-evaluation LLM client."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import tenacity

from problems.chains.client import LLMClient

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
    monkeypatch.setenv("GIGAEVO_PROGRAM_ID", "prog-abc")
    return log_file


class TestSuccessfulCall:
    async def test_records_call_log_with_timing_and_emits_llm_call(
        self, _log_file_env
    ) -> None:
        client = LLMClient(
            model="Qwen/Qwen3-8B", client_kwargs={"base_url": "http://x/v1"}
        )
        client.client.chat.completions.create = AsyncMock(
            return_value=_fake_response(10, 20, "hello")
        )

        result = await client("some prompt")

        assert result == "hello"
        assert len(client.call_logs) == 1
        log = client.call_logs[0]
        assert log.ok is True
        assert log.error_type is None
        assert log.prompt_tokens == 10
        assert log.completion_tokens == 20
        assert log.duration_ms >= 0.0
        assert log.model == "Qwen/Qwen3-8B"

        bodies = _read_llm_call_bodies(_log_file_env)
        assert len(bodies) == 1
        body = bodies[0]
        assert body["stage"] == "ChainValidatorLLM"
        assert body["program_id"] == "prog-abc"
        assert body["ok"] is True
        assert body["tokens_in"] == 10
        assert body["tokens_out"] == 20
        assert body["attempt"] == 1


class TestFailingCall:
    async def test_retries_and_emits_one_llm_call_per_attempt(
        self, _log_file_env
    ) -> None:
        client = LLMClient(
            model="Qwen/Qwen3-8B", client_kwargs={"base_url": "http://x/v1"}
        )
        client.client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        with pytest.raises(tenacity.RetryError):
            await client("some prompt")

        # 3 attempts (stop_after_attempt(3)), all failed.
        assert len(client.call_logs) == 3
        assert all(not log.ok for log in client.call_logs)
        assert all(log.error_type == "RuntimeError" for log in client.call_logs)

        bodies = _read_llm_call_bodies(_log_file_env)
        assert len(bodies) == 3
        assert [b["attempt"] for b in bodies] == [1, 2, 3]
        assert all(b["ok"] is False for b in bodies)
        assert all(b["error_type"] == "RuntimeError" for b in bodies)


class TestCopy:
    async def test_copy_shares_endpoint_but_has_isolated_call_logs(self) -> None:
        client = LLMClient(
            model="Qwen/Qwen3-8B", client_kwargs={"base_url": "http://x/v1"}
        )
        client.client.chat.completions.create = AsyncMock(
            return_value=_fake_response(1, 1, "ok")
        )
        await client("p")

        copy = client.copy()
        assert copy._endpoint == client._endpoint
        assert copy.call_logs == []
        assert len(client.call_logs) == 1
