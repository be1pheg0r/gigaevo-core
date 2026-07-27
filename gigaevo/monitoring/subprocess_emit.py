"""LLM_CALL emission for code that runs inside an isolated validator subprocess.

`CallValidatorFunction` (see `programs/stages/python_executors`) runs
`validate()` in a subprocess whose stdout is a binary length-prefixed
protocol and whose stderr/loguru output is discarded on the success path —
anything a validator's LLM client logs there never reaches the parent
process. The chain-eval LLM (the second `LLMClient` that `validate.py` spins
up per task) was therefore invisible to canonical-event logging: only the
mutator's calls showed up in `[LLM_CALL]` lines.

The one channel that does survive the subprocess boundary without protocol
or capture issues is a direct append to the run's log file, whose path is
threaded in via the `GIGAEVO_LOG_FILE` env var (see
`programs/stages/python_executors/execution.py`). Downstream tooling
(`log_audit.py`'s `EVENT_LINE_RE`, external cost-model analysis) parses
`[EVENT_NAME] {json}` lines regardless of which process wrote them, so a
plain file append is sufficient here — no loguru sink required.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os

from gigaevo.monitoring.events import LLMCall

_LOG_FILE_ENV = "GIGAEVO_LOG_FILE"
_PROGRAM_ID_ENV = "GIGAEVO_PROGRAM_ID"


def emit_llm_call_from_subprocess(
    *,
    stage: str,
    model: str,
    endpoint: str,
    attempt: int,
    ok: bool,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    error_type: str | None,
) -> None:
    """Best-effort LLM_CALL emission from inside an isolated validator subprocess.

    No-op if `GIGAEVO_LOG_FILE` isn't set (e.g. running outside a gigaevo
    subprocess, or in unit tests) or if the write fails for any reason —
    diagnostic logging must never break the actual LLM call.
    """
    log_file = os.environ.get(_LOG_FILE_ENV)
    if not log_file:
        return
    try:
        event = LLMCall(
            stage=stage,
            program_id=os.environ.get(_PROGRAM_ID_ENV),
            endpoint=endpoint,
            model=model,
            attempt=attempt,
            ok=ok,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            error_type=error_type,
        )
        payload = {"event": type(event).event, **event.model_dump(mode="json")}
        line = (
            f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
            f"INFO     | {stage}:emit:0 | "
            f"[{type(event).event}] {json.dumps(payload, ensure_ascii=False)}\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        return
