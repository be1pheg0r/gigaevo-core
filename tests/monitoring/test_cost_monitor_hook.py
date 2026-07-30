from __future__ import annotations

import pytest

from gigaevo.monitoring.cost_monitor_hook import CostMonitorHook
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.monitoring.emit import emit, reset_subscribers
from gigaevo.monitoring.events import LLMCall


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    reset_subscribers()


def _call(stage: str, tokens: int, latency_ms: float, tokens_out: int = 0) -> None:
    emit(LLMCall(
        stage=stage, endpoint="", model="x", ok=True,
        latency_ms=latency_ms, tokens_in=tokens, tokens_out=tokens_out,
    ))


async def _fire(hook: CostMonitorHook) -> None:
    await hook.__call__()


class TestMutantBucketing:
    @pytest.mark.asyncio
    async def test_calls_are_bucketed_per_mutant_not_per_call(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)  # never fire agent

        # Mutant 1: two calls of stage A.
        _call("A", 100, 0.0)
        _call("A", 50, 0.0)
        await _fire(hook)
        assert hook._tokens_by_stage["A"] == [150.0]

        # Mutant 2: one more call of stage A.
        _call("A", 200, 0.0)
        await _fire(hook)
        assert hook._tokens_by_stage["A"] == [150.0, 200.0]

    @pytest.mark.asyncio
    async def test_estimate_updates_prediction_every_mutant_not_just_on_agent_interval(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)

        assert pred.predicted_total_tokens == 0
        _call("A", 100, 1000.0)
        await _fire(hook)
        assert pred.predicted_total_tokens > 0

    @pytest.mark.asyncio
    async def test_mutant_with_no_calls_does_not_add_a_zero_point(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)

        _call("A", 100, 0.0)
        await _fire(hook)  # mutant 1: one point
        await _fire(hook)  # mutant 2: no LLM calls at all
        assert hook._tokens_by_stage["A"] == [100.0]

    @pytest.mark.asyncio
    async def test_tokens_out_bucketed_separately_for_duration_model(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)

        _call("A", 100, 1000.0, tokens_out=40)
        _call("A", 50, 500.0, tokens_out=10)
        await _fire(hook)
        assert hook._tokens_out_by_stage["A"] == [50.0]
        assert pred.predicted_duration_s > 0
