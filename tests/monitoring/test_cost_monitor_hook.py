from __future__ import annotations

import pytest

from gigaevo.monitoring.cost_monitor_hook import CostMonitorHook, _clamp_step
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.monitoring.emit import emit, reset_subscribers
from gigaevo.monitoring.events import BackpressureSample, LLMCall, StageExec


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


class TestLlmOverrideAppliesToDuration:
    @pytest.mark.asyncio
    async def test_golden_override_scales_predicted_duration(self) -> None:
        baseline_pred = CostPrediction(max_mutants=10, max_in_flight=1)
        baseline_hook = CostMonitorHook(agent=None, prediction=baseline_pred, interval=1000)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(baseline_hook)

        overridden_pred = CostPrediction(max_mutants=10, max_in_flight=1)
        overridden_pred.llm_golden_override = 1.5
        overridden_hook = CostMonitorHook(agent=None, prediction=overridden_pred, interval=1000)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(overridden_hook)

        assert overridden_pred.predicted_duration_s == pytest.approx(
            baseline_pred.predicted_duration_s * 1.5
        )

    @pytest.mark.asyncio
    async def test_growth_override_compounds_with_golden_override(self) -> None:
        baseline_pred = CostPrediction(max_mutants=10, max_in_flight=1)
        baseline_hook = CostMonitorHook(agent=None, prediction=baseline_pred, interval=1000)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(baseline_hook)

        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        pred.llm_golden_override = 1.2
        pred.llm_growth_override = 2.0
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(hook)

        assert pred.predicted_duration_s == pytest.approx(
            baseline_pred.predicted_duration_s * 1.2 * 2.0
        )

    @pytest.mark.asyncio
    async def test_out_of_range_override_is_ignored(self) -> None:
        baseline_pred = CostPrediction(max_mutants=10, max_in_flight=1)
        baseline_hook = CostMonitorHook(agent=None, prediction=baseline_pred, interval=1000)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(baseline_hook)

        pred = CostPrediction(max_mutants=10, max_in_flight=1)  # defaults: -1.0 (no override)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(hook)

        assert pred.predicted_duration_s == pytest.approx(baseline_pred.predicted_duration_s)


class FakeAgent:
    """Captures the tools/context CostMonitorHook builds for a real agent
    call, without hitting an LLM."""

    def __init__(self):
        self.tools = None
        self.seen_tools = None

    def build_prompt(self, state):
        self.seen_tools = self.tools
        return []

    async def acall_llm(self, state):
        return state

    def parse_response(self, state):
        return {}


class TestAgentSeesRealContext:
    @pytest.mark.asyncio
    async def test_agent_receives_real_backpressure_not_hardcoded(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=8)
        agent = FakeAgent()
        hook = CostMonitorHook(agent=agent, prediction=pred, interval=1)

        emit(BackpressureSample(
            producer_held=6, buffer_held=0, in_flight=6, max_in_flight=8, llm_active=6,
        ))
        _call("A", 100, 10.0)
        await _fire(hook)

        assert "in_flight=6/8" in agent.seen_tools.get_backpressure()

    @pytest.mark.asyncio
    async def test_agent_sees_previously_set_override_not_fixed_default(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        pred.llm_golden_override = 1.6  # set by an earlier calibration cycle
        agent = FakeAgent()
        hook = CostMonitorHook(agent=agent, prediction=pred, interval=1)

        _call("A", 100, 10.0)
        await _fire(hook)

        assert "golden=1.60" in agent.seen_tools.get_model_params()


class TestClampStep:
    def test_small_move_passes_through(self) -> None:
        assert _clamp_step(1.0, 1.1) == pytest.approx(1.1)

    def test_large_jump_is_capped_to_max_relative_step(self) -> None:
        # proposed=2.0 from current=1.0 is a +100% jump; default cap is 30%.
        assert _clamp_step(1.0, 2.0) == pytest.approx(1.3)

    def test_large_drop_is_capped_symmetrically(self) -> None:
        assert _clamp_step(1.0, 0.1) == pytest.approx(0.7)


def _stage_exec(stage: str, duration_ms: float, decision: str = "miss") -> None:
    emit(StageExec(
        stage=stage, program_id="p", decision=decision, duration_ms=duration_ms,
    ))


class TestNonLlmStageDurationCountsTowardPrediction:
    @pytest.mark.asyncio
    async def test_validator_time_raises_predicted_duration(self) -> None:
        """A task that's all validator time and ~zero LLM latency used to
        predict ~0s duration (only LLM_CALL latency was tracked). It should
        now reflect the real, dominant non-LLM cost."""
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)

        _call("A", 100, 1.0)  # near-zero LLM latency
        _stage_exec("CallValidatorFunction", 5000.0)  # 5s of real validator work
        await _fire(hook)

        # 10 mutants x ~5s/mutant validator time, /max_in_flight=1
        assert pred.predicted_duration_s > 40.0

    @pytest.mark.asyncio
    async def test_cache_hit_is_not_counted(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)

        _call("A", 100, 1.0)
        _stage_exec("CallValidatorFunction", 5000.0, decision="hit")
        await _fire(hook)

        assert pred.predicted_duration_s < 1.0

    @pytest.mark.asyncio
    async def test_untracked_stage_name_is_ignored(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)

        _call("A", 100, 1.0)
        _stage_exec("SomeUnrelatedStage", 5000.0)
        await _fire(hook)

        assert pred.predicted_duration_s < 1.0
