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


# Freeze the clock so the anchored estimate's elapsed term is exactly 0 and
# the override multipliers are the only thing moving the number.
_FROZEN = (lambda: 0.0)


class TestLlmOverrideAppliesToDuration:
    @pytest.mark.asyncio
    async def test_golden_override_scales_predicted_duration(self) -> None:
        baseline_pred = CostPrediction(max_mutants=10, max_in_flight=1)
        baseline_hook = CostMonitorHook(agent=None, prediction=baseline_pred, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(baseline_hook)

        overridden_pred = CostPrediction(max_mutants=10, max_in_flight=1)
        overridden_pred.llm_golden_override = 1.5
        overridden_hook = CostMonitorHook(agent=None, prediction=overridden_pred, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(overridden_hook)

        assert overridden_pred.predicted_duration_s == pytest.approx(
            baseline_pred.predicted_duration_s * 1.5
        )

    @pytest.mark.asyncio
    async def test_growth_override_compounds_with_golden_override(self) -> None:
        baseline_pred = CostPrediction(max_mutants=10, max_in_flight=1)
        baseline_hook = CostMonitorHook(agent=None, prediction=baseline_pred, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(baseline_hook)

        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        pred.llm_golden_override = 1.2
        pred.llm_growth_override = 2.0
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(hook)

        assert pred.predicted_duration_s == pytest.approx(
            baseline_pred.predicted_duration_s * 1.2 * 2.0
        )

    @pytest.mark.asyncio
    async def test_out_of_range_override_is_ignored(self) -> None:
        baseline_pred = CostPrediction(max_mutants=10, max_in_flight=1)
        baseline_hook = CostMonitorHook(agent=None, prediction=baseline_pred, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(baseline_hook)

        pred = CostPrediction(max_mutants=10, max_in_flight=1)  # defaults: -1.0 (no override)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000, clock=_FROZEN)
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
        # warmup_attempts=0 -> the first flush already trips the trigger, so
        # the post_step_hook path runs the agent without needing a real
        # confidence-interval breach.
        hook = CostMonitorHook(agent=agent, prediction=pred, interval=1, warmup_attempts=0)

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
        # warmup_attempts=0 -> the first flush already trips the trigger, so
        # the post_step_hook path runs the agent without needing a real
        # confidence-interval breach.
        hook = CostMonitorHook(agent=agent, prediction=pred, interval=1, warmup_attempts=0)

        _call("A", 100, 10.0)
        await _fire(hook)

        assert "golden=1.60" in agent.seen_tools.get_model_params()

    @pytest.mark.asyncio
    async def test_agent_sees_measured_concurrency_and_progress(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=8)
        agent = FakeAgent()
        # warmup_attempts=0 -> the first flush already trips the trigger, so
        # the post_step_hook path runs the agent without needing a real
        # confidence-interval breach.
        hook = CostMonitorHook(agent=agent, prediction=pred, interval=1, warmup_attempts=0)

        _call("A", 100, 10.0)
        await _fire(hook)

        progress = agent.seen_tools.get_progress()
        assert "achieved_concurrency=" in progress
        assert "max_in_flight=8" in progress


class TestConcurrencyLever:
    @pytest.mark.asyncio
    async def test_concurrency_override_scales_the_divisor_not_the_work(self) -> None:
        """Doubling the concurrency the agent expects should halve the
        predicted remaining time — the opposite direction to golden_ratio."""
        baseline = CostPrediction(max_mutants=10, max_in_flight=1)
        baseline_hook = CostMonitorHook(
            agent=None, prediction=baseline, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(baseline_hook)

        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        pred.llm_concurrency_override = 2.0
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(hook)

        assert pred.predicted_duration_s == pytest.approx(
            baseline.predicted_duration_s / 2.0
        )

    @pytest.mark.asyncio
    async def test_out_of_range_concurrency_override_is_ignored(self) -> None:
        baseline = CostPrediction(max_mutants=10, max_in_flight=1)
        baseline_hook = CostMonitorHook(
            agent=None, prediction=baseline, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(baseline_hook)

        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        pred.llm_concurrency_override = 99.0
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000, clock=_FROZEN)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(hook)

        assert pred.predicted_duration_s == pytest.approx(baseline.predicted_duration_s)


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


class TestSurpriseTrigger:
    """The agent is woken by the estimate breaching its own confidence
    interval, not by a fixed tick — see CostMonitorHook._check_agent_trigger."""

    def _hook(self, **kw):
        pred = CostPrediction(max_mutants=100, max_in_flight=8)
        defaults = dict(agent=None, prediction=pred, interval=1000, clock=_FROZEN,
                        warmup_attempts=10**9, cooldown_attempts=0)
        return CostMonitorHook(**{**defaults, **kw, "prediction": pred}), pred

    @pytest.mark.asyncio
    async def test_estimate_inside_previous_interval_does_not_wake_the_agent(self) -> None:
        hook, _ = self._hook()
        for _ in range(8):  # let the estimate settle; early flushes legitimately jump
            _call("A", 100, 1000.0, tokens_out=40)
            await _fire(hook)
        hook._agent_due = False
        _call("A", 100, 1000.0, tokens_out=40)  # nothing new happened
        await _fire(hook)
        assert hook._agent_due is False

    @pytest.mark.asyncio
    async def test_estimate_breaching_previous_interval_wakes_the_agent(self) -> None:
        hook, _ = self._hook()
        for _ in range(3):
            _call("A", 100, 1000.0, tokens_out=40)
            await _fire(hook)
        hook._agent_due = False
        # A call an order of magnitude bigger moves the estimate far outside
        # the interval the previous flush published.
        _call("A", 100_000, 900_000.0, tokens_out=40_000)
        await _fire(hook)
        assert hook._agent_due is True
        assert "interval" in hook._trigger_reason

    @pytest.mark.asyncio
    async def test_no_wakeup_once_almost_nothing_is_left_to_correct(self) -> None:
        # Clock jumps far past the predicted duration: the run is effectively
        # over, so the agent's multiplier would scale an empty tail.
        now = [0.0]
        hook, _ = self._hook(warmup_attempts=0, clock=lambda: now[0])
        now[0] = 10_000.0
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(hook)
        assert hook._agent_due is False

    @pytest.mark.asyncio
    async def test_budget_caps_total_wakeups(self) -> None:
        hook, _ = self._hook(warmup_attempts=0, max_agent_calls=0)
        _call("A", 100, 1000.0, tokens_out=40)
        await _fire(hook)
        assert hook._agent_due is False


class TestOutlierFlagHasTeeth:
    @pytest.mark.asyncio
    async def test_flagged_call_stops_shaping_the_extrapolated_tail(self) -> None:
        pred = CostPrediction(max_mutants=100, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000, clock=_FROZEN)
        for _ in range(4):
            _call("A", 100, 100.0, tokens_out=20)
            await _fire(hook)
        spike_index = len(hook._call_history)
        _call("A", 20_000, 100.0, tokens_out=20)  # one huge prompt
        await _fire(hook)
        with_spike = pred.predicted_total_tokens

        hook._flag_outliers([spike_index])
        _call("A", 100, 100.0, tokens_out=20)
        await _fire(hook)
        after_flag = pred.predicted_total_tokens

        # The spike's own tokens are still counted (they were really spent),
        # but the projection for the remaining ~95 buckets drops sharply.
        assert after_flag < with_spike
        assert after_flag >= sum(hook._tokens_by_stage["A"])

    @pytest.mark.asyncio
    async def test_unknown_call_index_is_ignored(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)
        hook._flag_outliers([999])  # never seen
        assert hook._flagged_buckets == {}


class TestAgentFeedback:
    @pytest.mark.asyncio
    async def test_agent_is_told_what_its_last_decision_did(self) -> None:
        pred = CostPrediction(max_mutants=100, max_in_flight=8)
        agent = FakeAgent()
        hook = CostMonitorHook(agent=agent, prediction=pred, interval=1, warmup_attempts=0)
        hook._last_adjustment = {
            "attempt": 12, "golden": 1.2, "growth": -1.0, "concurrency": -1.0,
            "outliers": 1, "reasoning": "sustained latency rise",
            "predicted_duration_s": 900.0, "elapsed_s": 100.0,
        }
        _call("A", 100, 10.0)
        await _fire(hook)

        outcome = agent.seen_tools.get_last_adjustment_outcome()
        assert "golden=1.20" in outcome
        assert "sustained latency rise" in outcome

    @pytest.mark.asyncio
    async def test_no_previous_decision_reads_cleanly(self) -> None:
        pred = CostPrediction(max_mutants=100, max_in_flight=8)
        agent = FakeAgent()
        hook = CostMonitorHook(agent=agent, prediction=pred, interval=1, warmup_attempts=0)
        _call("A", 100, 10.0)
        await _fire(hook)
        assert "no previous adjustment" in agent.seen_tools.get_last_adjustment_outcome()
