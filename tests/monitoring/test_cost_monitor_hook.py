from __future__ import annotations

import pytest

from gigaevo.monitoring.cost_monitor_hook import CostMonitorHook, _clamp_step
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.monitoring.emit import emit, reset_subscribers
from gigaevo.monitoring.events import BackpressureSample, LLMCall, MutationAttempted, StageExec


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


class TestObserverIsNotItsOwnTelemetry:
    """CostMonitorAgent's own LLM calls must never reach the cost model.

    They go through the same LLM_CALL bus as mutations, and on real
    alphaevolve runs they were 4.7-17.3% of all measured latency with single
    calls 10-15x the run median — i.e. the observer was the biggest outlier
    in its own telemetry, and was seen flagging itself as an anomaly.
    """

    @pytest.mark.asyncio
    async def test_own_calls_are_excluded_from_the_fit(self) -> None:
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)

        _call("MutationAgent", 4800, 30_000.0, tokens_out=1000)
        _call("CostMonitorAgent", 1140, 280_660.0, tokens_out=39)
        _call("MutationAgent", 4800, 30_000.0, tokens_out=1000)
        await _fire(hook)

        assert "CostMonitorAgent" not in hook._tokens_by_stage
        assert "CostMonitorAgent" not in hook._latency_by_stage
        assert hook._tokens_by_stage["MutationAgent"] == [11600.0]   # 2x(4800+1000)
        assert len(hook._call_history) == 2          # the agent call is not indexable
        assert len(hook._call_times) == 2            # nor does it inflate concurrency
        assert hook._observer_calls == 1             # but it is still counted somewhere

    @pytest.mark.asyncio
    async def test_the_agentless_stub_is_excluded_too(self) -> None:
        """Otherwise the ablation compares two different workloads."""
        pred = CostPrediction(max_mutants=10, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)
        _call("NoOpCostMonitorAgent", 500, 1000.0)
        _call("MutationAgent", 100, 10.0)
        await _fire(hook)
        assert set(hook._tokens_by_stage) == {"MutationAgent"}

    @pytest.mark.asyncio
    async def test_own_calls_do_not_get_projected_over_the_run(self) -> None:
        """The stage used to be extrapolated like any recurring per-mutant
        cost: 4 buckets in 22 attempts projected to 18 agent calls over 100,
        against a hard budget of 12 and an actual 5."""
        pred = CostPrediction(max_mutants=100, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)
        for i in range(4):
            _call("MutationAgent", 1000, 1000.0, tokens_out=200)
            _call("CostMonitorAgent", 1140, 280_000.0, tokens_out=39)
            emit(MutationAttempted(mutant_id=f"m{i}"))     # flushes a bucket
        # 4 buckets over 4 attempts -> 100 projected firings of MutationAgent
        # at 1200 tokens each, and none at all of the observer.
        assert hook._attempts == 4
        assert set(hook._tokens_by_stage) == {"MutationAgent"}
        assert pred.predicted_total_tokens == pytest.approx(100 * 1200, rel=0.35)


class TestOneWakeupRunsOnce:
    """The two dispatch paths must not both claim the same trigger."""

    @pytest.mark.asyncio
    async def test_accept_path_does_not_start_a_second_concurrent_call(self) -> None:
        import asyncio

        class SlowAgent(FakeAgent):
            def __init__(self):
                super().__init__()
                self.live = 0
                self.max_live = 0

            async def acall_llm(self, state):
                self.live += 1
                self.max_live = max(self.max_live, self.live)
                await asyncio.sleep(0.02)
                self.live -= 1
                return state

        pred = CostPrediction(max_mutants=100, max_in_flight=8)
        agent = SlowAgent()
        hook = CostMonitorHook(agent=agent, prediction=pred, interval=1,
                               warmup_attempts=1, cooldown_attempts=0)
        _call("MutationAgent", 3800, 30_000.0, tokens_out=1000)
        emit(MutationAttempted(mutant_id="m1"))       # arms + dispatches a task
        await asyncio.sleep(0)                        # task starts, blocks on the LLM
        assert agent.live == 1

        hook._agent_due = True                        # a later flush re-arms it
        _call("MutationAgent", 3800, 30_000.0, tokens_out=1000)
        await asyncio.gather(hook(), hook._agent_task)

        assert agent.max_live == 1, "two agent calls ran at once"

    @pytest.mark.asyncio
    async def test_the_budget_is_charged_once_per_wakeup(self) -> None:
        pred = CostPrediction(max_mutants=100, max_in_flight=8)
        hook = CostMonitorHook(agent=FakeAgent(), prediction=pred, interval=1,
                               warmup_attempts=1, cooldown_attempts=0)
        _call("MutationAgent", 100, 10.0)
        emit(MutationAttempted(mutant_id="m1"))
        if hook._agent_task is not None:
            await hook._agent_task
        await hook()                                   # accept lands right after
        assert hook._agent_calls == 1


class TestOnlineIntervalCalibration:
    """The interval is graded by its own misses and widened/narrowed to hit
    the target miscoverage — Gibbs & Candes ACI. Measured before this, the
    published interval covered the truth 64-68% while aiming at 90%."""

    def _hook(self, **kw):
        pred = CostPrediction(max_mutants=100, max_in_flight=1)
        return CostMonitorHook(agent=None, prediction=pred, interval=1000, **kw)

    @pytest.mark.asyncio
    async def test_repeated_misses_widen_the_interval(self) -> None:
        hook = self._hook(aci_gamma=0.2)
        start = hook._aci_scale
        for _ in range(20):
            hook._update_aci(True)
        assert hook._aci_scale > start * 1.5

    @pytest.mark.asyncio
    async def test_never_missing_narrows_it(self) -> None:
        hook = self._hook(aci_gamma=0.2)
        for _ in range(20):
            hook._update_aci(False)
        assert hook._aci_scale < 1.0

    @pytest.mark.asyncio
    async def test_it_settles_when_misses_arrive_at_the_target_rate(self) -> None:
        hook = self._hook(aci_gamma=0.1, aci_alpha=0.10)
        for i in range(400):
            hook._update_aci(i % 10 == 0)       # exactly 10% miscoverage
        assert 0.8 < hook._aci_scale < 1.25

    @pytest.mark.asyncio
    async def test_gamma_zero_restores_the_model_only_width(self) -> None:
        hook = self._hook(aci_gamma=0.0)
        for _ in range(50):
            hook._update_aci(True)
        assert hook._aci_scale == 1.0

    @pytest.mark.asyncio
    async def test_the_scale_cannot_run_away(self) -> None:
        hook = self._hook(aci_gamma=1.0)
        for _ in range(500):
            hook._update_aci(True)
        assert hook._aci_scale <= 5.0
        for _ in range(1000):
            hook._update_aci(False)
        assert hook._aci_scale >= 0.5

    @pytest.mark.asyncio
    async def test_a_wider_scale_really_reaches_the_published_interval(self) -> None:
        """The scale is only worth anything if it lands in CostPrediction."""
        narrow = self._hook(aci_gamma=0.0)
        for i in range(4):
            _call("A", 1000, 1000.0, tokens_out=200)
            emit(MutationAttempted(mutant_id=f"m{i}"))
        base = narrow._pred.ci_high_s - narrow._pred.ci_low_s

        reset_subscribers()
        wide = self._hook(aci_gamma=0.0)
        wide._aci_scale = 3.0
        for i in range(4):
            _call("A", 1000, 1000.0, tokens_out=200)
            emit(MutationAttempted(mutant_id=f"m{i}"))
        assert (wide._pred.ci_high_s - wide._pred.ci_low_s) > base * 2


class TestLeverGuards:
    def _hook(self):
        pred = CostPrediction(max_mutants=100, max_in_flight=8)
        return CostMonitorHook(agent=None, prediction=pred, interval=1000)

    def test_a_change_inside_the_deadband_is_refused(self) -> None:
        hook = self._hook()
        assert hook._gate_lever("golden_ratio", 1.00, 1.02) is None

    def test_a_real_change_goes_through_clamped(self) -> None:
        hook = self._hook()
        assert hook._gate_lever("golden_ratio", 1.00, 1.20) == pytest.approx(1.20)
        assert hook._gate_lever("growth_rate_mult", 1.00, 5.00) == pytest.approx(1.30)

    def test_reversing_the_last_move_is_taken_at_half_step(self) -> None:
        hook = self._hook()
        hook._agent_calls = 1
        assert hook._gate_lever("golden_ratio", 1.00, 1.40) == pytest.approx(1.30)
        hook._agent_calls = 2
        # full step down would allow 0.91; reversing halves it to 1.105
        assert hook._gate_lever("golden_ratio", 1.30, 0.50) == pytest.approx(1.105)

    def test_the_damping_expires(self) -> None:
        hook = self._hook()
        hook._agent_calls = 1
        hook._gate_lever("golden_ratio", 1.00, 1.40)
        hook._agent_calls = 9                     # far outside REVERSAL_WINDOW
        assert hook._gate_lever("golden_ratio", 1.30, 0.50) == pytest.approx(0.91)

    def test_a_blunt_lever_needs_a_persistence_claim(self) -> None:
        hook = self._hook()
        assert hook._blunt_lever_allowed(7) is True
        assert hook._blunt_lever_allowed(2) is False
        assert hook._blunt_lever_allowed(None) is False
        assert hook._blunt_lever_allowed("lots") is False


class TestBudgetMode:
    """`project_tokens` / `affordable_attempts` — the precompute question:
    what would N attempts cost, and how many attempts does a budget buy."""

    def _hook(self, attempts: int = 10, per_attempt: int = 1000) -> CostMonitorHook:
        pred = CostPrediction(max_mutants=1, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)
        for i in range(attempts):
            _call("A", per_attempt, 100.0, tokens_out=per_attempt // 10)
            emit(MutationAttempted(mutant_id=f"m{i}"))
        return hook

    def test_projection_grows_with_the_horizon(self) -> None:
        hook = self._hook()
        at10 = hook.project_tokens(10)[0]
        at100 = hook.project_tokens(100)[0]
        at200 = hook.project_tokens(200)[0]
        assert at10 < at100 < at200
        # Anchored on what was actually observed: 10 attempts x ~1100 tokens.
        assert at10 == pytest.approx(11_000, rel=0.05)
        # A flat token series extrapolates flat, so 100 attempts is ~10x.
        assert at100 == pytest.approx(110_000, rel=0.2)

    def test_projection_never_dips_below_what_was_spent(self) -> None:
        hook = self._hook()
        spent = hook.project_tokens(10)[0]
        # Asking about a horizon already passed cannot refund tokens.
        assert hook.project_tokens(3)[0] == pytest.approx(spent)

    def test_the_interval_brackets_the_point_estimate(self) -> None:
        lo, hi = self._hook().project_tokens(100)[1]
        point = self._hook().project_tokens(100)[0]
        assert lo <= point <= hi

    def test_budget_buys_a_horizon(self) -> None:
        hook = self._hook()
        n = hook.affordable_attempts(110_000)
        assert 70 <= n <= 140, n
        # More money buys more attempts; the answer is monotone in the budget.
        assert hook.affordable_attempts(220_000) > n

    def test_a_smaller_budget_buys_fewer_attempts(self) -> None:
        # How the console turns one interval into a range: spend the same
        # budget as if the run cost the top / bottom of its projection.
        hook = self._hook()
        point, (lo, hi) = hook.project_tokens(100)
        assert (hook.affordable_attempts(110_000 / (hi / point))
                <= hook.affordable_attempts(110_000)
                <= hook.affordable_attempts(110_000 / (lo / point)))

    def test_a_budget_already_overspent_is_zero_not_negative(self) -> None:
        assert self._hook().affordable_attempts(1.0) == 0

    def test_no_observations_means_no_projection(self) -> None:
        pred = CostPrediction(max_mutants=1, max_in_flight=1)
        hook = CostMonitorHook(agent=None, prediction=pred, interval=1000)
        assert hook.project_tokens(100) == (0.0, (0.0, 0.0))
