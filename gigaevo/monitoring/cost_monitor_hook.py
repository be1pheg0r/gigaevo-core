"""Cost Monitor Hook — call CostMonitorAgent every N mutants.

Designed as a post-step hook for the evolution engine.
"""
from __future__ import annotations

import json

from loguru import logger

from gigaevo.llm.agents.cost_monitor import (
    CostMonitorAgent,
    LlmCallRecord,
    _ToolSet,
)
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.monitoring.emit import emit, subscribe
from gigaevo.monitoring.events import BackpressureSample, CostAgentAdjustment, LLMCall, MutationAttempted, StageExec
from gigaevo.monitoring.growth_estimator import (
    RobustPowerLaw,
    estimate_by_stage,
    estimate_duration_by_stage,
)


def _clamp_step(current: float, proposed: float, max_rel_step: float = 0.3) -> float:
    """Limit how far one calibration call can move a multiplier relative to
    its current value — caps the blast radius of a single (possibly wrong)
    LLM judgment call, instead of letting it jump anywhere in the full
    accepted range (0.5–2.0 golden / 0.3–3.0 growth) in one step."""
    lo, hi = current * (1 - max_rel_step), current * (1 + max_rel_step)
    return max(lo, min(hi, proposed))


# Non-LLM pipeline stages whose wall-clock time the LLM-latency duration
# model is otherwise blind to. Named explicitly (rather than "everything
# that isn't an LLM stage") so the growth-law fit only sees stages known to
# be real, recurring per-mutant cost — see estimate_duration_by_stage's
# nonllm_duration_by_stage docstring for the historical evidence.
NON_LLM_DURATION_STAGES = frozenset({
    "CallProgramFunction", "CallValidatorFunction", "IntraMemoryStage",
})


class CostMonitorHook:
    """Runs CostMonitorAgent every N mutants, feeds results into CostPrediction."""

    def __init__(
        self,
        agent: CostMonitorAgent,
        prediction: CostPrediction,
        interval: int = 5,
    ):
        self._agent = agent
        self._pred = prediction
        self._interval = interval
        self._counter = 0
        self._call_history: list[LlmCallRecord] = []
        # Per-stage growth-law state. The hook flushes once per accepted
        # mutant (post_step_hook, see ingestor.py) AND once per mutation
        # attempt (MUTATION_ATTEMPTED, see mutant_task.py) — the latter
        # guarantees at least one estimate appears even for low/zero-accept
        # domains, where post_step_hook (gated on an ACCEPT landing) may
        # never fire within a short/small run. ``_history_flush_idx``
        # dedupes: whichever trigger fires first drains ``_call_history``,
        # the other is then a no-op for that batch of calls.
        self._history_flush_idx = 0
        self._tokens_by_stage: dict[str, list[float]] = {}
        self._tokens_out_by_stage: dict[str, list[float]] = {}
        self._latency_by_stage: dict[str, list[float]] = {}
        # Mutation attempts observed so far (MUTATION_ATTEMPTED fires per DAG
        # dispatch, before accept/reject). ``max_mutants`` caps ATTEMPTS, not
        # accepted mutants (MaxMutantsStopper watches
        # engine.metrics.mutations_created) — since flushes now track
        # attempts 1:1, this is the correct denominator for projecting each
        # stage's total firing count, no accept-rate correction needed.
        self._attempts = 0
        subscribe(LLMCall.event, self._on_llm_call)
        subscribe(MutationAttempted.event, self._on_mutation_attempted)
        subscribe(BackpressureSample.event, self._on_backpressure_sample)
        subscribe(StageExec.event, self._on_stage_exec)
        self._last_bp: BackpressureSample | None = None
        # Same drain-on-flush pattern as _call_history/_history_flush_idx,
        # but for non-LLM stage wall-clock time (see NON_LLM_DURATION_STAGES).
        self._stage_exec_history: list[tuple[str, float]] = []
        self._nonllm_flush_idx = 0
        self._nonllm_duration_by_stage: dict[str, list[float]] = {}

    def _on_llm_call(self, event: LLMCall) -> None:
        """Live subscriber — feeds every real LLM call into the history."""
        self.add_llm_call(event.tokens_in, event.tokens_out, event.latency_ms, event.stage)

    def _on_mutation_attempted(self, event: MutationAttempted) -> None:
        self._attempts += 1
        self._flush_mutant_bucket()

    def _on_backpressure_sample(self, event: BackpressureSample) -> None:
        self._last_bp = event

    def _on_stage_exec(self, event: StageExec) -> None:
        # A cache "hit" returns near-instantly and isn't representative of
        # future cost for that stage — only count real executions.
        if event.stage in NON_LLM_DURATION_STAGES and event.decision != "hit":
            self._stage_exec_history.append((event.stage, event.duration_ms))

    def _flush_mutant_bucket(self) -> None:
        """Sum calls since the last flush per stage, refit the growth law,
        and write the new estimate into the shared CostPrediction.

        Runs on every accepted mutant AND every mutation attempt, not
        gated by ``interval`` — only the LLM-agent calibration call in
        ``__call__`` is throttled to every N accepted mutants.
        """
        new_calls = self._call_history[self._history_flush_idx:]
        self._history_flush_idx = len(self._call_history)
        if not new_calls:
            return

        stage_tokens: dict[str, float] = {}
        stage_tokens_out: dict[str, float] = {}
        stage_latency: dict[str, float] = {}
        for rec in new_calls:
            stage_tokens[rec.stage] = stage_tokens.get(rec.stage, 0.0) + rec.tokens_in + rec.tokens_out
            stage_tokens_out[rec.stage] = stage_tokens_out.get(rec.stage, 0.0) + rec.tokens_out
            stage_latency[rec.stage] = stage_latency.get(rec.stage, 0.0) + rec.latency_ms
        for stage, tokens in stage_tokens.items():
            self._tokens_by_stage.setdefault(stage, []).append(tokens)
            self._tokens_out_by_stage.setdefault(stage, []).append(stage_tokens_out[stage])
            self._latency_by_stage.setdefault(stage, []).append(stage_latency[stage])

        new_stage_execs = self._stage_exec_history[self._nonllm_flush_idx:]
        self._nonllm_flush_idx = len(self._stage_exec_history)
        if new_stage_execs:
            nonllm_stage_totals: dict[str, float] = {}
            for stage, duration_ms in new_stage_execs:
                nonllm_stage_totals[stage] = nonllm_stage_totals.get(stage, 0.0) + duration_ms
            for stage, total_ms in nonllm_stage_totals.items():
                self._nonllm_duration_by_stage.setdefault(stage, []).append(total_ms)

        # Extrapolate each stage's total firing count from its observed
        # rate-per-attempt so far (some stages skip-cascade and don't fire
        # on every attempt — see lineage_memory_pipeline.py archive gating).
        # ``self._attempts`` is the flush-cadence denominator (buckets track
        # attempts via ``_on_mutation_attempted`` above), so this projects
        # directly against ``max_mutants`` (attempts cap) with no
        # accept-rate correction needed. Falls back to a denominator of 1
        # when no attempts have been observed yet (e.g. tests that emit
        # LLM_CALL without MUTATION_ATTEMPTED).
        total_units_by_stage = {
            stage: max(1, round(self._pred.max_mutants * len(series) / max(self._attempts, 1)))
            for stage, series in self._tokens_by_stage.items()
        }
        total_units_by_stage.update({
            stage: max(1, round(self._pred.max_mutants * len(series) / max(self._attempts, 1)))
            for stage, series in self._nonllm_duration_by_stage.items()
        })
        est = estimate_by_stage(
            self._tokens_by_stage, self._latency_by_stage,
            total_units_by_stage=total_units_by_stage,
            max_in_flight=self._pred.max_in_flight,
            law_cls=RobustPowerLaw,
        )
        # Duration uses the TTFT+TPOT physical model (latency ~ tokens_out),
        # not a growth law over call index — latency doesn't follow a growth
        # trend in this system (see estimate_duration_by_stage docstring).
        duration_s, duration_ci = estimate_duration_by_stage(
            self._tokens_out_by_stage, self._latency_by_stage,
            total_units_by_stage=total_units_by_stage,
            max_in_flight=self._pred.max_in_flight,
            nonllm_duration_by_stage=self._nonllm_duration_by_stage,
        )
        # Apply the CostMonitorAgent's live overrides (Layer 4) to the
        # growth-law duration estimate: golden_ratio is a safety margin,
        # growth_rate_mult reacts to a sustained size/latency trend the
        # agent detected. Sentinel -1 (out of range) means no change.
        llm_mult = 1.0
        if 0.5 <= self._pred.llm_golden_override <= 2.0:
            llm_mult *= self._pred.llm_golden_override
        if 0.3 <= self._pred.llm_growth_override <= 3.0:
            llm_mult *= self._pred.llm_growth_override
        if llm_mult != 1.0:
            duration_s *= llm_mult
            duration_ci = (duration_ci[0] * llm_mult, duration_ci[1] * llm_mult)
        self._pred.predicted_total_tokens = int(est.predicted_total_tokens)
        self._pred.predicted_duration_s = duration_s
        self._pred.token_ci_low, self._pred.token_ci_high = (
            int(est.tokens_ci[0]), int(est.tokens_ci[1])
        )
        self._pred.ci_low_s, self._pred.ci_high_s = duration_ci
        logger.info("[CostMonitorHook] mutant={} {}", self._counter, self._pred._log_estimate())
        # Compact single-line JSON alongside the human-readable block above —
        # consumers (e.g. tools/task_builder_web) parse this instead of the
        # multi-line text, which is fragile to regex across interleaved logs.
        logger.info(
            "[CostMonitorHookJSON] {}",
            json.dumps({
                "mutant": self._counter,
                "predicted_tokens": self._pred.predicted_total_tokens,
                "token_ci_low": self._pred.token_ci_low,
                "token_ci_high": self._pred.token_ci_high,
                "predicted_duration_s": self._pred.predicted_duration_s,
                "ci_low_s": self._pred.ci_low_s,
                "ci_high_s": self._pred.ci_high_s,
            }),
        )

    async def __call__(self) -> None:
        self._counter += 1
        self._flush_mutant_bucket()
        if self._counter % self._interval != 0:
            return

        # Build tool context from real live state, not fixed placeholders —
        # previously the agent always saw cold=0.57/golden=1.1/growth=1.0 and
        # backpressure_util=0.8 regardless of what was actually happening, so
        # its transient-vs-regime-change judgment was made half-blind.
        current_golden = self._pred.llm_golden_override if self._pred.llm_golden_override > 0 else 1.0
        current_growth = self._pred.llm_growth_override if self._pred.llm_growth_override > 0 else 1.0
        current_cold = self._pred.llm_cold_override if self._pred.llm_cold_override > 0 else 0.57
        if self._last_bp is not None:
            bp_util = self._last_bp.in_flight / self._last_bp.max_in_flight
            bp_in_flight, bp_max_in_flight = self._last_bp.in_flight, self._last_bp.max_in_flight
        else:
            bp_util, bp_in_flight, bp_max_in_flight = 0.8, 0, 8
        tools = _ToolSet(
            recent_calls=self._call_history[-20:],
            backpressure_util=bp_util,
            in_flight=bp_in_flight,
            max_in_flight=bp_max_in_flight,
            current_cold=current_cold,
            current_golden=current_golden,
            current_growth=current_growth,
        )
        self._agent.tools = tools

        try:
            # Build messages from the prompt
            messages = self._agent.build_prompt({"messages": []})
            state = {"messages": messages}
            # Run agent — acall_llm from base class
            result = await self._agent.acall_llm(state)
            result = self._agent.parse_response(result)

            adjustments = result.get("cost_adjustments", {})
            if adjustments:
                cold = adjustments.get("cold_start_factor", -1)
                golden = adjustments.get("golden_ratio", -1)
                growth = adjustments.get("growth_rate_mult", -1)
                reasoning = adjustments.get("reasoning", "")

                if cold > 0:
                    self._pred.llm_cold_override = cold
                if golden > 0:
                    self._pred.llm_golden_override = _clamp_step(current_golden, golden)
                if growth > 0:
                    self._pred.llm_growth_override = _clamp_step(current_growth, growth)
                if reasoning:
                    self._pred.llm_reasoning = reasoning

                logger.info("[CostMonitorHook] adjustments: cold={} golden={} growth={} reason={}",
                            cold, golden, growth, reasoning)
                emit(CostAgentAdjustment(
                    mutant_index=self._counter,
                    cold_start_factor=cold,
                    golden_ratio=golden,
                    growth_rate_mult=growth,
                    flag_outlier_indices=adjustments.get("flag_outlier_indices", []),
                    skip_calibration=adjustments.get("skip_calibration", False),
                    reasoning=reasoning,
                ))
        except Exception as e:
            logger.warning("[CostMonitorHook] agent call failed: {}", e)

    def add_llm_call(self, tokens_in: int, tokens_out: int, latency_ms: float, stage: str) -> None:
        """Record an LLM call for the agent to analyse."""
        self._call_history.append(LlmCallRecord(
            index=len(self._call_history),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            stage=stage,
        ))