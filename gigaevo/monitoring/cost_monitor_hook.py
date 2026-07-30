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
from gigaevo.monitoring.events import CostAgentAdjustment, LLMCall
from gigaevo.monitoring.growth_estimator import (
    RobustPowerLaw,
    estimate_by_stage,
    estimate_duration_by_stage,
)


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
        # Per-stage growth-law state. The hook fires once per accepted
        # mutant (see ingestor.py), so "calls since the last flush" IS one
        # mutant's calls — no program_id join needed. One point per stage
        # per mutant.
        self._history_flush_idx = 0
        self._tokens_by_stage: dict[str, list[float]] = {}
        self._tokens_out_by_stage: dict[str, list[float]] = {}
        self._latency_by_stage: dict[str, list[float]] = {}
        subscribe(LLMCall.event, self._on_llm_call)

    def _on_llm_call(self, event: LLMCall) -> None:
        """Live subscriber — feeds every real LLM call into the history."""
        self.add_llm_call(event.tokens_in, event.tokens_out, event.latency_ms, event.stage)

    def _flush_mutant_bucket(self) -> None:
        """Sum calls since the last flush per stage, refit the growth law,
        and write the new estimate into the shared CostPrediction.

        Runs on EVERY accepted mutant, not gated by ``interval`` — only the
        LLM-agent calibration call below is throttled to every N mutants.
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

        # Extrapolate each stage's total firing count from its observed
        # rate-per-mutant so far (some stages skip-cascade and don't fire
        # on every mutant — see lineage_memory_pipeline.py archive gating).
        total_units_by_stage = {
            stage: max(1, round(self._pred.max_mutants * len(series) / max(self._counter, 1)))
            for stage, series in self._tokens_by_stage.items()
        }
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
        )
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

        # Build tool context
        tools = _ToolSet(
            recent_calls=self._call_history[-20:],
            backpressure_util=0.8,
            current_cold=0.57,
            current_golden=1.1,
            current_growth=1.0,
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
                    self._pred.llm_golden_override = golden
                if growth > 0:
                    self._pred.llm_growth_override = growth
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