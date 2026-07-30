"""Cost Monitor Hook — call CostMonitorAgent every N mutants.

Designed as a post-step hook for the evolution engine.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from gigaevo.llm.agents.cost_monitor import (
    CostMonitorAgent,
    LlmCallRecord,
    _ToolSet,
)
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.monitoring.emit import emit, subscribe
from gigaevo.monitoring.events import CostAgentAdjustment, LLMCall


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
        subscribe(LLMCall.event, self._on_llm_call)

    def _on_llm_call(self, event: LLMCall) -> None:
        """Live subscriber — feeds every real LLM call into the history."""
        self.add_llm_call(event.tokens_in, event.tokens_out, event.latency_ms, event.stage)

    async def __call__(self) -> None:
        self._counter += 1
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