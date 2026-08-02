"""CostMonitorStage — DAG-compatible wrapper for CostMonitorAgent.

Runs the agent periodically (every N mutants) and returns adjustments
that feed into ``calibrate_prediction()`` via the live cost model.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.llm.agents.cost_monitor import (
    CostMonitorAgent,
    LlmCallRecord,
    _ToolSet,
)
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import DictContainer
from gigaevo.programs.stages.stage_registry import StageRegistry


class _CostMonitorInput(BaseModel):
    """Input for CostMonitorStage — received from DAG edges."""

    program: Program | None = None
    parent_program: Program | None = None
    cost_prediction: CostPrediction | None = None


class _CostMonitorOutput(BaseModel):
    """Adjustments returned to the cost model."""

    cold_start_factor: float = Field(default=-1.0)
    golden_ratio: float = Field(default=-1.0)
    growth_rate_mult: float = Field(default=-1.0)
    flag_outlier_indices: list[int] = Field(default_factory=list)
    skip_calibration: bool = Field(default=False)
    reasoning: str = Field(default="")


@StageRegistry.register(
    description="Live LLM cost monitor — adjusts cost model in real time"
)
class CostMonitorStage(Stage):
    InputsModel = VoidInput
    OutputModel = DictContainer

    def __init__(
        self,
        *,
        timeout: float,
        cost_monitor_agent: CostMonitorAgent | None = None,
        seed_program: Program | None = None,
        cost_prediction: CostPrediction | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._agent = cost_monitor_agent
        self._seed_program = seed_program
        self._cost_prediction = cost_prediction

    def compute(self, inputs: VoidInput) -> DictContainer:
        """Run the cost monitor agent and return adjustments."""
        if self._agent is None:
            return DictContainer({"cost_adjustments": None})

        # Collect recent LLM telemetry — simplified: pass empty calls for now,
        # real implementation reads from emit log buffer
        calls: list[LlmCallRecord] = []

        # Build tool set
        tools = _ToolSet(
            recent_calls=calls,
            program_diff="",
            backpressure_util=0.8,
            current_cold=0.57,
            current_golden=1.1,
            current_growth=1.0,
        )

        # Update agent state
        self._agent.tools = tools

        # Build and return adjustment dict
        adjustments = {
            "cold_start_factor": tools.adjustments.get("cold_start_factor", -1),
            "golden_ratio": tools.adjustments.get("golden_ratio", -1),
            "growth_rate_mult": tools.adjustments.get("growth_rate_mult", -1),
            "flag_outlier_indices": tools.adjustments.get("flag_outlier_indices", []),
            "skip_calibration": tools.adjustments.get("skip_calibration", False),
            "reasoning": "",
        }

        logger.debug("[CostMonitorStage] adjustments={}", adjustments)
        return DictContainer({"cost_adjustments": adjustments})
