from __future__ import annotations

import loguru
from pydantic import BaseModel, Field, TypeAdapter

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import DictContainer
from gigaevo.programs.stages.stage_registry import StageRegistry

SYSTEM = (
    "You are a complexity analyst for an evolutionary code system.  Given "
    "a seed program and the system prompt describing the problem, assess the "
    "expected evolutionary difficulty.\n\n"
)

USER_TEMPLATE = (
    "Seed code:\n{seed}\n\n"
    "System prompt (problem description):\n{system_prompt}\n\n"
    "Return a JSON object with:\n"
    '{"complexity": <int 0-10>, "growth_rate": <"slow"|"medium"|"fast">, '
    '"bottleneck_hint": <short string>}'
)


class _CostAssessmentOutput(BaseModel):
    complexity: int = Field(default=5, ge=0, le=10)
    growth_rate: str = Field(default="medium")
    bottleneck_hint: str = Field(default="unknown")


PARSER = TypeAdapter(_CostAssessmentOutput)


@StageRegistry.register(description="Pre-flight LLM cost assessment")
class CostAssessmentStage(Stage):
    InputsModel = VoidInput
    OutputModel = DictContainer

    def __init__(
        self,
        *,
        timeout: float,
        llm: LangGraphAgent | None = None,
        seed_program: Program | None = None,
        system_prompt: str = "",
    ) -> None:
        super().__init__(timeout=timeout)
        self._seed_program = seed_program
        self._system_prompt = system_prompt
        self._llm = llm
        self._cached: DictContainer | None = None

    async def compute(self, program: Program) -> DictContainer:
        if self._cached is not None:
            return self._cached

        default = DictContainer(
            data={
                "complexity": "5",
                "growth_rate": "medium",
                "bottleneck_hint": "seed not available",
            }
        )

        if self._seed_program is None or self._llm is None:
            self._cached = default
            return default

        seed_code = self._seed_program.code
        user_msg = USER_TEMPLATE.format(
            system_prompt=(self._system_prompt[:2000] or "(none)"),
            seed=seed_code[:2000],
        )

        try:
            result = await self._llm.arun(user_msg)
            raw = result if isinstance(result, str) else str(result)
            # Try to find JSON in LLM output
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                obj = PARSER.validate_json(raw[start:end])
                # Override defaults from LLM response
                if (
                    obj.complexity == 5
                    and obj.growth_rate == "medium"
                    and obj.bottleneck_hint == "unknown"
                ):
                    # Likely not a real response, use defaults
                    data = {
                        "complexity": "5",
                        "growth_rate": "medium",
                        "bottleneck_hint": "parse_error",
                    }
                else:
                    data = {
                        "complexity": str(obj.complexity),
                        "growth_rate": obj.growth_rate,
                        "bottleneck_hint": obj.bottleneck_hint,
                    }
            else:
                data = {
                    "complexity": "5",
                    "growth_rate": "medium",
                    "bottleneck_hint": "parse_error",
                }
        except Exception:
            loguru.logger.opt(exception=True).debug(
                "[{}] cost assessment failed", self.stage_name
            )
            data = {
                "complexity": "5",
                "growth_rate": "medium",
                "bottleneck_hint": "exception",
            }

        self._cached = DictContainer(data=data)
        return self._cached
