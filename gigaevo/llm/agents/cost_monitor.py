"""CostMonitorAgent — LLM-powered cost model adjuster.

Runs periodically (every N mutants) and analyses recent LLM telemetry
to decide whether token/latency spikes are transient or indicative of a
regime change.  Returns structured adjustments that feed into
``calibrate_prediction()``.

Tool set
--------
get_recent_calls(n)          — last N LLM_CALL records (tokens, latency, stage)
get_program_diff(program_id) — diff of the mutated program vs its parent
get_backpressure()           — current pipeline backpressure snapshot
get_model_params()           — current cost model state (cold_factor, golden, …)
adjust_model(params)         — apply corrections to the live cost model
flag_as_outlier(idx)         — mark a specific call as transient (exclude from rate)

"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent


# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a **Cost Monitor Agent** for an evolutionary code optimisation "
    "pipeline.  Every few mutations you receive a snapshot of the last N "
    "LLM calls (tokens, latency, stage) plus the current backpressure "
    "state.  Your job: decide whether any cost deviations are **transient** "
    "(single outlier — ignore) or a **regime change** (program grew, model "
    "switched, GPU overloaded — adjust the cost model).\n\n"
    "## Available actions (via JSON output)\n"
    "- `cold_start_factor`: float 0.3–1.0 — override warmup multiplier\n"
    "- `golden_ratio`: float 0.8–2.0 — safety margin for GPU variance\n"
    "- `growth_rate_mult`: float 0.5–3.0 — adjust program growth rate γ\n"
    "- `flag_outlier_indices`: list[int] — mark these call indices as outliers\n"
    "- `skip_calibration`: bool — skip the next scheduled calibration\n"
    "- `reasoning`: string ≤ 80 chars — why you made these decisions\n\n"
    "## Decision rules\n"
    "1. Single-call spike (one call 3× median, rest normal) → flag outlier\n"
    "2. Sustained growth (last 5 calls trending up) → raise growth_rate_mult\n"
    "3. Backpressure > 80 % → raise golden_ratio by 0.1\n"
    "4. Backpressure < 20 % → lower golden_ratio by 0.05\n"
    "5. Program length grew > 20 % vs baseline → raise growth_rate_mult by 0.2\n"
    "6. Latency spike with normal tokens → GPU contention, raise golden_ratio\n"
    "7. Everything normal → no changes (all defaults)\n"
)


# ── Input / Output schemas ─────────────────────────────────────────────────

@dataclass
class LlmCallRecord:
    index: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    stage: str

    def to_line(self) -> str:
        return (
            f"#{self.index} {self.stage} IN={self.tokens_in} OUT={self.tokens_out} "
            f"lat={self.latency_ms:.0f}ms"
        )


class _AdjustmentOutput(BaseModel):
    cold_start_factor: float = Field(default=-1.0, description="Override cold_start (−1 = no change, 0.3–1.0)")
    golden_ratio: float = Field(default=-1.0, description="Override golden_ratio (−1 = no change, 0.8–2.0)")
    growth_rate_mult: float = Field(default=-1.0, description="Multiply growth_rate γ (−1 = no change, 0.5–3.0)")
    flag_outlier_indices: list[int] = Field(default_factory=list, description="Indices of calls to ignore")
    skip_calibration: bool = Field(default=False)
    reasoning: str = Field(default="")


# ── Tool implementations (reuse gigaevo-core infrastructure) ────────────────

class _ToolSet:
    """In-memory tools used during a single agent call."""

    def __init__(
        self,
        recent_calls: list[LlmCallRecord],
        program_diff: str = "",
        backpressure_util: float = 0.8,
        in_flight: int = 0,
        max_in_flight: int = 8,
        current_cold: float = 0.57,
        current_golden: float = 1.1,
        current_growth: float = 1.0,
        baseline_program_len: int = 0,
        current_program_len: int = 0,
    ):
        self._calls = recent_calls
        self._diff = program_diff
        self._util = backpressure_util
        self._in_flight = in_flight
        self._max_flight = max_in_flight
        self._cold = current_cold
        self._golden = current_golden
        self._growth = current_growth
        self._baseline_len = baseline_program_len
        self._current_len = current_program_len
        # Accumulated adjustments
        self.adjustments: dict[str, Any] = {}

    # ── Observability tools ────────────────────────────────────────────────

    def get_recent_calls(self, n: int = 10) -> str:
        """Return the last N LLM calls as a compact table."""
        calls = self._calls[-n:]
        if not calls:
            return "(no LLM calls yet)"
        lines = [f"Last {len(calls)} LLM calls:"]
        tokens = [c.tokens_in + c.tokens_out for c in calls]
        median_tok = _median(tokens)
        for c in calls:
            marker = ""
            if (c.tokens_in + c.tokens_out) > median_tok * 2.5:
                marker = " ← OUTLIER"
            lines.append(c.to_line() + marker)
        lines.append(f"Median tokens: {median_tok}")
        return "\n".join(lines)

    def get_program_diff(self) -> str:
        """Return the diff of the current program vs its parent."""
        return self._diff or "(no program diff available)"

    def get_backpressure(self) -> str:
        """Return current pipeline backpressure state."""
        return (
            f"Pipeline: in_flight={self._in_flight}/{self._max_flight} "
            f"(util={self._util:.0%})"
        )

    def get_model_params(self) -> str:
        """Return current cost model parameters."""
        return (
            f"cold_start={self._cold:.2f} golden={self._golden:.2f} "
            f"growth×={self._growth:.2f} "
            f"program_len: {self._baseline_len}→{self._current_len}"
        )

    # ── Action tools — accumulate adjustments ──────────────────────────────

    def adjust_model(
        self,
        cold_start_factor: float = -1,
        golden_ratio: float = -1,
        growth_rate_mult: float = -1,
    ) -> str:
        """Adjust cost model parameters. −1 means 'no change'."""
        changes = []
        if 0.2 <= cold_start_factor <= 1.0:
            self.adjustments["cold_start_factor"] = cold_start_factor
            changes.append(f"cold={cold_start_factor:.2f}")
        if 0.5 <= golden_ratio <= 2.0:
            self.adjustments["golden_ratio"] = golden_ratio
            changes.append(f"golden={golden_ratio:.2f}")
        if 0.3 <= growth_rate_mult <= 3.0:
            self.adjustments["growth_rate_mult"] = growth_rate_mult
            changes.append(f"growth×={growth_rate_mult:.2f}")
        return (
            f"Adjusted: {', '.join(changes)}" if changes
            else "No adjustments made (values out of range)"
        )

    def flag_as_outlier(self, index: int) -> str:
        """Mark a specific call index as a transient outlier."""
        current: list = self.adjustments.setdefault("flag_outlier_indices", [])
        current.append(index)
        return f"Call #{index} flagged as outlier"

    def skip_next_calibration(self) -> str:
        """Skip the next scheduled calibration point."""
        self.adjustments["skip_calibration"] = True
        return "Next calibration will be skipped"


def _median(lst: list[int | float]) -> float:
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ── Tool schema for the LLM prompt ──────────────────────────────────────────

TOOL_SCHEMA = """Available tools — call via tool name with arguments:

get_recent_calls(n=10) → table of last N LLM calls
get_program_diff()     → diff of current vs parent program
get_backpressure()     → pipeline in_flight/util %
get_model_params()     → current cold/golden/growth/program_len

Action tools:
adjust_model(cold_start_factor=0.6, golden_ratio=1.15, growth_rate_mult=1.2)
flag_as_outlier(index=3)
skip_next_calibration()
"""


# ── Agent ───────────────────────────────────────────────────────────────────

class CostMonitorAgent(LangGraphAgent):
    """LLM agent that analyses cost telemetry and adjusts the cost model."""

    def __init__(
        self,
        llm,
        *,
        tools: _ToolSet | None = None,
    ):
        super().__init__(llm)
        self._tools = tools or _ToolSet([])

    @property
    def tools(self) -> _ToolSet:
        return self._tools

    @tools.setter
    def tools(self, ts: _ToolSet) -> None:
        self._tools = ts

    def build_prompt(self, state: dict[str, Any]) -> list[BaseMessage]:
        tools = self._tools

        # Build rich context — embed tool outputs inline
        calls_table = tools.get_recent_calls(15)
        bp = tools.get_backpressure()
        params = tools.get_model_params()
        diff = tools.get_program_diff()

        context = (
            f"{calls_table}\n\n"
            f"{bp}\n"
            f"{params}\n\n"
            f"Program diff:\n{diff[:2000] if diff else '(none)'}"
        )

        user = HumanMessage(content=(
            f"{context}\n\n"
            "Analyse the telemetry above.  If adjustments are needed, respond "
            "with a JSON object:\n"
            '{"cold_start_factor": float, "golden_ratio": float, '
            '"growth_rate_mult": float, "flag_outlier_indices": [int], '
            '"skip_calibration": bool, "reasoning": "..."}\n\n'
            "Use -1 for any parameter you do NOT want to change."
        ))
        return [SystemMessage(content=SYSTEM_PROMPT), user]

    def parse_response(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply LLM decisions to the tool set."""
        response = state.get("llm_response")
        if response is None:
            return state

        content = getattr(response, "content", "")
        try:
            # Strip code fences if present
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
        except (json.JSONDecodeError, Exception):
            return state

        # Apply adjustments
        tools = self._tools
        cold = data.get("cold_start_factor", -1)
        golden = data.get("golden_ratio", -1)
        growth = data.get("growth_rate_mult", -1)
        outliers = data.get("flag_outlier_indices", [])
        skip = data.get("skip_calibration", False)
        reasoning = data.get("reasoning", "")

        tools.adjust_model(cold_start_factor=cold, golden_ratio=golden, growth_rate_mult=growth)

        for idx in outliers:
            tools.flag_as_outlier(idx)

        if skip:
            tools.skip_next_calibration()

        state["cost_adjustments"] = {
            "cold_start_factor": tools.adjustments.get("cold_start_factor", -1),
            "golden_ratio": tools.adjustments.get("golden_ratio", -1),
            "growth_rate_mult": tools.adjustments.get("growth_rate_mult", -1),
            "flag_outlier_indices": tools.adjustments.get("flag_outlier_indices", []),
            "skip_calibration": tools.adjustments.get("skip_calibration", False),
            "reasoning": reasoning,
        }
        return state