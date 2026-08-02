"""CostMonitorAgent — LLM-powered cost model adjuster."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent

SYSTEM_PROMPT = (
    "You are a **Cost Monitor Agent** for an evolutionary code optimisation "
    "pipeline.  A statistical estimator forecasts what the run will cost; it "
    "is good at extrapolating trends and blind to WHY a trend changed.  That "
    "is your entire job: name the CAUSE of the deviation you were woken for. "
    "You are not asked what happened — the estimator already measured that. "
    "You are asked what is behind it.\n\n"
    "## Why you no longer flag outliers\n"
    "Marking single slow calls as outliers used to be your main action (18 of "
    "20 wakeups).  It was measured on 7 recorded runs and it made the forecast "
    "WORSE on every one of them.  Two reasons: the growth law is a Theil-Sen "
    "fit that already tolerates ~29% contaminated points, and the estimator "
    "runs about 25% BELOW the truth at a quarter of the way in — removing the "
    "high calls pushes it further down.  A spike that already happened is real "
    "cost.  Do not try to explain spikes away; explain what is CAUSING them.\n\n"
    "## The one question\n"
    "Pick exactly one `cause`:\n"
    "- `servers` — the infrastructure got slower or faster.  Latency moved "
    "while tokens_out did not; or achieved concurrency fell away from "
    "max_in_flight.  The work did not change, the machine did.\n"
    "- `programs` — the evolved programs themselves got bigger or more "
    "expensive.  tokens_out trending up across recent calls, program length "
    "grown against baseline.  Every remaining call will carry this.\n"
    "- `task` — this problem is structurally costlier than the early calls "
    "suggested (long validation, heavy retries), so the whole remaining run "
    "needs more margin, not a faster/slower trend.\n"
    "- `noise` — nothing durable is happening.  One odd call, or a spike that "
    "has already passed.  **This is the correct answer most of the time.**\n\n"
    "Then give `magnitude`: how much that cause moves the model, as a "
    "multiplier.  >1 means more cost / slower; <1 means less cost / faster. "
    "Use -1 for `noise`.  Each cause drives exactly one lever, chosen for you:\n"
    "- `servers` scales the measured concurrency divisor (0.3–3.0).  Slower "
    "servers = magnitude below 1.\n"
    "- `programs` scales the fitted growth rate of remaining work (0.5–3.0)\n"
    "- `task` scales a flat safety margin on remaining work (0.8–2.0)\n\n"
    "`programs` and `task` are blunt — they apply to everything still ahead — "
    "so they need `sustained_over_calls`: over how many recent calls the "
    "deviation actually held.  Below 5 the change is refused.  `servers` needs "
    "no persistence claim: a contention change is immediate.\n\n"
    "## What you are correcting\n"
    "`predicted_duration = elapsed_so_far + remaining_service_time / "
    "achieved_concurrency`.  `elapsed_so_far` is measured, not guessed — your "
    "lever only ever scales the REMAINING work or the concurrency divisor, "
    "never wall time already spent.  `achieved_concurrency` is measured over a "
    "trailing window: how many LLM calls the system really runs at once, which "
    "is NOT max_in_flight (that is the DAG mutant dispatch cap).\n\n"
    "`get_last_adjustment_outcome()` shows what you did last time and where "
    "the estimate went afterwards.  If your last correction pushed the "
    "estimate the wrong way, reverse it rather than compounding it.  Every "
    "tool's output is written out below — you cannot call them, so work only "
    "from what is shown.  Your own calls are not in this telemetry.\n\n"
    "## Deciding\n"
    "0. Judge on LATENCY, not token size — tokens barely vary in this system\n"
    "1. Latency moved, tokens_out flat → `servers`\n"
    "2. Achieved concurrency far below max_in_flight and the pipeline is "
    "starved → `servers`, magnitude below 1\n"
    "3. Achieved concurrency climbing across the window → `servers`, "
    "magnitude slightly above 1\n"
    "4. tokens_out trending up across the last 5+ calls, or program length "
    "grown >20% vs baseline → `programs`\n"
    "5. Costs uniformly higher than the early calls implied, with no trend and "
    "no server story → `task`\n"
    "6. A single spike, or anything you cannot attribute → `noise`.  Prefer "
    "this: the measured terms are usually right and every lever adds variance.\n"
)


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


# Each diagnosis controls exactly one model lever.
CAUSE_LEVER = {
    "servers": "concurrency_mult",  # the only lever on the divisor
    "programs": "growth_rate_mult",  # remaining work grows faster than fitted
    "task": "golden_ratio",  # flat margin on remaining work
    "noise": None,  # no lever; the correct answer most times
}
# These levers affect all remaining work and require sustained evidence.
BLUNT_CAUSES = frozenset({"programs", "task"})


class _AdjustmentOutput(BaseModel):
    cause: str = Field(default="noise", description="servers | programs | task | noise")
    magnitude: float = Field(
        default=-1.0, description="Multiplier for the cause's lever (−1 = no change)"
    )
    sustained_over_calls: int = Field(
        default=0, description="Calls the deviation held for; <5 refuses programs/task"
    )
    reasoning: str = Field(default="")


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
        achieved_concurrency: float = 0.0,
        current_concurrency_mult: float = 1.0,
        elapsed_s: float = 0.0,
        attempts_done: int = 0,
        max_mutants: int = 0,
        predicted_duration_s: float = 0.0,
        trigger_reason: str = "",
        last_adjustment: dict[str, Any] | None = None,
        already_flagged: list[int] | None = None,
        miscoverage_rate: float | None = None,
        miscoverage_target: float = 0.10,
        width_scale: float = 1.0,
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
        self._conc = achieved_concurrency
        self._conc_mult = current_concurrency_mult
        self._elapsed = elapsed_s
        self._attempts = attempts_done
        self._max_mutants = max_mutants
        self._pred_duration = predicted_duration_s
        self._trigger = trigger_reason
        self._last_adj = last_adjustment
        self._flagged = set(already_flagged or ())
        self._miscov = miscoverage_rate
        self._miscov_target = miscoverage_target
        self._width_scale = width_scale
        self.adjustments: dict[str, Any] = {}

    def get_calibration(self) -> str:
        """How often the estimator has been surprising itself, against target.

        The agent is woken BY a miscoverage event, so this is the scoreboard
        for the loop it sits in — without it the agent is asked to judge one
        event with no idea whether such events are rare or constant.
        """
        if self._miscov is None:
            return "(no calibration history yet)"
        state = (
            "about right"
            if abs(self._miscov - self._miscov_target) < 0.05
            else "too often — the interval has been too narrow"
            if self._miscov > self._miscov_target
            else "rarely — the interval has been generous"
        )
        return (
            f"the estimate has landed outside its own previous interval "
            f"{self._miscov:.0%} of the time (target {self._miscov_target:.0%}): {state}. "
            f"The interval width is currently x{self._width_scale:.2f} of the model's own "
            f"estimate, adjusted automatically. Being woken is not itself proof that "
            f"something changed."
        )

    def get_recent_calls(self, n: int = 10) -> str:
        """Return recent calls and mark latency outliers and prior flags."""
        calls = self._calls[-n:]
        if not calls:
            return "(no LLM calls yet)"
        median_lat = _median([c.latency_ms for c in calls])
        median_tok = _median([c.tokens_in + c.tokens_out for c in calls])
        lines = [f"Last {len(calls)} LLM calls:"]
        for c in calls:
            if c.index in self._flagged:
                marker = " ← ALREADY FLAGGED (do not flag again)"
            elif median_lat > 0 and c.latency_ms > median_lat * 2.5:
                marker = f" ← SLOW ({c.latency_ms / median_lat:.1f}x median latency)"
            else:
                marker = ""
            lines.append(c.to_line() + marker)
        lines.append(
            f"Median latency: {median_lat:.0f}ms · median tokens: {median_tok:.0f}"
        )
        if self._flagged:
            lines.append(f"Already flagged this run: {sorted(self._flagged)}")
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
            f"growth×={self._growth:.2f} concurrency×={self._conc_mult:.2f} "
            f"program_len: {self._baseline_len}→{self._current_len}"
        )

    def get_progress(self) -> str:
        """Return run progress and the two measured terms of the duration
        estimate, so the agent can see WHERE the estimate comes from before
        deciding which lever (if any) to move."""
        rem = max(self._pred_duration - self._elapsed, 0.0)
        return (
            f"attempts={self._attempts}/{self._max_mutants} "
            f"elapsed={self._elapsed:.0f}s (measured) "
            f"remaining={rem:.0f}s (predicted) "
            f"achieved_concurrency={self._conc:.1f} calls in parallel "
            f"vs max_in_flight={self._max_flight} (DAG dispatch cap)"
        )

    def get_trigger(self) -> str:
        """Why you were woken up on this particular attempt."""
        return self._trigger or "(scheduled check, no specific anomaly)"

    def get_last_adjustment_outcome(self) -> str:
        """What your PREVIOUS decision was and what happened to the estimate
        since — the only way to tell whether your last correction helped."""
        a = self._last_adj
        if not a:
            return "(no previous adjustment this run)"
        promised = a["predicted_duration_s"] - a["elapsed_s"]
        spent = self._elapsed - a["elapsed_s"]
        attempts_since = self._attempts - a["attempt"]
        moved = [
            f"{k}={a[k]:.2f}" for k in ("golden", "growth", "concurrency") if a[k] > 0
        ]
        drift = ""
        if promised > 1 and self._pred_duration > 0:
            delta = (self._pred_duration - a["predicted_duration_s"]) / a[
                "predicted_duration_s"
            ]
            drift = (
                f"; since then the estimate moved {delta:+.0%} "
                f"({a['predicted_duration_s']:.0f}s -> {self._pred_duration:.0f}s)"
            )
        return (
            f"At attempt {a['attempt']} you set {', '.join(moved) or 'nothing'}"
            f" and flagged {a['outliers']} outlier(s); reason: {a['reasoning'] or '(none)'}. "
            f"{attempts_since} attempts and {spent:.0f}s have passed{drift}."
        )

    def adjust_model(
        self,
        golden_ratio: float = -1,
        growth_rate_mult: float = -1,
        concurrency_mult: float = -1,
    ) -> str:
        """Adjust cost model parameters. −1 means 'no change'."""
        changes = []
        if 0.5 <= golden_ratio <= 2.0:
            self.adjustments["golden_ratio"] = golden_ratio
            changes.append(f"golden={golden_ratio:.2f}")
        if 0.3 <= growth_rate_mult <= 3.0:
            self.adjustments["growth_rate_mult"] = growth_rate_mult
            changes.append(f"growth×={growth_rate_mult:.2f}")
        if 0.3 <= concurrency_mult <= 3.0:
            self.adjustments["concurrency_mult"] = concurrency_mult
            changes.append(f"concurrency×={concurrency_mult:.2f}")
        return (
            f"Adjusted: {', '.join(changes)}"
            if changes
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


TOOL_SCHEMA = """Available tools — call via tool name with arguments:

get_recent_calls(n=10) → table of last N LLM calls
get_program_diff()     → diff of current vs parent program
get_backpressure()     → pipeline in_flight/util %
get_model_params()     → current cold/golden/growth/concurrency multipliers
get_progress()         → attempts done, measured elapsed vs predicted remaining,
                         achieved LLM concurrency vs max_in_flight
get_trigger()          → why you were woken up on this attempt
get_last_adjustment_outcome() → your previous decision and what happened since

Action tools:
adjust_model(golden_ratio=1.15, growth_rate_mult=1.2, concurrency_mult=0.8)
flag_as_outlier(index=3)
skip_next_calibration()
"""


class CostMonitorState(TypedDict, total=False):
    messages: list[BaseMessage]
    llm_response: Any
    metadata: dict[str, Any]


class CostMonitorAgent(LangGraphAgent):
    """LLM agent that analyses cost telemetry and adjusts the cost model."""

    StateSchema = CostMonitorState

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

    async def arun(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the full build_prompt -> call_llm -> parse_response graph.

        CostMonitorHook drives the steps manually instead (it needs to set
        ``self.tools`` between calls), but LangGraphAgent.arun is abstract —
        without a concrete override the class can't be instantiated at all.
        """
        return await self.graph.ainvoke(state or {"messages": []})

    def build_prompt(self, state: dict[str, Any]) -> list[BaseMessage]:
        """Inline the non-callable tool outputs into the model prompt."""
        tools = self._tools

        context = (
            f"WHY YOU ARE AWAKE\n{tools.get_trigger()}\n\n"
            f"HOW OFTEN THIS HAPPENS\n{tools.get_calibration()}\n\n"
            f"RUN PROGRESS\n{tools.get_progress()}\n\n"
            f"YOUR PREVIOUS DECISION\n{tools.get_last_adjustment_outcome()}\n\n"
            f"RECENT LLM CALLS\n{tools.get_recent_calls(15)}\n\n"
            f"PIPELINE\n{tools.get_backpressure()}\n"
            f"CURRENT MODEL PARAMS\n{tools.get_model_params()}\n\n"
            f"PROGRAM DIFF\n{(tools.get_program_diff() or '(none)')[:2000]}"
        )

        user = HumanMessage(
            content=(
                f"{context}\n\n"
                "Name the CAUSE behind the trigger above. Respond with a JSON "
                "object:\n"
                '{"cause": "servers" | "programs" | "task" | "noise", '
                '"magnitude": float, "sustained_over_calls": int, '
                '"reasoning": "..."}\n\n'
                '"noise" with magnitude -1 changes nothing and is a valid, common '
                "and often correct answer — say why in `reasoning`."
            )
        )
        return [SystemMessage(content=SYSTEM_PROMPT), user]

    def parse_response(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply LLM decisions to the tool set."""
        response = state.get("llm_response")
        if response is None:
            return state

        content = getattr(response, "content", "")
        try:
            text = content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
        except (json.JSONDecodeError, Exception):
            return state

        # Map the diagnosis to its sole permitted lever.
        tools = self._tools
        cause = str(data.get("cause", "noise")).strip().lower()
        if cause not in CAUSE_LEVER:
            cause = "noise"
        try:
            magnitude = float(data.get("magnitude", -1))
        except (TypeError, ValueError):
            magnitude = -1.0
        reasoning = data.get("reasoning", "")

        levers = {
            "golden_ratio": -1.0,
            "growth_rate_mult": -1.0,
            "concurrency_mult": -1.0,
        }
        lever = CAUSE_LEVER[cause]
        if lever is not None and magnitude > 0:
            levers[lever] = magnitude

        tools.adjust_model(
            golden_ratio=levers["golden_ratio"],
            growth_rate_mult=levers["growth_rate_mult"],
            concurrency_mult=levers["concurrency_mult"],
        )

        state["cost_adjustments"] = {
            "cause": cause,
            "golden_ratio": tools.adjustments.get("golden_ratio", -1),
            "growth_rate_mult": tools.adjustments.get("growth_rate_mult", -1),
            "concurrency_mult": tools.adjustments.get("concurrency_mult", -1),
            "flag_outlier_indices": tools.adjustments.get("flag_outlier_indices", []),
            "sustained_over_calls": data.get("sustained_over_calls", 0),
            "skip_calibration": tools.adjustments.get("skip_calibration", False),
            "reasoning": reasoning,
        }
        return state


class _DummyResponse:
    """Minimal stand-in for an AIMessage — only `.content` is read downstream."""

    content: str = "{}"


class NoOpCostMonitorAgent(CostMonitorAgent):
    """Same build_prompt/parse_response path as CostMonitorAgent, but
    ``acall_llm`` never calls a real LLM — it returns an empty-JSON response.

    Used for the "without agent" ablation: CostMonitorHook still runs and
    still logs growth-law predictions ([CostMonitorHookJSON]), but no LLM
    call is made and no cost-model parameters are ever adjusted (parse_response
    reads all fields as -1/absent from "{}", which CostMonitorHook treats as
    "no change" — see the `if cold > 0` / `if golden > 0` / `if growth > 0`
    guards in CostMonitorHook.__call__).
    """

    def __init__(self):
        super().__init__(llm=None)

    async def acall_llm(self, state: dict) -> dict:
        state["llm_response"] = _DummyResponse()
        return state
