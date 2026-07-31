"""CostMonitorAgent — LLM-powered cost model adjuster.

Woken by CostMonitorHook when the live estimate breaches the confidence
interval its own previous estimate published (not on a fixed tick), and
decides whether that surprise is a transient outlier or a regime change.
Returns structured adjustments consumed by ``CostMonitorHook``.

Tool set
--------
get_recent_calls(n)          — last N LLM_CALL records (tokens, latency, stage)
get_program_diff(program_id) — diff of the mutated program vs its parent
get_backpressure()           — current pipeline backpressure snapshot
get_model_params()           — current cost model state (cold_factor, golden, …)
get_progress()               — attempts, measured elapsed vs predicted remaining
get_trigger()                — why this call was woken up
get_last_adjustment_outcome()— previous decision and what the estimate did since
adjust_model(params)         — apply corrections to the live cost model
flag_as_outlier(idx)         — winsorise that call's bucket out of the growth fit

"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent


# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a **Cost Monitor Agent** for an evolutionary code optimisation "
    "pipeline.  You are NOT on a timer: you are woken up when the estimator "
    "surprises itself — its new estimate landed outside the confidence "
    "interval its own previous estimate published.  `get_trigger()` tells you "
    "what happened.  Your job is to explain that specific event: is it a "
    "**transient outlier** (one slow or huge call — flag it, leave the model "
    "alone) or a **regime change** (programs grew, model switched, servers "
    "slowed — correct the model)?\n\n"
    "Default to 'transient'.  Flagging an outlier is cheap and surgical: the "
    "flagged call's bucket is winsorised out of the growth-law fit, so a spike "
    "stops being extrapolated over the whole remaining run, and nothing else "
    "changes.  Moving a multiplier is blunt — it applies to everything still "
    "ahead — so only do it when the deviation has PERSISTED across several "
    "calls, and say so in `reasoning`.\n\n"
    "`get_last_adjustment_outcome()` shows what you did last time and where "
    "the estimate went afterwards.  If your last correction pushed the "
    "estimate the wrong way, undo it rather than compounding it.\n\n"
    "`get_progress()` gives the two measured terms of the estimate and the "
    "measured concurrency.  Every tool's output is already written out for "
    "you below — you cannot call them, so work only from what is shown.\n\n"
    "Calls marked ALREADY FLAGGED have been winsorised out of the fit "
    "already.  Do not flag them again: it changes nothing and spends a "
    "wakeup.  Your own calls do not appear in this telemetry at all.\n\n"
    "## How the estimate you are correcting is built\n"
    "`predicted_duration = elapsed_so_far + remaining_service_time / "
    "achieved_concurrency`.  `elapsed_so_far` is measured, not guessed — your "
    "levers only ever scale the REMAINING work or the concurrency divisor, "
    "never wall time already spent.  `achieved_concurrency` is measured over "
    "a trailing window: it is how many LLM calls the system really runs at "
    "once, which is NOT the same as max_in_flight (that is the DAG mutant "
    "dispatch cap).\n\n"
    "## Available actions (via JSON output)\n"
    "- `golden_ratio`: float 0.8–2.0 — safety margin on remaining work\n"
    "- `growth_rate_mult`: float 0.5–3.0 — remaining work will grow/shrink "
    "faster than the fitted trend (compounds with golden_ratio)\n"
    "- `concurrency_mult`: float 0.3–3.0 — the measured concurrency is about "
    "to change (>1 = throughput recovering, <1 = servers slowing down). Use "
    "this instead of golden_ratio when the cause is the SERVERS, not the "
    "programs — it is the only lever that touches the divisor\n"
    "- `flag_outlier_indices`: list[int] — mark these call indices as outliers\n"
    "- `sustained_over_calls`: int — REQUIRED whenever you set golden_ratio or "
    "growth_rate_mult: over how many recent calls the deviation actually held. "
    "Below 5 the change is refused, because a blunt lever must not answer a "
    "single spike\n"
    "- `skip_calibration`: bool — skip the next scheduled calibration\n"
    "- `reasoning`: string ≤ 80 chars — why you made these decisions\n\n"
    "## Decision rules\n"
    "0. A call marked SLOW is 2.5×+ the median LATENCY of the window; tokens "
    "in this system barely vary, so judge spikes on latency, not size\n"
    "1. Single-call spike (one call 3× median, rest normal) → flag outlier\n"
    "2. Sustained token growth (last 5 calls trending up) → raise growth_rate_mult\n"
    "3. Latency rising while tokens_out flat → server contention, lower "
    "concurrency_mult (do NOT raise golden_ratio: the work did not grow)\n"
    "4. Achieved concurrency far below max_in_flight AND the pipeline is "
    "starved (in_flight low, few accepts) → concurrency will stay low, "
    "lower concurrency_mult\n"
    "5. Achieved concurrency climbing across the window → run is still ramping "
    "up, raise concurrency_mult slightly\n"
    "6. Program length grew > 20 % vs baseline → raise growth_rate_mult by 0.2\n"
    "7. Everything normal → no changes (all defaults). Prefer no change: the "
    "measured terms are usually right, and every lever you move adds variance.\n"
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
    golden_ratio: float = Field(default=-1.0, description="Override golden_ratio (−1 = no change, 0.8–2.0)")
    growth_rate_mult: float = Field(default=-1.0, description="Multiply growth_rate γ (−1 = no change, 0.5–3.0)")
    concurrency_mult: float = Field(default=-1.0, description="Multiply MEASURED achieved LLM concurrency (−1 = no change, 0.3–3.0)")
    flag_outlier_indices: list[int] = Field(default_factory=list, description="Indices of calls to ignore")
    sustained_over_calls: int = Field(default=0, description="Calls the deviation held for; <5 refuses golden/growth")
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
        achieved_concurrency: float = 0.0,
        current_concurrency_mult: float = 1.0,
        elapsed_s: float = 0.0,
        attempts_done: int = 0,
        max_mutants: int = 0,
        predicted_duration_s: float = 0.0,
        trigger_reason: str = "",
        last_adjustment: dict[str, Any] | None = None,
        already_flagged: list[int] | None = None,
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
        # Accumulated adjustments
        self.adjustments: dict[str, Any] = {}

    # ── Observability tools ────────────────────────────────────────────────

    def get_recent_calls(self, n: int = 10) -> str:
        """Return the last N LLM calls as a compact table.

        Outliers are marked on LATENCY, not tokens. Measured across the
        alphaevolve runs, token max/median per call is 1.6-2.5x — under the
        2.5x threshold the marker used, so it essentially never fired and the
        model was left recomputing spikes by hand off the ``lat=`` column.
        Latency max/median is 11-27x, which is where the spikes actually are.
        Already-flagged calls are labelled so a wakeup is not spent
        rediscovering a spike that has already been winsorised.
        """
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
        lines.append(f"Median latency: {median_lat:.0f}ms · median tokens: {median_tok:.0f}")
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
        moved = [f"{k}={a[k]:.2f}" for k in ("golden", "growth", "concurrency") if a[k] > 0]
        drift = ""
        if promised > 1 and self._pred_duration > 0:
            delta = (self._pred_duration - a["predicted_duration_s"]) / a["predicted_duration_s"]
            drift = (f"; since then the estimate moved {delta:+.0%} "
                     f"({a['predicted_duration_s']:.0f}s -> {self._pred_duration:.0f}s)")
        return (
            f"At attempt {a['attempt']} you set {', '.join(moved) or 'nothing'}"
            f" and flagged {a['outliers']} outlier(s); reason: {a['reasoning'] or '(none)'}. "
            f"{attempts_since} attempts and {spent:.0f}s have passed{drift}."
        )

    # ── Action tools — accumulate adjustments ──────────────────────────────

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


# ── Agent ───────────────────────────────────────────────────────────────────

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
        """Inline every tool's output into the prompt.

        The tools are not callable by the model — whatever is not written here
        the agent never sees. ``get_trigger``, ``get_progress`` and
        ``get_last_adjustment_outcome`` used to be omitted while the system
        prompt told the agent to consult them, so it decided without knowing
        why it was woken, how far along the run was, or whether its previous
        correction had helped. Ordered from the question ("why am I awake")
        through the evidence to the levers.
        """
        tools = self._tools

        context = (
            f"WHY YOU ARE AWAKE\n{tools.get_trigger()}\n\n"
            f"RUN PROGRESS\n{tools.get_progress()}\n\n"
            f"YOUR PREVIOUS DECISION\n{tools.get_last_adjustment_outcome()}\n\n"
            f"RECENT LLM CALLS\n{tools.get_recent_calls(15)}\n\n"
            f"PIPELINE\n{tools.get_backpressure()}\n"
            f"CURRENT MODEL PARAMS\n{tools.get_model_params()}\n\n"
            f"PROGRAM DIFF\n{(tools.get_program_diff() or '(none)')[:2000]}"
        )

        user = HumanMessage(content=(
            f"{context}\n\n"
            "Analyse the telemetry above and answer the trigger specifically. "
            "Respond with a JSON object:\n"
            '{"golden_ratio": float, "growth_rate_mult": float, '
            '"concurrency_mult": float, "flag_outlier_indices": [int], '
            '"sustained_over_calls": int, '
            '"skip_calibration": bool, "reasoning": "..."}\n\n'
            "Use -1 for any parameter you do NOT want to change. Changing "
            "nothing (every value -1, no indices) is a valid and often correct "
            "answer — say why in `reasoning`."
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
        golden = data.get("golden_ratio", -1)
        growth = data.get("growth_rate_mult", -1)
        # concurrency_mult used to be dropped here — not passed to adjust_model
        # and not copied into cost_adjustments — while the system prompt calls
        # it "the only lever that touches the divisor" and three of the seven
        # decision rules require it. The agent diagnosed server contention
        # correctly and then had to reach for golden_ratio, which the prompt
        # explicitly forbids in that case.
        conc = data.get("concurrency_mult", -1)
        outliers = data.get("flag_outlier_indices", [])
        skip = data.get("skip_calibration", False)
        reasoning = data.get("reasoning", "")

        tools.adjust_model(golden_ratio=golden, growth_rate_mult=growth,
                           concurrency_mult=conc)

        for idx in outliers:
            tools.flag_as_outlier(idx)

        if skip:
            tools.skip_next_calibration()

        state["cost_adjustments"] = {
            "golden_ratio": tools.adjustments.get("golden_ratio", -1),
            "growth_rate_mult": tools.adjustments.get("growth_rate_mult", -1),
            "concurrency_mult": tools.adjustments.get("concurrency_mult", -1),
            "flag_outlier_indices": tools.adjustments.get("flag_outlier_indices", []),
            "sustained_over_calls": data.get("sustained_over_calls", 0),
            "skip_calibration": tools.adjustments.get("skip_calibration", False),
            "reasoning": reasoning,
        }
        return state

# ── No-op agent for ablation studies ────────────────────────────────────────

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
