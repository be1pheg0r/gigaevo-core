"""What the CostMonitorAgent is actually shown, and what it can actually move.

Both were broken in ways reading the system prompt could not reveal: three
observability tools were named in the prompt but never inlined into it, and
`concurrency_mult` was advertised as the only lever on the divisor while
`parse_response` silently dropped it.
"""
from __future__ import annotations

from gigaevo.llm.agents.cost_monitor import (
    SYSTEM_PROMPT,
    CostMonitorAgent,
    LlmCallRecord,
    _ToolSet,
)


class _Resp:
    def __init__(self, content: str):
        self.content = content


def _agent(tools: _ToolSet) -> CostMonitorAgent:
    a = CostMonitorAgent(llm=None)
    a.tools = tools
    return a


def _tools(**kw) -> _ToolSet:
    base = dict(
        recent_calls=[
            LlmCallRecord(index=0, tokens_in=3800, tokens_out=900, latency_ms=27_000, stage="MutationAgent"),
            LlmCallRecord(index=1, tokens_in=3900, tokens_out=950, latency_ms=28_000, stage="MutationAgent"),
            LlmCallRecord(index=2, tokens_in=3850, tokens_out=2842, latency_ms=350_000, stage="MutationAgent"),
        ],
        trigger_reason="estimate jumped above its own interval at attempt 20",
        elapsed_s=715.0, attempts_done=20, max_mutants=100,
        predicted_duration_s=13_109.0, achieved_concurrency=7.2, max_in_flight=8,
    )
    base.update(kw)
    return _ToolSet(**base)


class TestEveryToolReachesTheModel:
    def test_prompt_carries_trigger_progress_and_last_outcome(self) -> None:
        t = _tools(last_adjustment={
            "attempt": 8, "golden": 1.2, "growth": -1.0, "concurrency": -1.0,
            "outliers": 1, "reasoning": "one slow call",
            "predicted_duration_s": 3251.0, "elapsed_s": 321.0,
        })
        text = _agent(t).build_prompt({"messages": []})[1].content

        # the three that used to be missing
        assert "jumped above its own interval at attempt 20" in text
        assert "attempts=20/100" in text
        assert "achieved_concurrency" in text
        assert "At attempt 8 you set golden=1.20" in text
        assert "one slow call" in text
        # and the ones that were already there
        assert "MutationAgent" in text
        assert "in_flight=" in text

    def test_prompt_never_promises_a_tool_it_does_not_inline(self) -> None:
        """The system prompt tells the agent to consult these by name; the
        tools are not callable, so anything named must be in the user turn."""
        text = _agent(_tools()).build_prompt({"messages": []})[1].content
        for tool in ("get_trigger", "get_progress", "get_last_adjustment_outcome"):
            assert tool in SYSTEM_PROMPT, f"{tool} no longer referenced — update this test"
        assert "WHY YOU ARE AWAKE" in text
        assert "YOUR PREVIOUS DECISION" in text


class TestConcurrencyLeverIsWired:
    def test_concurrency_mult_survives_parse_response(self) -> None:
        a = _agent(_tools())
        out = a.parse_response({"llm_response": _Resp(
            '{"cold_start_factor": -1, "golden_ratio": -1, "growth_rate_mult": -1, '
            '"concurrency_mult": 0.8, "flag_outlier_indices": [], '
            '"skip_calibration": false, "reasoning": "server contention"}')})
        adj = out["cost_adjustments"]
        assert adj["concurrency_mult"] == 0.8
        assert adj["golden_ratio"] == -1, "must not silently reach for the wrong lever"

    def test_the_json_shape_offered_to_the_model_lists_it(self) -> None:
        text = _agent(_tools()).build_prompt({"messages": []})[1].content
        assert '"concurrency_mult"' in text

    def test_out_of_range_values_are_refused(self) -> None:
        a = _agent(_tools())
        out = a.parse_response({"llm_response": _Resp('{"concurrency_mult": 42}')})
        assert out["cost_adjustments"]["concurrency_mult"] == -1


class TestOutlierMarkerTracksLatency:
    def test_a_latency_spike_is_marked_even_when_tokens_are_flat(self) -> None:
        """Tokens have no outliers in this system (max/median 1.6-2.5x), so a
        token-based marker never fired; latency max/median is 11-27x."""
        table = _tools().get_recent_calls(10)
        spike = [ln for ln in table.splitlines() if ln.startswith("#2 ")][0]
        assert "SLOW" in spike
        assert all("SLOW" not in ln for ln in table.splitlines() if ln.startswith(("#0 ", "#1 ")))

    def test_already_flagged_calls_say_so(self) -> None:
        table = _tools(already_flagged=[2]).get_recent_calls(10)
        assert "ALREADY FLAGGED" in table
        assert "Already flagged this run: [2]" in table
        # and it must not also be offered as a fresh spike
        assert "SLOW" not in table


class TestAgentIsShownItsScoreboard:
    """The agent is woken BY a miscoverage event; without knowing how often
    that happens it cannot tell a real regime change from routine noise."""

    def test_a_high_rate_is_reported_as_the_interval_being_too_narrow(self) -> None:
        t = _tools(miscoverage_rate=0.35, miscoverage_target=0.10, width_scale=1.4)
        text = _agent(t).build_prompt({"messages": []})[1].content
        assert "35% of the time" in text
        assert "too narrow" in text
        assert "x1.40" in text

    def test_a_rate_on_target_says_so(self) -> None:
        assert "about right" in _tools(miscoverage_rate=0.11).get_calibration()

    def test_a_low_rate_reads_as_a_generous_interval(self) -> None:
        assert "generous" in _tools(miscoverage_rate=0.01).get_calibration()

    def test_no_history_does_not_invent_one(self) -> None:
        assert "no calibration history" in _tools().get_calibration()
