#!/usr/bin/env python3
"""Replay a historical run.py log's raw canonical events through the
CURRENT CostMonitorHook / growth_estimator code — to check whether a
prediction-math change would score better on already-collected data,
without running any evolution (no LLM budget, no wall-clock time).

Usage:
    python3 tools/cost_ablation/replay_from_log.py path/to/one.log [more.log ...]
    python3 tools/cost_ablation/replay_from_log.py --out-dir experiments/replay_check *.log

Scope: this replays the DETERMINISTIC part of the prediction pipeline —
static baseline + growth-law fit (LLM latency) + non-LLM stage duration
(CallProgramFunction/CallValidatorFunction/IntraMemoryStage) — by feeding
the log's own [LLM_CALL], [STAGE_EXEC] and [MUTATION_ATTEMPTED] lines
through a fresh CostMonitorHook wired to the real emit/subscribe bus, using
a NoOpCostMonitorAgent. It does NOT replay CostMonitorAgent's LLM-driven
calibration cycle (golden/growth overrides) — that needs a real LLM call
each time and can't be meaningfully reproduced offline; use
run_ablation.py for an apples-to-apples with-agent/without-agent test.

Each replayed log produces the same summary shape as build_report.py's
summarize() (pred/actual tokens & duration, error %, per-attempt series),
so this reuses build_report.py's LaTeX table + error-decay plot builders
directly — pass --out-dir to get comparison_table.tex/.png and
error_decay_plots.png across all replayed logs, exactly like a real
ablation run's report.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import build_error_decay_png, build_table_png, build_table_tex  # noqa: E402

from loguru import logger

from gigaevo.llm.agents.cost_monitor import NoOpCostMonitorAgent
from gigaevo.monitoring.cost_monitor_hook import CostMonitorHook, _clamp_step
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.monitoring.emit import emit, reset_subscribers
from gigaevo.monitoring.events import BackpressureSample, LLMCall, MutationAttempted, StageExec

TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
MAX_MUTANTS_RE = re.compile(r"Evolution running \(max_mutants=(\d+)\)")
MAX_IN_FLIGHT_RE = re.compile(r"max_in_flight=(\d+)")
DUR_RE = re.compile(r"Duration: ([\d.]+)s")
TOK_RE = re.compile(r"\[TokenTracker:(\w+)\] ([\w.\-]+): \d+ ctx \+ \d+ gen \(\d+ reasoning\) = \d+ \(cumulative: (\d+)\)")
ADJ_RE = re.compile(
    r"\[CostMonitorHook\] adjustments: cold=(\S+) golden=(\S+) growth=(\S+)(?: conc=(\S+))?")

RAW_EVENT_PATTERNS = {
    "LLM_CALL": re.compile(r"\[LLM_CALL\] (\{.*\})"),
    "STAGE_EXEC": re.compile(r"\[STAGE_EXEC\] (\{.*\})"),
    "MUTATION_ATTEMPTED": re.compile(r"\[MUTATION_ATTEMPTED\] (\{.*\})"),
    "BACKPRESSURE_SAMPLE": re.compile(r"\[BACKPRESSURE_SAMPLE\] (\{.*\})"),
}
EVENT_CLASSES = {
    "LLM_CALL": LLMCall,
    "STAGE_EXEC": StageExec,
    "MUTATION_ATTEMPTED": MutationAttempted,
    "BACKPRESSURE_SAMPLE": BackpressureSample,
}


def replay_log(path: Path, *, apply_agent: bool = False,
               outlier_policy: bool = False) -> dict | None:
    """Parse ``path``'s raw events + ground truth, replay the events through
    a fresh CostMonitorHook using the currently-installed code, and return a
    summary dict shaped like build_report.summarize()'s output — or None if
    the log doesn't have enough data to replay (no events / no ground truth)."""
    max_mutants = max_in_flight = actual_duration = None
    tok_cum: dict[str, int] = {}
    events_in_order: list[tuple[str, dict]] = []
    # CostMonitorAgent's own historical decisions, replayed in place so the
    # deterministic estimator can be A/B'd against estimator+agent on the very
    # same event stream. The agent's LLM call itself is not re-run — these are
    # the multipliers it actually chose during the original run.
    adjustments: list[tuple[float, float, float, float]] = []

    t0 = None
    line_ts = 0.0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            mts = TS_RE.match(line)
            if mts:
                t = datetime.strptime(mts.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
                if t0 is None:
                    t0 = t
                line_ts = t - t0
            if max_mutants is None:
                m = MAX_MUTANTS_RE.search(line)
                if m:
                    max_mutants = int(m.group(1))
            if max_in_flight is None:
                m = MAX_IN_FLIGHT_RE.search(line)
                if m:
                    max_in_flight = int(m.group(1))
            m = DUR_RE.search(line)
            if m:
                actual_duration = float(m.group(1))
            m = TOK_RE.search(line)
            if m:
                tok_cum[m.group(2)] = int(m.group(3))
            m = ADJ_RE.search(line)
            if m:
                def _f(v):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return -1.0
                adjustments.append((line_ts, _f(m.group(1)), _f(m.group(2)),
                                    _f(m.group(3)), _f(m.group(4))))
                continue
            for name, pattern in RAW_EVENT_PATTERNS.items():
                mm = pattern.search(line)
                if mm:
                    try:
                        payload = json.loads(mm.group(1))
                    except json.JSONDecodeError:
                        break
                    payload.pop("event", None)
                    events_in_order.append((name, payload, line_ts))
                    break

    if max_mutants is None or max_in_flight is None or not events_in_order:
        return None
    actual_tokens = sum(tok_cum.values())
    if not actual_duration or not actual_tokens:
        return None

    reset_subscribers()
    try:
        pred = CostPrediction(max_mutants=max_mutants, max_in_flight=max_in_flight)
        # The anchored duration model reads wall time; drive it from the log's
        # own timestamps so a replay reproduces what the live run would have
        # predicted, not how fast the replay itself runs.
        clock_now = [0.0]
        hook = CostMonitorHook(  # noqa: F841
            agent=NoOpCostMonitorAgent(), prediction=pred, interval=5,
            clock=lambda: clock_now[0],
        )

        pending_adj = list(adjustments) if apply_agent else []
        series = []
        for name, payload, ts in events_in_order:
            clock_now[0] = ts
            while pending_adj and pending_adj[0][0] <= ts:
                _, cold, golden, growth, conc = pending_adj.pop(0)
                if cold > 0:
                    pred.llm_cold_override = cold
                if golden > 0:
                    pred.llm_golden_override = _clamp_step(
                        pred.llm_golden_override if pred.llm_golden_override > 0 else 1.0, golden)
                if growth > 0:
                    pred.llm_growth_override = _clamp_step(
                        pred.llm_growth_override if pred.llm_growth_override > 0 else 1.0, growth)
                if conc > 0:
                    pred.llm_concurrency_override = _clamp_step(
                        pred.llm_concurrency_override if pred.llm_concurrency_override > 0 else 1.0,
                        conc)
            try:
                event = EVENT_CLASSES[name](**payload)
            except Exception:
                continue
            emit(event)
            if name == "MUTATION_ATTEMPTED":
                # Offline there is no async loop, so the hook never dispatches
                # the agent and ``_agent_due`` would latch True forever.
                # Consume the trigger exactly as _run_agent() would, so the
                # replay reproduces the real wake-up CADENCE (the LLM call
                # itself still isn't reproduced — see the module docstring).
                fired = hook._agent_due
                if fired:
                    hook._agent_due = False
                    hook._agent_calls += 1
                    hook._last_agent_attempt = hook._attempts
                    if outlier_policy:
                        # Deterministic stand-in for the LLM's judgement, so the
                        # MECHANISM can be measured separately from the model's
                        # answers: on every wake-up, flag the single biggest
                        # call in the recent window if it dwarfs the median.
                        recent = hook._call_history[-20:]
                        if len(recent) >= 5:
                            lats = sorted(r.latency_ms for r in recent)
                            med = lats[len(lats) // 2]
                            worst = max(recent, key=lambda r: r.latency_ms)
                            if med > 0 and worst.latency_ms > 2.5 * med:
                                hook._flag_outliers([worst.index])
                series.append({
                    "agent_due": fired,
                    "predicted_tokens": pred.predicted_total_tokens,
                    "predicted_duration_s": pred.predicted_duration_s,
                    "ci_low_s": pred.ci_low_s,
                    "ci_high_s": pred.ci_high_s,
                    "elapsed_s": ts,
                })
    finally:
        reset_subscribers()

    if not series:
        return None
    last = series[-1]
    return dict(
        path=str(path), n_points=len(series), series=series,
        # only decisions that actually moved a multiplier; the agent's
        # "everything normal" answer logs all -1 and changes nothing
        n_trigger_fires=sum(1 for x in series if x["agent_due"]),
        n_agent_adjustments=sum(1 for a in adjustments if any(v > 0 for v in a[1:])),
        pred_tokens=last["predicted_tokens"], pred_duration=last["predicted_duration_s"],
        actual_tokens=actual_tokens, actual_duration=actual_duration,
        tok_err_pct=(last["predicted_tokens"] - actual_tokens) / actual_tokens * 100,
        dur_err_pct=(last["predicted_duration_s"] - actual_duration) / actual_duration * 100,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path, help="run.py log file(s) to replay")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="if set, write comparison_table.tex/.png + error_decay_plots.png here")
    args = ap.parse_args()

    # emit() re-logs every replayed event through loguru at INFO — noisy and
    # pointless for an offline batch tool, silence it.
    logger.remove()

    rows_by_task: dict[str, dict[str, dict]] = {}
    print(f"{'log':45s} {'n':>5s} {'pred_tok':>10s} {'act_tok':>10s} {'tok_err%':>9s} "
          f"{'pred_dur':>9s} {'act_dur':>9s} {'dur_err%':>9s}")
    for log_path in args.logs:
        result = replay_log(log_path)
        if not result:
            print(f"{log_path.name:45s} -- skipped (no events / no ground truth)")
            continue
        print(f"{log_path.name:45s} {result['n_points']:5d} {result['pred_tokens']:10,d} "
              f"{result['actual_tokens']:10,d} {result['tok_err_pct']:+9.1f} "
              f"{result['pred_duration']:9.0f} {result['actual_duration']:9.0f} "
              f"{result['dur_err_pct']:+9.1f}")
        rows_by_task[log_path.stem] = {"withagent": result}  # reuse build_report's row shape

    if args.out_dir and rows_by_task:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "comparison_table.tex").write_text(build_table_tex(rows_by_task), encoding="utf-8")
        build_table_png(rows_by_task, args.out_dir / "comparison_table.png")
        build_error_decay_png(rows_by_task, args.out_dir / "error_decay_plots.png")
        print(f"\nreport written to {args.out_dir}")


if __name__ == "__main__":
    main()
