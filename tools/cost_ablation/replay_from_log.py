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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import build_error_decay_png, build_table_png, build_table_tex  # noqa: E402

from loguru import logger

from gigaevo.llm.agents.cost_monitor import NoOpCostMonitorAgent
from gigaevo.monitoring.cost_monitor_hook import CostMonitorHook
from gigaevo.monitoring.cost_predictor import CostPrediction
from gigaevo.monitoring.emit import emit, reset_subscribers
from gigaevo.monitoring.events import BackpressureSample, LLMCall, MutationAttempted, StageExec

MAX_MUTANTS_RE = re.compile(r"Evolution running \(max_mutants=(\d+)\)")
MAX_IN_FLIGHT_RE = re.compile(r"max_in_flight=(\d+)")
DUR_RE = re.compile(r"Duration: ([\d.]+)s")
TOK_RE = re.compile(r"\[TokenTracker:(\w+)\] ([\w.\-]+): \d+ ctx \+ \d+ gen \(\d+ reasoning\) = \d+ \(cumulative: (\d+)\)")

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


def replay_log(path: Path) -> dict | None:
    """Parse ``path``'s raw events + ground truth, replay the events through
    a fresh CostMonitorHook using the currently-installed code, and return a
    summary dict shaped like build_report.summarize()'s output — or None if
    the log doesn't have enough data to replay (no events / no ground truth)."""
    max_mutants = max_in_flight = actual_duration = None
    tok_cum: dict[str, int] = {}
    events_in_order: list[tuple[str, dict]] = []

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
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
            for name, pattern in RAW_EVENT_PATTERNS.items():
                mm = pattern.search(line)
                if mm:
                    try:
                        payload = json.loads(mm.group(1))
                    except json.JSONDecodeError:
                        break
                    payload.pop("event", None)
                    events_in_order.append((name, payload))
                    break

    if max_mutants is None or max_in_flight is None or not events_in_order:
        return None
    actual_tokens = sum(tok_cum.values())
    if not actual_duration or not actual_tokens:
        return None

    reset_subscribers()
    try:
        pred = CostPrediction(max_mutants=max_mutants, max_in_flight=max_in_flight)
        hook = CostMonitorHook(agent=NoOpCostMonitorAgent(), prediction=pred, interval=5)  # noqa: F841

        series = []
        for name, payload in events_in_order:
            try:
                event = EVENT_CLASSES[name](**payload)
            except Exception:
                continue
            emit(event)
            if name == "MUTATION_ATTEMPTED":
                series.append({
                    "predicted_tokens": pred.predicted_total_tokens,
                    "predicted_duration_s": pred.predicted_duration_s,
                })
    finally:
        reset_subscribers()

    if not series:
        return None
    last = series[-1]
    return dict(
        path=str(path), n_points=len(series), series=series,
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
