#!/usr/bin/env python3
"""Is the prediction interval honest, and does online calibration fix it?

Two numbers per run, both read off the replayed prediction series:

  miscoverage — how often a new estimate lands outside the interval the
    PREVIOUS estimate published. This is exactly CostMonitorAgent's wake-up
    condition, so it is also the wake-up rate the harness will see.

  coverage — how often the published interval actually contained the run's
    eventual measured duration. An interval that is never breached but never
    contains the truth is just a wide interval; both numbers are needed.

Measured on the alphaevolve ablation before online calibration: coverage
64-68% against a 90% target — the interval claimed precision it did not
have. Sweeping ``--gammas`` replays the same logs at several calibration
step sizes; gamma=0 is the model-only width, i.e. the old behaviour.

Costs nothing: no LLM calls, no evolution, just the collected logs.

Usage:
    python3 tools/cost_ablation/check_calibration.py experiments/<dir>/*.log
    python3 tools/cost_ablation/check_calibration.py --gammas 0 0.05 0.1 <logs>
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

# Running a script puts its own directory on sys.path, not the cwd — so the
# repo root has to be added explicitly or `gigaevo` is invisible.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_from_log import replay_log  # noqa: E402

from loguru import logger


def score(series: list[dict], actual_duration: float) -> tuple[float, float] | None:
    """(miscoverage %, coverage %) for one replayed run."""
    if len(series) < 5 or not actual_duration:
        return None
    breaches = covered = n_cov = 0
    prev: tuple[float, float] | None = None
    for p in series:
        lo, hi = p["ci_low_s"], p["ci_high_s"]
        if prev and prev[1] > prev[0] > 0 and not (prev[0] <= p["predicted_duration_s"] <= prev[1]):
            breaches += 1
        if hi > lo:
            n_cov += 1
            covered += int(lo <= actual_duration <= hi)
        prev = (lo, hi)
    return (100 * breaches / (len(series) - 1), 100 * covered / max(n_cov, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.05, 0.1],
                    help="calibration step sizes to compare; 0 = model-only width")
    ap.add_argument("--alpha", type=float, default=0.10, help="target miscoverage")
    ap.add_argument("--ci-method", default="montecarlo")
    args = ap.parse_args()
    logger.remove()

    print(f"target: miscoverage {args.alpha:.0%}, coverage {1 - args.alpha:.0%}\n")
    print(f"{'gamma':>6s} {'arm':>10s} {'runs':>5s} {'miscoverage':>12s} {'coverage':>10s}")
    for gamma in args.gammas:
        arms: dict[str, list[tuple[float, float]]] = {"noagent": [], "withagent": []}
        for path in args.logs:
            r = replay_log(path, ci_method=args.ci_method, aci_gamma=gamma, aci_alpha=args.alpha)
            if not r:
                continue
            s = score(r["series"], r["actual_duration"])
            if s:
                arms["withagent" if "withagent" in path.name else "noagent"].append(s)
        for arm, rows in arms.items():
            if not rows:
                continue
            print(f"{gamma:6.2f} {arm:>10s} {len(rows):5d} "
                  f"{st.median(r[0] for r in rows):11.1f}% {st.median(r[1] for r in rows):9.1f}%")
        print()


if __name__ == "__main__":
    main()
