#!/usr/bin/env python3
"""Launch with-agent / without-agent cost-prediction ablation runs for a set
of gigaevo problems, wait for completion, then build the comparison report.

Usage (from repo root, e.g. on the summer-school server):

    python3 tools/cost_ablation/run_ablation.py \\
        --tasks algotune/algotune_lqr algotune/algotune_markowitz \\
                adversarial/code/pop_a toy_kadane \\
        --max-mutants 100 --llm summer_school_servers \\
        --out-dir experiments/cost_ablation_$(date +%Y%m%d_%H%M%S)

Each task is launched twice: once with ``+cost_monitor=enabled`` (the real
LLM-driven CostMonitorAgent) and once with ``+cost_monitor=agentless`` (the
NoOpCostMonitorAgent stub — growth-law prediction/logging still runs, no LLM
call is made) — see config/cost_monitor/{enabled,agentless}.yaml. This gives
a same-task, same-conditions "with agent" vs "without agent" comparison.

Redis db numbers are auto-picked from the free range 0-127 (this server has
128 logical dbs; db >= 128 crashes hydra's redis_storage instantiation) and
verified empty before use, to avoid the "dirty db makes the run finish
suspiciously fast without doing real work" trap.

The script then polls each log for a ``Duration:`` line (written by
run.py's own final log line) until every run finishes or --timeout is hit,
and finally invokes build_report.py to produce the comparison artifacts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import redis

REPO_ROOT = Path(__file__).resolve().parents[2]
REDIS_DB_RANGE = range(0, 128)
CONDITIONS = [("noagent", "agentless"), ("withagent", "enabled")]


def find_free_dbs(n: int, host: str = "localhost", port: int = 6379) -> list[int]:
    """Scan redis db indices 0-127 and return the first n that are empty."""
    free = []
    for db in REDIS_DB_RANGE:
        r = redis.Redis(host=host, port=port, db=db)
        if not r.keys("*"):
            free.append(db)
        if len(free) >= n:
            break
    if len(free) < n:
        raise RuntimeError(f"Only found {len(free)} free redis dbs (0-127), need {n}")
    return free


def launch(task: str, db: int, max_mutants: int, llm: str, cost_monitor: str, log_path: Path) -> subprocess.Popen:
    cmd = [
        "python3", "run.py",
        f"problem.name={task}",
        f"max_mutants={max_mutants}",
        f"llm={llm}",
        f"redis.db={db}",
        "redis.resume=true",
        f"+cost_monitor={cost_monitor}",
    ]
    log_f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )


def is_finished(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "Duration:" in text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True,
                     help="problem.name values, e.g. algotune/algotune_lqr toy_kadane")
    ap.add_argument("--max-mutants", type=int, default=100)
    ap.add_argument("--llm", default="summer_school_servers")
    ap.add_argument("--out-dir", required=True,
                     help="relative to repo root, e.g. experiments/cost_ablation_20260801_120000")
    ap.add_argument("--poll-interval", type=int, default=60, help="seconds between health checks")
    ap.add_argument("--timeout", type=int, default=3600 * 3, help="max seconds to wait for all runs")
    ap.add_argument("--skip-report", action="store_true", help="only launch + wait, skip building the report")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dbs = find_free_dbs(len(args.tasks) * len(CONDITIONS))

    manifest = []
    procs = []
    i = 0
    for task in args.tasks:
        key = task.replace("/", "_")
        for cond_name, cost_monitor in CONDITIONS:
            db = dbs[i]
            i += 1
            log_path = out_dir / f"{key}_{cond_name}.log"
            proc = launch(task, db, args.max_mutants, args.llm, cost_monitor, log_path)
            manifest.append({
                "task": task, "key": key, "condition": cond_name, "db": db,
                "log": str(log_path.relative_to(REPO_ROOT)), "pid": proc.pid,
            })
            procs.append((proc, log_path))
            print(f"launched {task} [{cond_name}] db={db} pid={proc.pid} -> {log_path}")
            time.sleep(2)  # stagger to avoid a redis/ssh connection burst

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written to {out_dir / 'manifest.json'}")

    print(f"Waiting for {len(procs)} runs to finish (poll every {args.poll_interval}s, timeout {args.timeout}s)...")
    start = time.time()
    while time.time() - start < args.timeout:
        done_flags = [is_finished(lp) for _, lp in procs]
        if all(done_flags):
            print("All runs finished.")
            break
        print(f"  {sum(done_flags)}/{len(procs)} finished ({int(time.time() - start)}s elapsed)")
        time.sleep(args.poll_interval)
    else:
        print("WARNING: timeout reached before all runs finished — proceeding with whatever is done.",
              file=sys.stderr)

    if not args.skip_report:
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "build_report.py"),
             "--manifest", str(out_dir / "manifest.json")],
            check=True,
        )


if __name__ == "__main__":
    main()
