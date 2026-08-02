#!/usr/bin/env python3
"""Launch with-agent / without-agent cost-prediction ablation runs for a set
of gigaevo problems, wait for completion, then build the comparison report.

Usage (from repo root):

    python3 tools/cost_ablation/run_ablation.py \\
        --tasks algotune/algotune_lqr algotune/algotune_markowitz \\
                adversarial/code/pop_a toy_kadane \\
        --max-mutants 100 --llm single \\
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
from pathlib import Path
import subprocess
import sys
import time
import zlib

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


def launch(
    task: str,
    db: int,
    max_mutants: int,
    llm: str,
    cost_monitor: str,
    log_path: Path,
    seed: int | None = None,
    overrides: list[str] | None = None,
) -> subprocess.Popen:
    cmd = [
        "python3",
        "run.py",
        f"problem.name={task}",
        f"max_mutants={max_mutants}",
        f"llm={llm}",
        f"redis.db={db}",
        "redis.resume=true",
        f"+cost_monitor={cost_monitor}",
    ]
    if seed is not None:
        cmd.append(f"+seed={seed}")
    if overrides:
        cmd.extend(overrides)
    log_f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )


def is_finished(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "Duration:" in text


def seed_for(base: int, key: str, rep: int) -> int:
    """Seed for one (task, replicate) — shared by BOTH arms of that pair.

    crc32, not ``hash()``: Python randomises string hashing per process, so
    ``hash()`` would give a different seed every time the driver is invoked
    and the experiment would not be reproducible across restarts.
    """
    return base + rep * 1000 + zlib.crc32(key.encode()) % 997


def write_manifest(out_dir: Path, manifest: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def run_wave(
    tasks: list[str], out_dir: Path, args, manifest: list[dict], rep: int = 0
) -> None:
    """Launch both conditions for every task in ``tasks``, then wait for them.

    Appends to ``manifest`` and flushes it to disk after EVERY launch, not
    once the wave has finished: the manifest is what tools/cost_lab_web reads
    to know an experiment exists at all, and a wave takes tens of minutes.
    Writing it only at the end made a freshly launched experiment invisible
    in the console for the whole first wave.
    """
    dbs = find_free_dbs(len(tasks) * len(CONDITIONS))

    procs = []
    i = 0
    for task in tasks:
        key = task.replace("/", "_")
        # ONE seed per (task, replicate): both arms of a pair see the same
        # model-routing draw, so the comparison is matched rather than two
        # independent samples. Pairing removes the shared variance and is the
        # cheapest statistical power available at a fixed compute budget.
        seed = None if args.seed is None else seed_for(args.seed, key, rep)
        for cond_name, cost_monitor in CONDITIONS:
            db = dbs[i]
            i += 1
            suffix = f"_rep{rep}" if rep else ""
            log_path = out_dir / f"{key}_{cond_name}{suffix}.log"
            proc = launch(
                task, db, args.max_mutants, args.llm, cost_monitor, log_path, seed
            )
            manifest.append(
                {
                    "task": task,
                    "key": key,
                    "condition": cond_name,
                    "db": db,
                    "replicate": rep,
                    "seed": seed,
                    "log": str(log_path.relative_to(REPO_ROOT)),
                    "pid": proc.pid,
                }
            )
            write_manifest(out_dir, manifest)
            procs.append((proc, log_path))
            print(
                f"launched {task} [{cond_name}] db={db} pid={proc.pid} -> {log_path}",
                flush=True,
            )
            time.sleep(2)  # stagger to avoid a redis/ssh connection burst

    print(
        f"Waiting for {len(procs)} runs (poll {args.poll_interval}s, timeout {args.timeout}s)...",
        flush=True,
    )
    start = time.time()
    while time.time() - start < args.timeout:
        done_flags = [is_finished(lp) for _, lp in procs]
        if all(done_flags):
            print("Wave finished.", flush=True)
            break
        print(
            f"  {sum(done_flags)}/{len(procs)} finished ({int(time.time() - start)}s elapsed)",
            flush=True,
        )
        time.sleep(args.poll_interval)
    else:
        print(
            "WARNING: wave timed out — killing stragglers and moving on.",
            file=sys.stderr,
            flush=True,
        )
        for proc, _ in procs:
            if proc.poll() is None:
                proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="problem.name values, e.g. algotune/algotune_lqr toy_kadane",
    )
    ap.add_argument("--max-mutants", type=int, default=100)
    ap.add_argument("--llm", default="single")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="relative to repo root, e.g. experiments/cost_ablation_20260801_120000",
    )
    ap.add_argument(
        "--poll-interval", type=int, default=60, help="seconds between health checks"
    )
    ap.add_argument(
        "--timeout", type=int, default=3600 * 3, help="max seconds to wait per wave"
    )
    ap.add_argument(
        "--wave-size",
        type=int,
        default=0,
        help="tasks launched concurrently (0 = all at once). Each task costs 2 runs, "
        "and every run holds up to max_in_flight LLM slots — oversubscribing the "
        "server queues calls and distorts the achieved-concurrency measurement.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed. Both arms of a task get the SAME seed, so the two "
        "conditions are compared on matched model-routing draws instead of "
        "two independent samples. Omit for the old unpaired behaviour.",
    )
    ap.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="how many seeds per task. A single-seed ablation can rank two "
        "variants the opposite way from another seed, so >1 is what makes "
        "a result reportable.",
    )
    ap.add_argument(
        "--skip-report",
        action="store_true",
        help="only launch + wait, skip building the report",
    )
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    # Empty manifest up front: the console lists an experiment by its manifest,
    # so this is what makes the run appear the moment the driver starts rather
    # than after the first wave (redis db scan + stagger take a while).
    write_manifest(out_dir, [])

    step = args.wave_size if args.wave_size > 0 else len(args.tasks)
    manifest: list[dict] = []
    for rep in range(max(args.replicates, 1)):
        for w, start_i in enumerate(range(0, len(args.tasks), step), 1):
            batch = args.tasks[start_i : start_i + step]
            print(f"=== replicate {rep} wave {w}: {' '.join(batch)}", flush=True)
            run_wave(batch, out_dir, args, manifest, rep=rep)
            write_manifest(out_dir, manifest)
    print(f"manifest written to {out_dir / 'manifest.json'}", flush=True)

    if not args.skip_report:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "build_report.py"),
                "--manifest",
                str(out_dir / "manifest.json"),
            ],
            check=True,
        )


def _selftest() -> None:
    """The one property the paired design rests on: both arms of a task share
    a seed, and that seed is stable across driver restarts."""
    a = seed_for(7, "alphaevolve_packing_circles_n_26", 0)
    b = seed_for(7, "alphaevolve_packing_circles_n_26", 0)
    assert a == b, "same task+replicate must give the same seed"
    assert a != seed_for(7, "alphaevolve_packing_circles_n_26", 1), (
        "replicates must differ"
    )
    assert a != seed_for(7, "alphaevolve_erdos_minimum_overlap", 0), "tasks must differ"
    # stable across processes — the whole point of not using hash()
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,zlib;sys.path.insert(0,'tools/cost_ablation');"
            "from run_ablation import seed_for;print(seed_for(7,'alphaevolve_packing_circles_n_26',0))",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == str(a), (out.stdout, out.stderr)
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
