#!/usr/bin/env python3
"""Run the seven-domain validation set with the default stall timeout."""

from pathlib import Path
import subprocess
import sys
import time

import redis

REPO = Path(__file__).resolve().parents[2]

TASKS = [
    ("algotune/algotune_lqr", 92),
    ("algotune/algotune_kalman_filter", 96),
    ("algotune/algotune_markowitz", 97),
    ("algotune/algotune_convolve_1d", 98),
    ("tabular_regression", 93),
    ("chains/nlp/gsm8k/static", 94),
    ("chains/nlp/hotpotqa/static", 99),
]
MAX_MUTANTS = 15
PER_DOMAIN_TIMEOUT = 900


def main() -> None:
    logs_dir = REPO / "experiments" / f"rerun_7domains_{int(time.time())}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for task, db in TASKS:
        safe = task.replace("/", "_")
        print("=" * 70)
        print(f"START [{task}] (db={db})")
        try:
            redis.Redis(db=db).flushdb()
        except Exception as e:
            print("flush error:", e)

        cmd = [
            sys.executable,
            str(REPO / "run.py"),
            f"problem.name={task}",
            f"max_mutants={MAX_MUTANTS}",
            "llm=summer_school_servers",
            f"redis.db={db}",
            "redis.resume=true",
            "+cost_monitor=enabled",
        ]
        log_file = logs_dir / f"{safe}.log"
        print("Command:", " ".join(cmd))

        t0 = time.time()
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=PER_DOMAIN_TIMEOUT)
                status = "OK" if proc.returncode == 0 else "FAIL"
                code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                status = "TIMEOUT"
                code = -1
        duration = time.time() - t0
        print(f"[{task}] {int(duration)}s -> {status} (exit={code})")
        summary.append((task, db, code, int(duration), status))

    print("\n=== SUMMARY ===")
    with open(logs_dir / "summary.tsv", "w") as f:
        f.write("task\tredis_db\texit_code\tduration_s\tstatus\n")
        for row in summary:
            f.write("\t".join(str(x) for x in row) + "\n")
    for row in summary:
        print(row)
    print("\nLogs:", logs_dir)


if __name__ == "__main__":
    main()
