#!/usr/bin/env python3
"""What is left after a constant, and is any of it predictable?

`calibrate_tail.py` showed that a fixed multiplier on the remaining term
removes about half the estimator's error at 10% progress and a third at 25%,
and that what stays is spread BETWEEN tasks — one task needs 1.0, another
1.6. A constant cannot reach that by construction; per-task adaptation is the
only thing that can, and it is the only job the cost-monitor agent can hold
that a formula cannot take from it.

So: is that residual predictable from anything observable at the time?

For every log, at every checkpoint, this records
  * the multiplier the remaining term SHOULD have had, and
  * features the agent could be shown at that moment,
then ranks the features by Spearman correlation with the multiplier.

A feature that correlates is one to put in the prompt. If nothing correlates,
the residual is task identity the agent has no way to see, and that has to be
known before building anything on top of it.

Usage:
    python3 tools/cost_ablation/residual_features.py experiments/*/*noagent*.log
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics as st
import sys
import tempfile

import numpy as np
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from replay_from_log import hook_from_log, replay_log  # noqa: E402

GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
FOCUS = (0.10, 0.25)  # where a budget decision is actually taken


def truncate(path: Path, frac: float) -> Path | None:
    """A copy of the log cut at ``frac`` of its mutation attempts."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    total = txt.count("[MUTATION_ATTEMPTED]")
    cut = int(total * frac)
    if cut < 2:
        return None
    out, seen = [], 0
    for line in txt.splitlines(True):
        out.append(line)
        if "[MUTATION_ATTEMPTED]" in line:
            seen += 1
            if seen >= cut:
                break
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8")
    f.writelines(out)
    f.close()
    return Path(f.name)


def features(hook) -> dict[str, float]:
    """Everything the agent could be told, computed from what it already has."""
    lat = [x for v in hook._latency_by_stage.values() for x in v]
    tok = [x for v in hook._tokens_by_stage.values() for x in v]
    nonllm = [x for v in hook._nonllm_duration_by_stage.values() for x in v]
    llm_total = sum(lat) or 1.0
    med_lat = st.median(lat) if lat else 0.0
    f = {
        "стадий_LLM": float(len(hook._tokens_by_stage)),
        "вызовов_на_попытку": len(hook._call_history) / max(hook._attempts, 1),
        "доля_не_LLM_времени": sum(nonllm) / (llm_total + sum(nonllm) or 1.0),
        "разброс_латентности": (max(lat) / med_lat) if med_lat else 0.0,
        "кв_отклонение_латентности": (st.pstdev(lat) / med_lat)
        if med_lat and len(lat) > 1
        else 0.0,
        "медиана_токенов": st.median(tok) if tok else 0.0,
        "конкурентность": hook._concurrency,
        "конкурентность_к_капу": hook._concurrency / max(hook._pred.max_in_flight, 1),
        "доля_упавших_вызовов": sum(1 for r in hook._call_history if r.tokens_in == 0)
        / max(len(hook._call_history), 1),
    }
    return f


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+", type=Path)
    args = ap.parse_args()
    logger.remove()

    # multiplier per (checkpoint, log) and the features that went with it
    mult: dict[float, list[float]] = {p: [] for p in GRID}
    feats: dict[float, list[dict[str, float]]] = {p: [] for p in GRID}
    fams: dict[float, list[str]] = {p: [] for p in GRID}

    for path in args.logs:
        try:
            full = replay_log(path)
        except Exception as exc:  # noqa: BLE001
            print(f"пропуск {path.name}: {exc}", flush=True)
            continue
        if not full:
            continue
        actual, series = full["actual_duration"], full["series"]
        family = path.stem.split("_noagent")[0]
        for p in GRID:
            i = min(len(series) - 1, max(0, round(p * len(series)) - 1))
            pt = series[i]
            elapsed = pt.get("elapsed_s") or 0.0
            pred_rest, real_rest = (
                pt["predicted_duration_s"] - elapsed,
                actual - elapsed,
            )
            if pred_rest <= 1e-6 or real_rest <= 1e-6:
                continue
            cut = truncate(path, p)
            if cut is None:
                continue
            try:
                hook = hook_from_log(cut, ci_method=None)
            finally:
                cut.unlink(missing_ok=True)
            if hook is None or not hook._call_history:
                continue
            mult[p].append(real_rest / pred_rest)
            feats[p].append(features(hook))
            fams[p].append(family)

    print(f"\nразобрано логов: {len(set(f for v in fams.values() for f in v))}\n")

    print("=== множитель на остаток по прогрессу ===")
    print(
        f"{'прогресс':>9} {'n':>4} {'медиана':>9} {'IQR':>16} {'|err| после константы':>22}"
    )
    for p in GRID:
        ms = sorted(mult[p])
        if len(ms) < 4:
            continue
        k = st.median(ms)
        q1, q3 = ms[len(ms) // 4], ms[3 * len(ms) // 4]
        left = st.median([abs(1 - m / k) * 100 for m in ms])
        print(
            f"{p:>9.0%} {len(ms):>4} {k:>9.2f} {f'[{q1:.2f}, {q3:.2f}]':>16} {left:>21.1f}%"
        )

    print("\n=== чем предсказуем остаток (Spearman с множителем) ===")
    for p in FOCUS:
        rows, ms = feats[p], mult[p]
        if len(ms) < 8:
            continue
        print(f"\n-- прогресс {p:.0%}, n={len(ms)} --")
        scored = []
        for key in rows[0]:
            xs = np.array([r[key] for r in rows], dtype=float)
            if np.allclose(xs, xs[0]):
                continue
            rho, pv = sp_stats.spearmanr(xs, ms)
            scored.append((abs(rho), rho, pv, key))
        for _, rho, pv, key in sorted(scored, reverse=True):
            mark = "  <-- значимо" if pv < 0.05 else ""
            print(f"{key:>28} rho={rho:+.2f}  p={pv:.3f}{mark}")

        # Task identity is the ceiling: if the same task repeats with a
        # similar multiplier, the residual IS the task and the agent only has
        # to recognise which one it is.
        byfam: dict[str, list[float]] = {}
        for fam, m in zip(fams[p], ms):
            byfam.setdefault(fam, []).append(m)
        rep = {k: v for k, v in byfam.items() if len(v) > 1}
        if rep:
            within = st.median([max(v) / max(min(v), 1e-9) for v in rep.values()])
            allm = sorted(ms)
            across = allm[3 * len(allm) // 4] / max(allm[len(allm) // 4], 1e-9)
            print(
                f"{'разброс внутри задачи':>28} x{within:.2f}   "
                f"между задачами x{across:.2f}   (задач с повторами: {len(rep)})"
            )

        # Does knowing WHICH task this is beat knowing nothing? Leave-one-out,
        # because a per-task median that includes the point it is scoring
        # would flatter itself. This is the direct test of "show the agent the
        # task": if the task prior does not beat the global one, the task is
        # not the thing to show.
        glob_err, task_err, n_task = [], [], 0
        for i, m in enumerate(ms):
            others = [x for j, x in enumerate(ms) if j != i]
            same = [x for j, x in enumerate(ms) if j != i and fams[p][j] == fams[p][i]]
            glob_err.append(abs(1 - m / st.median(others)) * 100)
            if len(same) >= 2:
                task_err.append(abs(1 - m / st.median(same)) * 100)
                n_task += 1
        if n_task >= 6:
            # compare on the same subset, else the two are not comparable
            paired_glob = [
                g
                for g, f_ in zip(glob_err, fams[p])
                if sum(1 for x in fams[p] if x == f_) >= 3
            ]
            print(
                f"{'приор: общий':>28} {st.median(paired_glob):5.1f}%   "
                f"{'приор: по задаче':>20} {st.median(task_err):5.1f}%   (n={n_task})"
            )


if __name__ == "__main__":
    main()
