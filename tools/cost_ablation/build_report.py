#!/usr/bin/env python3
"""Parse run_ablation.py logs and build a LaTeX comparison table (+ image)
and per-task prediction-error-decay plots.

Usage:
    python3 tools/cost_ablation/build_report.py --manifest experiments/cost_ablation_.../manifest.json

For each (task, condition) log this parses:
  - every ``[CostMonitorHookJSON] {...}`` line -> a time series of
    predicted_tokens / predicted_duration_s (+ CI) over the run
  - the final ``[TokenTracker:<model>] ... (cumulative: N)`` line per model
    -> actual total tokens (summed across models)
  - the ``Duration: Xs`` line -> actual wall-clock duration

It writes into ``<out-dir>/report/``:
  - comparison_table.tex   (booktabs LaTeX source, no compiler required)
  - comparison_table.png   (matplotlib rendering of the same table — always
                             produced, since a LaTeX toolchain may not be
                             installed on the machine running the ablation)
  - error_decay_plots.png  (one subplot per task, one line per condition:
                             |predicted - actual| / actual, in % vs run
                             progress 0->1 — should trend down)

Run with --selftest (no manifest needed) to sanity-check the parser against
an in-memory synthetic log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

CMJ_RE = re.compile(r"\[CostMonitorHookJSON\] (\{.*\})")
DUR_RE = re.compile(r"Duration: ([\d.]+)s")
TOK_RE = re.compile(
    r"\[TokenTracker:(\w+)\] ([\w.\-]+): \d+ ctx \+ \d+ gen \(\d+ reasoning\) = \d+ \(cumulative: (\d+)\)"
)

COND_LABEL = {"noagent": "no agent", "withagent": "with agent"}


def parse_log(path: Path) -> tuple[list[dict], float | None, int]:
    """Returns (series, actual_duration_s, actual_total_tokens)."""
    series: list[dict] = []
    duration: float | None = None
    tok_cum: dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = CMJ_RE.search(line)
            if m:
                try:
                    series.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass
                continue
            m = DUR_RE.search(line)
            if m:
                duration = float(m.group(1))
                continue
            m = TOK_RE.search(line)
            if m:
                tok_cum[m.group(2)] = int(m.group(3))
    return series, duration, sum(tok_cum.values())


def summarize(repo_root: Path, entry: dict) -> dict | None:
    path = repo_root / entry["log"]
    if not path.exists():
        return None
    series, duration, actual_tokens = parse_log(path)
    if not series or not duration or not actual_tokens:
        return None
    last = series[-1]
    pred_tokens, pred_duration = last["predicted_tokens"], last["predicted_duration_s"]
    return dict(
        task=entry["task"],
        key=entry["key"],
        condition=entry["condition"],
        n_points=len(series),
        series=series,
        pred_tokens=pred_tokens,
        pred_duration=pred_duration,
        actual_tokens=actual_tokens,
        actual_duration=duration,
        tok_err_pct=(pred_tokens - actual_tokens) / actual_tokens * 100,
        dur_err_pct=(pred_duration - duration) / duration * 100,
    )


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def build_table_tex(rows_by_task: dict[str, dict[str, dict]]) -> str:
    lines = [
        r"\documentclass[border=4pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\begin{document}",
        r"\small",
        r"\begin{tabular}{l l r r r r r r r}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Condition} & \textbf{Pred.\ tok} & \textbf{Act.\ tok} & "
        r"\textbf{Tok.\ err \%} & \textbf{Pred.\ dur (s)} & \textbf{Act.\ dur (s)} & "
        r"\textbf{Dur.\ err \%} & \textbf{N pts} \\",
        r"\midrule",
    ]
    for key, conds in rows_by_task.items():
        first = True
        for cond in ("noagent", "withagent"):
            s = conds.get(cond)
            if not s:
                continue
            task_cell = esc(key) if first else ""
            color = r"\rowcolor{green!12}" if cond == "withagent" else ""
            lines.append(
                f"{color}{task_cell} & {COND_LABEL[cond]} & "
                f"{s['pred_tokens']:,} & {s['actual_tokens']:,} & {s['tok_err_pct']:+.1f} & "
                f"{s['pred_duration']:.0f} & {s['actual_duration']:.0f} & {s['dur_err_pct']:+.1f} & "
                f"{s['n_points']} \\\\"
            )
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def build_table_png(rows_by_task: dict[str, dict[str, dict]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table_rows = [
        [
            "Task",
            "Condition",
            "Pred tok",
            "Act tok",
            "Tok err %",
            "Pred dur (s)",
            "Act dur (s)",
            "Dur err %",
            "N pts",
        ]
    ]
    row_colors = [None]
    for key, conds in rows_by_task.items():
        first = True
        for cond in ("noagent", "withagent"):
            s = conds.get(cond)
            if not s:
                continue
            table_rows.append(
                [
                    key if first else "",
                    COND_LABEL[cond],
                    f"{s['pred_tokens']:,}",
                    f"{s['actual_tokens']:,}",
                    f"{s['tok_err_pct']:+.1f}",
                    f"{s['pred_duration']:.0f}",
                    f"{s['actual_duration']:.0f}",
                    f"{s['dur_err_pct']:+.1f}",
                    str(s["n_points"]),
                ]
            )
            row_colors.append("#e8f7ee" if cond == "withagent" else None)
            first = False

    fig_h = 0.45 * len(table_rows) + 0.5
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for r, color in enumerate(row_colors):
        for c in range(len(table_rows[0])):
            cell = tbl[r, c]
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#dddddd")
            elif color:
                cell.set_facecolor(color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_error_decay_png(
    rows_by_task: dict[str, dict[str, dict]], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(rows_by_task)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    colors = {"noagent": "#8b96a3", "withagent": "#ffb454"}

    for ax, (key, conds) in zip(axes_flat, rows_by_task.items()):
        for cond in ("noagent", "withagent"):
            s = conds.get(cond)
            if not s:
                continue
            xs = np.arange(len(s["series"])) / max(len(s["series"]) - 1, 1)
            errs = [
                abs(p["predicted_duration_s"] - s["actual_duration"])
                / s["actual_duration"]
                * 100
                for p in s["series"]
            ]
            ax.plot(xs, errs, label=COND_LABEL[cond], color=colors[cond], linewidth=1.8)
        ax.set_title(key, fontsize=11, fontweight="bold")
        ax.set_xlabel("run progress (0->1)")
        ax.set_ylabel("|duration pred error|, %")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Prediction error over time: with agent vs without agent", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


GRID = [round(0.05 * i, 2) for i in range(1, 21)]  # 5%, 10%, ... 100%


def err_at(s: dict, p: float, field: str) -> float:
    """Signed % error of ``field`` at run progress ``p`` (0..1].

    Progress is share of the run's own prediction points, so conditions that
    emitted different numbers of points are still compared at the same stage
    of the run rather than at the same list index.
    """
    series = s["series"]
    i = min(len(series) - 1, max(0, int(round(p * len(series))) - 1))
    actual = (
        s["actual_duration"] if field == "predicted_duration_s" else s["actual_tokens"]
    )
    return (series[i][field] - actual) / actual * 100


def build_progress_grid(
    rows_by_task: dict[str, dict[str, dict]], out_dir: Path
) -> None:
    """Error-vs-progress grid: the headline comparison, table + figure.

    CSV holds every (task, condition, progress) cell; the figure shows the
    median |error| across tasks, which is what the run-level claim rests on
    (mean is hostage to a single cold-start outlier).
    """
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fields = [("predicted_duration_s", "duration"), ("predicted_tokens", "tokens")]

    with open(out_dir / "progress_grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["task", "condition", "progress", "metric", "signed_err_pct", "abs_err_pct"]
        )
        for key, conds in rows_by_task.items():
            for cond, s in conds.items():
                for field, label in fields:
                    for p in GRID:
                        e = err_at(s, p, field)
                        w.writerow(
                            [key, cond, f"{p:.2f}", label, f"{e:.2f}", f"{abs(e):.2f}"]
                        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"noagent": "#8b96a3", "withagent": "#2a9d5c"}
    for ax, (field, label) in zip(axes, fields):
        for cond in ("noagent", "withagent"):
            series = [
                [abs(err_at(s, p, field)) for p in GRID]
                for conds in rows_by_task.values()
                if (s := conds.get(cond))
            ]
            if not series:
                continue
            arr = np.array(series)
            med = np.median(arr, axis=0)
            ax.plot(
                [p * 100 for p in GRID],
                med,
                color=colors[cond],
                linewidth=2.2,
                marker="o",
                markersize=4,
                label=f"{COND_LABEL[cond]} (n={len(series)})",
                linestyle="-" if cond == "withagent" else (0, (4, 2)),
            )
            ax.fill_between(
                [p * 100 for p in GRID],
                np.percentile(arr, 25, axis=0),
                np.percentile(arr, 75, axis=0),
                color=colors[cond],
                alpha=0.13,
            )
        ax.axhline(10, color="#c0c0c0", linestyle=":", linewidth=1)
        ax.set_xlabel("run progress, %")
        ax.set_ylabel(f"|{label} prediction error|, %")
        ax.set_title(
            f"{label.capitalize()} error vs progress (median, IQR band)",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Automatic estimator vs estimator + CostMonitorAgent, alphaevolve tasks",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "error_vs_progress.png", dpi=160)
    plt.close(fig)


def build_grid_table_png(
    rows_by_task: dict[str, dict[str, dict]], out_dir: Path
) -> None:
    """Median |error| per checkpoint as a compact two-row-per-metric table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    checkpoints = [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
    header = ["Metric", "Condition"] + [f"{int(p * 100)}%" for p in checkpoints]
    body, colors = [header], [None]
    for field, label in [
        ("predicted_duration_s", "duration"),
        ("predicted_tokens", "tokens"),
    ]:
        for cond in ("noagent", "withagent"):
            vals = []
            for p in checkpoints:
                col = [
                    abs(err_at(s, p, field))
                    for conds in rows_by_task.values()
                    if (s := conds.get(cond))
                ]
                vals.append(f"{np.median(col):.1f}" if col else "-")
            body.append([label if cond == "noagent" else "", COND_LABEL[cond]] + vals)
            colors.append("#e3f5ea" if cond == "withagent" else None)

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(body) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=body, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for r, color in enumerate(colors):
        for c in range(len(header)):
            cell = tbl[r, c]
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#dddddd")
            elif color:
                cell.set_facecolor(color)
    ax.set_title(
        "Median |prediction error| %, by run progress", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "progress_grid_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # entry["log"] paths are stored relative to the repo root (see
    # run_ablation.py), independent of how deep --out-dir is nested — so
    # resolve repo root from this script's own location, not from manifest_path.
    repo_root = Path(__file__).resolve().parents[2]

    rows_by_task: dict[str, dict[str, dict]] = {}
    for entry in manifest:
        s = summarize(repo_root, entry)
        if not s:
            print(
                f"skipping {entry['key']}[{entry['condition']}]: no usable data (crashed / no CostMonitorHookJSON)"
            )
            continue
        rows_by_task.setdefault(entry["key"], {})[entry["condition"]] = s

    out_dir = manifest_path.parent / "report"
    out_dir.mkdir(exist_ok=True)

    tex_src = build_table_tex(rows_by_task)
    (out_dir / "comparison_table.tex").write_text(tex_src, encoding="utf-8")
    build_table_png(rows_by_task, out_dir / "comparison_table.png")
    build_error_decay_png(rows_by_task, out_dir / "error_decay_plots.png")
    build_progress_grid(rows_by_task, out_dir)
    build_grid_table_png(rows_by_task, out_dir)

    print(f"report written to {out_dir}")
    return out_dir


def _selftest() -> None:
    """Assert-based self-check: parse_log() against an in-memory synthetic log."""
    import tempfile

    fake_log = (
        "2026-01-01 00:00:00.000 | INFO | x | [CostMonitorHookJSON] "
        '{"mutant": 0, "predicted_tokens": 1000, "token_ci_low": 500, "token_ci_high": 1500, '
        '"predicted_duration_s": 10.0, "ci_low_s": 5.0, "ci_high_s": 15.0}\n'
        "2026-01-01 00:00:01.000 | INFO | x | [CostMonitorHookJSON] "
        '{"mutant": 1, "predicted_tokens": 900, "token_ci_low": 600, "token_ci_high": 1200, '
        '"predicted_duration_s": 9.5, "ci_low_s": 6.0, "ci_high_s": 13.0}\n'
        "2026-01-01 00:00:02.000 | DEBUG | x | [TokenTracker:default] qwen: 100 ctx + 50 gen (0 reasoning) "
        "= 150 (cumulative: 950)\n"
        "2026-01-01 00:00:03.000 | INFO | __main__:run_experiment:115 | Duration: 42.0s (0.01h)\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write(fake_log)
        tmp_path = Path(f.name)
    try:
        series, duration, actual_tokens = parse_log(tmp_path)
        assert len(series) == 2, series
        assert series[-1]["predicted_tokens"] == 900
        assert duration == 42.0
        assert actual_tokens == 950
        s = summarize(
            tmp_path.parent,
            {"task": "t", "key": "t", "condition": "noagent", "log": tmp_path.name},
        )
        assert s is not None
        assert s["tok_err_pct"] == (900 - 950) / 950 * 100

        # err_at: p=1.0 must land on the last point (== the final-table number),
        # small p on the first, and never index out of range for any grid value.
        assert err_at(s, 1.0, "predicted_tokens") == s["tok_err_pct"]
        assert err_at(s, 0.05, "predicted_tokens") == (1000 - 950) / 950 * 100
        for p in GRID:
            for f in ("predicted_tokens", "predicted_duration_s"):
                assert isinstance(err_at(s, p, f), float)
        print("selftest OK")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--manifest", type=Path, help="path to manifest.json written by run_ablation.py"
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="run the built-in parser self-check and exit",
    )
    args = ap.parse_args()

    if args.selftest:
        _selftest()
    elif args.manifest:
        run(args.manifest)
    else:
        ap.error("either --manifest or --selftest is required")
