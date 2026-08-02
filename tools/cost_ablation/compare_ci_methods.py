#!/usr/bin/env python3
"""Compare interval coverage and relative width across replay methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loguru import logger
from replay_from_log import CI_METHODS_TO_COMPARE, replay_log  # noqa: E402

CHECKPOINT_FRACS = [0.10, 0.25, 0.50, 1.0]
# Exclude a watchdog-terminated run whose actual horizon differs from the model.
EXCLUDE_FROM_AGGREGATE = {"toy_kadane_withagent_postfix"}


def _method_key(method: str | None) -> str:
    return method or "heuristic"


def evaluate_log(path: Path) -> dict | None:
    result = replay_log(path, ci_checkpoint_fracs=CHECKPOINT_FRACS)
    if not result or not result["ci_checkpoints"]:
        return None
    actual_tokens = result["actual_tokens"]
    actual_duration = result["actual_duration"]
    rows = []
    for ckpt in result["ci_checkpoints"]:
        row = {"frac": ckpt["frac"], "attempts": ckpt["attempts"]}
        for method in CI_METHODS_TO_COMPARE:
            key = _method_key(method)
            tok_lo, tok_hi = ckpt[key]["tokens_ci"]
            dur_lo, dur_hi = ckpt[key]["duration_ci"]
            row[key] = dict(
                tok_covered=tok_lo <= actual_tokens <= tok_hi,
                tok_rel_width=(tok_hi - tok_lo) / actual_tokens
                if actual_tokens
                else float("nan"),
                dur_covered=dur_lo <= actual_duration <= dur_hi,
                dur_rel_width=(dur_hi - dur_lo) / actual_duration
                if actual_duration
                else float("nan"),
            )
        rows.append(row)
    return dict(
        path=str(path),
        name=path.stem,
        checkpoints=rows,
        actual_tokens=actual_tokens,
        actual_duration=actual_duration,
    )


def aggregate(per_log: list[dict]) -> dict:
    """Median coverage/width per method per checkpoint fraction, across logs
    (excluding EXCLUDE_FROM_AGGREGATE)."""
    included = [r for r in per_log if r["name"] not in EXCLUDE_FROM_AGGREGATE]
    out: dict[float, dict[str, dict[str, float]]] = {}
    for frac in CHECKPOINT_FRACS:
        out[frac] = {}
        for method in CI_METHODS_TO_COMPARE:
            key = _method_key(method)
            vals = [
                r["checkpoints"][CHECKPOINT_FRACS.index(frac)][key] for r in included
            ]
            out[frac][key] = dict(
                tok_coverage=statistics.mean(v["tok_covered"] for v in vals),
                tok_median_width=statistics.median(v["tok_rel_width"] for v in vals),
                dur_coverage=statistics.mean(v["dur_covered"] for v in vals),
                dur_median_width=statistics.median(v["dur_rel_width"] for v in vals),
            )
    return out


METHOD_COLOR = {
    "heuristic": "#8b96a3",
    "bootstrap": "#2a6fb0",
    "montecarlo": "#c9a227",
    "bayesian": "#8b3fa8",
}


def build_comparison_png(agg: dict, out: Path) -> None:
    """Coverage vs. relative width, grouped by checkpoint, one panel each for
    tokens and duration. A method that's both narrower (shorter bar in the
    bottom row) and better-covering (taller bar in the top row, closer to
    the 90% target line) beats the heuristic; narrower-but-under-covering
    just means it got lucky on 8 runs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = list(CI_METHODS_TO_COMPARE)
    method_keys = [_method_key(m) for m in methods]
    fracs = CHECKPOINT_FRACS
    x = np.arange(len(fracs))
    bw = 0.8 / len(method_keys)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("tok_coverage", "token CI coverage", axes[0, 0], True),
        ("dur_coverage", "duration CI coverage", axes[0, 1], True),
        ("tok_median_width", "token CI median relative width", axes[1, 0], False),
        ("dur_median_width", "duration CI median relative width", axes[1, 1], False),
    ]
    for field, title, ax, is_coverage in panels:
        for i, key in enumerate(method_keys):
            vals = [agg[frac][key][field] for frac in fracs]
            ax.bar(x + i * bw, vals, width=bw, label=key, color=METHOD_COLOR[key])
        if is_coverage:
            ax.axhline(
                0.9,
                color="#333",
                linestyle="--",
                linewidth=1,
                label="90% target (alpha=0.1)",
            )
            ax.set_ylim(0, 1.05)
        ax.set_xticks(x + bw * (len(method_keys) - 1) / 2)
        ax.set_xticklabels([f"{f:.0%}" for f in fracs])
        ax.set_xlabel("run progress at checkpoint")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "CI-construction method comparison: heuristic vs. bootstrap / Monte Carlo / Bayesian\n"
        "(replayed on the 9 collected logs, final-checkpoint columns collapse to a point — see README)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_dashboard_html(
    agg: dict, per_log: list[dict], png_name: str, out: Path
) -> None:
    """Self-contained HTML page: the comparison figure plus the raw
    coverage/width table, so the result can be opened in a browser without
    re-running anything."""
    rows_html = []
    for frac in CHECKPOINT_FRACS:
        for method in CI_METHODS_TO_COMPARE:
            key = _method_key(method)
            m = agg[frac][key]
            rows_html.append(
                f"<tr><td>{frac:.0%}</td><td>{key}</td>"
                f"<td>{m['tok_coverage']:.0%}</td><td>{m['tok_median_width']:.2f}</td>"
                f"<td>{m['dur_coverage']:.0%}</td><td>{m['dur_median_width']:.2f}</td></tr>"
            )
    logs_html = "".join(
        f"<li>{r['name']}: actual {r['actual_tokens']:,} tokens, {r['actual_duration']:.0f}s"
        f"{' (excluded from aggregate — stalled early)' if r['name'] in EXCLUDE_FROM_AGGREGATE else ''}</li>"
        for r in per_log
    )
    out.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<title>CI method comparison — cost model</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ padding: 0.35rem 0.7rem; text-align: right; border-bottom: 1px solid #ddd; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #f2f2f2; }}
img {{ max-width: 100%; }}
</style></head><body>
<h1>CI-construction method comparison</h1>
<p>Replays the 9 collected logs (tools/cost_ablation/compare_ci_methods.py) through
growth_estimator's four CI methods — the current 1/sqrt(n) heuristic and three
probabilistic alternatives (bootstrap, Monte Carlo, closed-form Bayesian) — at
10/25/50/100% run progress, and checks whether the run's actual final
tokens/duration fell inside each method's published interval.</p>
<img src="{png_name}" alt="coverage and width comparison">
<h2>Aggregate (median across {len(per_log) - len(EXCLUDE_FROM_AGGREGATE)} runs)</h2>
<table><tr><th>progress</th><th>method</th><th>token coverage</th><th>token width</th>
<th>duration coverage</th><th>duration width</th></tr>
{"".join(rows_html)}</table>
<h2>Logs replayed</h2>
<ul>{logs_html}</ul>
<p><em>Note: at the 100% checkpoint there's no tail left to be uncertain about, so
all three probabilistic methods fall back to the heuristic width for tokens, and
the duration CI collapses to a single point (width 0) regardless of method — that
column measures whether the point estimate is exactly exact, not CI quality.</em></p>
</body></html>""",
        encoding="utf-8",
    )


def print_table(agg: dict) -> None:
    print(
        f"\n{'frac':>5s} {'method':>10s} {'tok_cov':>8s} {'tok_width':>10s} "
        f"{'dur_cov':>8s} {'dur_width':>10s}"
    )
    for frac in CHECKPOINT_FRACS:
        for method in CI_METHODS_TO_COMPARE:
            key = _method_key(method)
            m = agg[frac][key]
            print(
                f"{frac:5.2f} {key:>10s} {m['tok_coverage']:8.0%} {m['tok_median_width']:10.2f} "
                f"{m['dur_coverage']:8.0%} {m['dur_median_width']:10.2f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+", type=Path, help="run.py log file(s) to replay")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="if set, write comparison.json + ci_methods_dashboard.html here",
    )
    args = ap.parse_args()

    logger.remove()  # replay re-logs every event at INFO; noisy for a batch tool

    per_log = []
    for log_path in args.logs:
        result = evaluate_log(log_path)
        if not result:
            print(f"{log_path.name}: skipped (no events / no ground truth)")
            continue
        per_log.append(result)
        print(
            f"{log_path.name}: {len(result['checkpoints'])} checkpoints, "
            f"actual_tokens={result['actual_tokens']:,}, actual_duration={result['actual_duration']:.0f}s"
        )

    if not per_log:
        print("no logs produced usable checkpoints")
        return

    agg = aggregate(per_log)
    print_table(agg)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "comparison.json").write_text(
            json.dumps(
                {"per_log": per_log, "aggregate": {str(k): v for k, v in agg.items()}},
                indent=2,
            ),
            encoding="utf-8",
        )
        build_comparison_png(agg, args.out_dir / "ci_method_comparison.png")
        build_dashboard_html(
            agg, per_log, "ci_method_comparison.png", args.out_dir / "dashboard.html"
        )
        print(
            f"\nreport written to {args.out_dir} (comparison.json, ci_method_comparison.png, dashboard.html)"
        )


if __name__ == "__main__":
    main()
