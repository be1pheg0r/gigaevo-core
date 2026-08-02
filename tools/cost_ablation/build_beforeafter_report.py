#!/usr/bin/env python3
"""Compare recorded predictions with current-estimator replay on the same logs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loguru import logger
from replay_from_log import replay_log  # noqa: E402

CMJ_RE = re.compile(r"\[CostMonitorHookJSON\] (\{.*\})")
DUR_RE = re.compile(r"Duration: ([\d.]+)s")
TOK_RE = re.compile(
    r"\[TokenTracker:\w+\] ([\w.\-]+): \d+ ctx \+ \d+ gen \(\d+ reasoning\) = \d+ \(cumulative: (\d+)\)"
)
LLM_RE = re.compile(r"\[LLM_CALL\] (\{.*\})")
ATT_RE = re.compile(r"\[MUTATION_ATTEMPTED\]")
ADJ_LINE_RE = re.compile(
    r"\[CostMonitorHook\] adjustments: cold=(\S+) golden=(\S+) growth=(\S+)"
)
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
BP_RE = re.compile(r'"max_in_flight": (\d+)')

# The two conditions compared on every figure: the deterministic estimator
# alone, and the same estimator with CostMonitorAgent's historical decisions
# replayed on top. Same log, same events — only the agent lever is toggled.
CONDS = ("auto", "agent")
COND_LABEL = {"auto": "auto", "agent": "auto + agent"}
COND_COLOR = {"auto": "#8b96a3", "agent": "#2a9d5c"}


def parse_before(path: Path) -> dict | None:
    """The log's own live predictions + ground truth + raw LLM telemetry."""
    series: list[dict] = []
    duration = None
    tok_cum: dict[str, int] = {}
    ev_tokens = 0
    llm_service_s = 0.0
    llm_first = llm_last = None
    max_in_flight = 8
    t0 = None
    attempts = 0
    agent_calls: list[int] = []  # attempt index of every agent invocation
    agent_effective: list[int] = []  # ...that actually moved a multiplier
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            mts = TS_RE.match(line)
            ts = None
            if mts:
                ts = datetime.strptime(mts.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
                if t0 is None:
                    t0 = ts
            if ATT_RE.search(line):
                attempts += 1
                continue
            m = ADJ_LINE_RE.search(line)
            if m:
                agent_calls.append(attempts)

                def _f(v):
                    try:
                        return float(v)
                    except ValueError:
                        return -1.0

                if any(_f(g) > 0 for g in m.groups()):
                    agent_effective.append(attempts)
                continue
            m = CMJ_RE.search(line)
            if m:
                try:
                    series.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass
                continue
            m = LLM_RE.search(line)
            if m:
                d = json.loads(m.group(1))
                ev_tokens += d["tokens_in"] + d["tokens_out"]
                llm_service_s += d["latency_ms"] / 1000.0
                start = (ts or 0) - d["latency_ms"] / 1000.0
                llm_first = start if llm_first is None else min(llm_first, start)
                llm_last = ts if llm_last is None else max(llm_last, ts)
                continue
            m = DUR_RE.search(line)
            if m:
                duration = float(m.group(1))
                continue
            m = TOK_RE.search(line)
            if m:
                tok_cum[m.group(1)] = int(m.group(2))
                continue
            m = BP_RE.search(line)
            if m:
                max_in_flight = int(m.group(1))
    actual_tokens = sum(tok_cum.values())
    if not series or not duration or not actual_tokens:
        return None
    last = series[-1]
    span = (llm_last - llm_first) if (llm_first is not None and llm_last) else 0.0
    return dict(
        n_points=len(series),
        series=series,
        pred_tokens=last["predicted_tokens"],
        pred_duration=last["predicted_duration_s"],
        actual_tokens=actual_tokens,
        actual_duration=duration,
        tok_err_pct=(last["predicted_tokens"] - actual_tokens) / actual_tokens * 100,
        dur_err_pct=(last["predicted_duration_s"] - duration) / duration * 100,
        event_tokens=ev_tokens,
        coverage_pct=ev_tokens / actual_tokens * 100,
        llm_service_s=llm_service_s,
        llm_span_s=span,
        max_in_flight=max_in_flight,
        attempts=attempts,
        agent_calls=agent_calls,
        agent_effective=agent_effective,
        achieved_concurrency=(llm_service_s / span) if span > 0 else 0.0,
    )


# ------------------------------------------------------------------ outputs


def at_mutation(s: dict, n_att: int, n: int) -> tuple[float, float, float, float]:
    """(pred_tok, tok_err%, pred_dur, dur_err%) as of mutation attempt ``n``.

    The two conditions are sampled at the same run PROGRESS, not the same list
    index: the shipped estimator flushed once per CostMonitorHookJSON line
    (~2x per attempt), the replay flushes once per attempt.
    """
    series = s["series"]
    i = min(len(series) - 1, max(0, round(n / max(n_att, 1) * len(series)) - 1))
    p = series[i]
    return (
        p["predicted_tokens"],
        (p["predicted_tokens"] - s["actual_tokens"]) / s["actual_tokens"] * 100,
        p["predicted_duration_s"],
        (p["predicted_duration_s"] - s["actual_duration"]) / s["actual_duration"] * 100,
    )


def build_table_tex(rows: dict[str, dict[str, dict]], pred_at: int) -> str:
    lines = [
        r"\documentclass[border=4pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\begin{document}",
        r"\small",
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        rf"\textbf{{Task}} & \textbf{{Model}} & \textbf{{Pred.\ tok @{pred_at}}} & \textbf{{Act.\ tok}} & "
        rf"\textbf{{Tok.\ err \%}} & \textbf{{Pred.\ dur @{pred_at} (s)}} & \textbf{{Act.\ dur (s)}} & "
        r"\textbf{Dur.\ err \%} \\",
        r"\midrule",
    ]
    for key, conds in rows.items():
        first = True
        n_att = conds["agent"]["n_points"]
        for cond in CONDS:
            s = conds.get(cond)
            if not s:
                continue
            pt, te, pd, de = at_mutation(s, n_att, pred_at)
            color = r"\rowcolor{green!12}" if cond == "after" else ""
            lines.append(
                f"{color}{(key.replace('_', chr(92) + '_')) if first else ''} & {COND_LABEL[cond]} & "
                f"{pt:,.0f} & {s['actual_tokens']:,} & {te:+.1f} & "
                f"{pd:.0f} & {s['actual_duration']:.0f} & {de:+.1f} \\\\"
            )
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{document}"]
    return "\n".join(lines)


def build_table_png(rows: dict[str, dict[str, dict]], out: Path, pred_at: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body = [
        [
            "Task",
            "Model",
            f"Pred tok @{pred_at}",
            "Act tok",
            "Tok err %",
            f"Pred dur @{pred_at} (s)",
            "Act dur (s)",
            "Dur err %",
        ]
    ]
    colors = [None]
    for key, conds in rows.items():
        first = True
        n_att = conds["agent"]["n_points"]
        for cond in CONDS:
            s = conds.get(cond)
            if not s:
                continue
            pt, te, pd, de = at_mutation(s, n_att, pred_at)
            body.append(
                [
                    key if first else "",
                    COND_LABEL[cond],
                    f"{pt:,.0f}",
                    f"{s['actual_tokens']:,}",
                    f"{te:+.1f}",
                    f"{pd:.0f}",
                    f"{s['actual_duration']:.0f}",
                    f"{de:+.1f}",
                ]
            )
            colors.append("#e3f5ea" if cond == "agent" else None)
            first = False

    fig, ax = plt.subplots(figsize=(13, 0.45 * len(body) + 0.5))
    ax.axis("off")
    widths = [0.22, 0.13, 0.11, 0.11, 0.09, 0.11, 0.11, 0.09]
    tbl = ax.table(cellText=body, cellLoc="center", loc="center", colWidths=widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for r, color in enumerate(colors):
        for c in range(len(body[0])):
            cell = tbl[r, c]
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#dddddd")
            elif color:
                cell.set_facecolor(color)
            if r > 0 and c in (4, 7) and body[r][c]:
                val = abs(float(body[r][c]))
                cell.set_text_props(color="#b03030" if val > 15 else "#1d7a41")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_error_decay_png(
    rows: dict[str, dict[str, dict]], out: Path, field: str, ylabel: str, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(rows)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.6 * nrows), squeeze=False)
    flat = axes.flatten()
    for ax, (key, conds) in zip(flat, rows.items()):
        n = len(conds["agent"]["series"])
        for cond in CONDS:
            s = conds[cond]
            actual = (
                s["actual_duration"]
                if field == "predicted_duration_s"
                else s["actual_tokens"]
            )
            xs = np.arange(1, len(s["series"]) + 1)
            errs = [abs(p[field] - actual) / actual * 100 for p in s["series"]]
            ax.plot(
                xs,
                errs,
                color=COND_COLOR[cond],
                linewidth=1.8,
                label=COND_LABEL[cond],
                linestyle="-" if cond == "agent" else (0, (4, 2)),
            )
        for j, pt in enumerate(conds["agent"]["series"], 1):
            if pt.get("agent_due"):
                ax.axvline(j, color="#2a9d5c", alpha=0.18, linewidth=1.2, zorder=0)
        ax.axhline(10, color="#c0c0c0", linestyle=":", linewidth=1)
        ax.set_ylim(0, 120)
        ax.set_xlim(1, n)
        n_adj = conds["agent"]["n_agent_adjustments"]
        ax.set_title(
            f"{key}  — {n_adj} agent adjustment{'s' if n_adj != 1 else ''}",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.set_xlabel("mutation attempts observed")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_agent_activity_png(rows: dict[str, dict[str, dict]], out: Path) -> None:
    """When the agent is actually awake, old schedule vs new trigger.

    The old schedule is a post_step_hook throttled to every 5 ACCEPTED
    mutants, so its cadence is hostage to the task's accept rate. The new
    trigger fires when the live estimate breaches the interval its own
    previous estimate published.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(rows)
    fig, ax = plt.subplots(figsize=(12, 0.85 * len(keys) + 2))
    for i, k in enumerate(keys):
        n = rows[k]["agent"]["n_points"]
        old_all = rows[k]["diag"]["agent_calls"]
        old_eff = set(rows[k]["diag"]["agent_effective"])
        new = [j for j, p in enumerate(rows[k]["agent"]["series"], 1) if p["agent_due"]]
        y = len(keys) - i
        ax.hlines(y + 0.16, 1, n, color="#e2e4e8", linewidth=7, zorder=1)
        ax.hlines(y - 0.16, 1, n, color="#e2e4e8", linewidth=7, zorder=1)
        ax.scatter(
            old_all,
            [y + 0.16] * len(old_all),
            s=42,
            zorder=3,
            color=["#c0504d" if a in old_eff else "#8b96a3" for a in old_all],
            marker="|",
            linewidths=2.4,
        )
        ax.scatter(
            new,
            [y - 0.16] * len(new),
            s=42,
            color="#2a9d5c",
            marker="|",
            linewidths=2.4,
            zorder=3,
        )
        ax.text(
            -2,
            y + 0.16,
            "every 5 accepted",
            ha="right",
            va="center",
            fontsize=8,
            color="#666",
        )
        ax.text(
            -2,
            y - 0.16,
            "CI breach",
            ha="right",
            va="center",
            fontsize=8,
            color="#2a6b45",
            fontweight="bold",
        )
        ax.text(-2, y + 0.52, k, ha="right", va="center", fontsize=9, fontweight="bold")
        ax.text(
            n + 2, y + 0.16, f"{len(old_all)}", va="center", fontsize=8, color="#666"
        )
        ax.text(
            n + 2,
            y - 0.16,
            f"{len(new)}",
            va="center",
            fontsize=8,
            color="#2a6b45",
            fontweight="bold",
        )

    ax.set_xlim(-46, 118)
    ax.set_ylim(0.3, len(keys) + 1.0)
    ax.set_yticks([])
    ax.set_xlabel("mutation attempt")
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    handles = [
        plt.Line2D(
            [],
            [],
            color="#8b96a3",
            marker="|",
            linestyle="",
            markersize=9,
            markeredgewidth=2.4,
            label="woken, no change made",
        ),
        plt.Line2D(
            [],
            [],
            color="#c0504d",
            marker="|",
            linestyle="",
            markersize=9,
            markeredgewidth=2.4,
            label="woken, multiplier moved",
        ),
        plt.Line2D(
            [],
            [],
            color="#2a9d5c",
            marker="|",
            linestyle="",
            markersize=9,
            markeredgewidth=2.4,
            label="new trigger would wake it",
        ),
    ]
    ax.legend(
        handles=handles,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    ax.set_title(
        "When the agent is awake: accepted-mutant tick vs surprise trigger",
        fontsize=12,
        fontweight="bold",
        pad=42,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_concurrency_png(rows: dict[str, dict[str, dict]], out: Path) -> None:
    """The root cause of the duration error, in one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = list(rows)
    conc = [rows[k]["diag"]["achieved_concurrency"] for k in keys]
    maxif = [rows[k]["diag"]["max_in_flight"] for k in keys]
    # error Little's Law @ max_in_flight implies, vs @ measured concurrency
    err_maxif = [
        (
            rows[k]["diag"]["llm_service_s"] / rows[k]["diag"]["max_in_flight"]
            - rows[k]["diag"]["actual_duration"]
        )
        / rows[k]["diag"]["actual_duration"]
        * 100
        for k in keys
    ]
    y = np.arange(len(keys))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.55 * len(keys) + 3))

    ax1.barh(
        y, conc, color=["#b03030" if c < m else "#2a6fb0" for c, m in zip(conc, maxif)]
    )
    ax1.axvline(
        8,
        color="#333",
        linestyle="--",
        linewidth=1.5,
        label="max_in_flight = 8 (the divisor used)",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(keys, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("LLM calls actually running in parallel")
    ax1.set_title(
        "Measured concurrency vs the hardcoded divisor", fontsize=11, fontweight="bold"
    )
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3, axis="x")
    for i, c in enumerate(conc):
        ax1.text(c + 0.15, i, f"{c:.1f}", va="center", fontsize=8)

    ax2.barh(
        y, err_maxif, color=["#b03030" if abs(e) > 15 else "#c9a227" for e in err_maxif]
    )
    ax2.axvline(0, color="#333", linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("duration error implied by dividing by 8, %")
    ax2.set_title(
        "...which reproduces the whole observed error spread",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(alpha=0.3, axis="x")
    for i, e in enumerate(err_maxif):
        ax2.text(
            e + (2 if e >= 0 else -2),
            i,
            f"{e:+.0f}%",
            va="center",
            ha="left" if e >= 0 else "right",
            fontsize=8,
        )

    fig.suptitle(
        "Root cause of the duration error: Little's Law divided by the wrong number",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_coverage_png(rows: dict[str, dict[str, dict]], out: Path) -> None:
    """Residual token error after the fix == exactly the missing event coverage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(rows)
    missing = [100 - rows[k]["diag"]["coverage_pct"] for k in keys]
    resid = [-rows[k]["auto"]["tok_err_pct"] for k in keys]

    fig, ax = plt.subplots(figsize=(7.2, 6))
    lim = max(max(missing), max(resid)) * 1.25 + 1
    ax.plot(
        [0, lim],
        [0, lim],
        color="#999",
        linestyle="--",
        linewidth=1.2,
        label="y = x (error explained entirely by missing events)",
    )
    ax.scatter(missing, resid, s=70, color="#2a6fb0", zorder=3)
    # Runs with full coverage all land on (0, 0) — stack their labels.
    seen: dict[tuple[int, int], int] = {}
    for k, x, yv in zip(keys, missing, resid):
        bucket = (round(x), round(yv))
        dy = -3 - 11 * seen.get(bucket, 0)
        seen[bucket] = seen.get(bucket, 0) + 1
        ax.annotate(k, (x, yv), textcoords="offset points", xytext=(8, dy), fontsize=8)
    ax.set_xlim(-1, lim)
    ax.set_ylim(-1, lim)
    ax.set_xlabel("tokens missing from the LLM_CALL event stream, %")
    ax.set_ylabel("token underestimate remaining after the math fix, %")
    ax.set_title(
        "After anchoring, the only token error left\nis the telemetry the estimator never saw",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--pred-at",
        type=int,
        default=10,
        help="table samples the prediction after this many mutation attempts (default 10)",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="log stems to drop from the figures (e.g. stalled runs)",
    )
    args = ap.parse_args()
    logger.remove()

    rows: dict[str, dict[str, dict]] = {}
    for path in args.logs:
        if path.stem in args.exclude:
            print(f"excluded: {path.stem}")
            continue
        diag = parse_before(path)
        auto = replay_log(path)
        agent = replay_log(path, apply_agent=True)
        if not diag or not auto or not agent:
            print(f"skipped (no usable data): {path.name}")
            continue
        rows[path.stem] = {"diag": diag, "auto": auto, "agent": agent}
        n_att = agent["n_points"]
        _, tb, _, db = at_mutation(auto, n_att, args.pred_at)
        _, ta, _, da = at_mutation(agent, n_att, args.pred_at)
        print(
            f"{path.stem:34s} @{args.pred_at:<3d} "
            f"tok {tb:+7.1f} -> {ta:+7.1f} | dur {db:+7.1f} -> {da:+7.1f} "
            f"| {agent['n_agent_adjustments']} agent adjustments"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison_table.tex").write_text(
        build_table_tex(rows, args.pred_at), encoding="utf-8"
    )
    build_table_png(rows, args.out_dir / "comparison_table.png", args.pred_at)
    build_error_decay_png(
        rows,
        args.out_dir / "error_decay_duration.png",
        "predicted_duration_s",
        "|duration pred error|, %",
        "Duration prediction error as the run proceeds: auto vs auto + agent",
    )
    build_error_decay_png(
        rows,
        args.out_dir / "error_decay_tokens.png",
        "predicted_tokens",
        "|token pred error|, %",
        "Token prediction error as the run proceeds: auto vs auto + agent",
    )
    build_agent_activity_png(rows, args.out_dir / "agent_activity.png")
    build_concurrency_png(rows, args.out_dir / "concurrency_diagnosis.png")
    build_coverage_png(rows, args.out_dir / "token_coverage.png")
    print(f"\nreport written to {args.out_dir}")


if __name__ == "__main__":
    main()
