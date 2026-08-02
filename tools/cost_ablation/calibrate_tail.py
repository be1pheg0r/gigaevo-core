#!/usr/bin/env python3
"""Calibrate the remaining-work multiplier from replayed logs.

    m* = (actual_duration - elapsed) / (predicted_duration - elapsed)

Reports the median multiplier and its cross-run spread at each checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics as st
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from replay_from_log import replay_log  # noqa: E402

CHECKPOINTS = (0.10, 0.25, 0.50)


def ideal_multipliers(res: dict) -> dict[float, float]:
    """The scale the remaining term needed, per checkpoint."""
    out: dict[float, float] = {}
    series, actual = res["series"], res["actual_duration"]
    for p in CHECKPOINTS:
        i = min(len(series) - 1, max(0, round(p * len(series)) - 1))
        pt = series[i]
        elapsed = pt.get("elapsed_s") or 0.0
        pred_rest = pt["predicted_duration_s"] - elapsed
        real_rest = actual - elapsed
        # Too close to the end for the remaining term to mean anything.
        if pred_rest <= 1e-6 or real_rest <= 1e-6:
            continue
        out[p] = real_rest / pred_rest
    return out


def summarise(name: str, ms: list[float]) -> str:
    ms = sorted(ms)
    q1, q3 = ms[len(ms) // 4], ms[(3 * len(ms)) // 4]
    return (
        f"{name:>7} n={len(ms):<3} медиана {st.median(ms):5.2f}  "
        f"IQR [{q1:.2f}, {q3:.2f}]  разброс {ms[0]:.2f}..{ms[-1]:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+", type=Path)
    args = ap.parse_args()
    logger.remove()

    per_cp: dict[float, list[float]] = {p: [] for p in CHECKPOINTS}
    per_log: list[tuple[str, dict[float, float]]] = []
    for path in args.logs:
        try:
            res = replay_log(path)
        except Exception as exc:  # noqa: BLE001 - one bad log must not stop the sweep
            print(f"пропуск {path.name}: {exc}", flush=True)
            continue
        if not res:
            continue
        ms = ideal_multipliers(res)
        if not ms:
            continue
        per_log.append((path.stem, ms))
        for p, m in ms.items():
            per_cp[p].append(m)

    print(f"\nлогов разобрано: {len(per_log)}\n")
    print("=== множитель, который сделал бы оценку точной ===")
    for p in CHECKPOINTS:
        if per_cp[p]:
            print(summarise(f"{p:.0%}", per_cp[p]))

    print("\n=== что снимает константа, а что нет ===")
    print(
        f"{'прогресс':>9} {'|err| как есть':>15} {'|err| после константы':>22} {'остаток':>10}"
    )
    for p in CHECKPOINTS:
        ms = per_cp[p]
        if len(ms) < 4:
            continue
        k = st.median(ms)
        raw = [abs(1 - m) * 100 for m in ms]  # error with no correction
        left = [abs(1 - m / k) * 100 for m in ms]  # error after the constant
        print(
            f"{p:>9.0%} {st.median(raw):>14.1f}% {st.median(left):>21.1f}% "
            f"{st.median(left) / max(st.median(raw), 1e-9):>9.0%}"
        )

    print("\n=== переносится ли константа на другие логи (split-half) ===")
    for p in CHECKPOINTS:
        rows = [(n, m[p]) for n, m in per_log if p in m]
        if len(rows) < 8:
            continue
        a = [m for i, (_, m) in enumerate(rows) if i % 2 == 0]
        b = [m for i, (_, m) in enumerate(rows) if i % 2 == 1]
        ka, kb = st.median(a), st.median(b)
        cross = st.median([abs(1 - m / ka) * 100 for m in b])
        own = st.median([abs(1 - m / kb) * 100 for m in b])
        print(
            f"{p:>9.0%} константа A={ka:.2f} B={kb:.2f} | "
            f"на чужой половине {cross:.1f}%, на своей {own:.1f}%"
        )


if __name__ == "__main__":
    main()
