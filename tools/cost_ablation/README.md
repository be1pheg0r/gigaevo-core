# cost_ablation

Runs the same gigaevo problem(s) twice — once with the LLM-driven
CostMonitorAgent (`+cost_monitor=enabled`), once with a no-op stand-in
(`+cost_monitor=agentless`) — and builds a comparison table + per-task
prediction-error-over-time plots. Built 2026-07-31 to replace the ad-hoc
manual process (launch via ssh, hand-parse logs, hand-build plots) used in
that session's report.

## Usage

```bash
python3 tools/cost_ablation/run_ablation.py     --tasks algotune/algotune_lqr algotune/algotune_markowitz             adversarial/code/pop_a toy_kadane     --max-mutants 100 --llm summer_school_servers     --out-dir experiments/cost_ablation_$(date +%Y%m%d_%H%M%S)
```

This launches 2×N `nohup python3 run.py ...` runs on auto-picked empty
redis dbs (0-127), polls until every log has a `Duration:` line, then calls
`build_report.py` automatically. Outputs land in `<out-dir>/`:

- `manifest.json` — task/condition/db/log/pid for every launched run
- `<key>_noagent.log`, `<key>_withagent.log` — raw run logs
- `report/comparison_table.tex` — LaTeX source (booktabs), no compiler needed to view it
- `report/comparison_table.png` — matplotlib rendering of the same table (always produced)
- `report/error_decay_plots.png` — one subplot per task, |predicted − actual| duration error % vs run progress, one line per condition

To just rebuild the report from an existing manifest (e.g. after fixing a
crashed task and rerunning it manually):

```bash
python3 tools/cost_ablation/build_report.py --manifest experiments/<out-dir>/manifest.json
```

To sanity-check the log parser without launching anything:

```bash
python3 tools/cost_ablation/build_report.py --selftest
```

## Checking the prediction math against old logs — no evolution run needed

`replay_from_log.py` feeds a log's own `[LLM_CALL]`/`[STAGE_EXEC]`/
`[MUTATION_ATTEMPTED]` lines through a fresh `CostMonitorHook` running the
**currently installed** code, and compares the result against that log's
own actual tokens/duration. Use this to check whether a change to
`growth_estimator.py` / `cost_monitor_hook.py` would have predicted better
on data you already have, without spending any LLM budget or wall-clock
time re-running evolution:

```bash
python3 tools/cost_ablation/replay_from_log.py experiments/*/*.log
python3 tools/cost_ablation/replay_from_log.py --out-dir experiments/replay_check path/to/one.log
```

Scope: this only replays the deterministic part of the pipeline (static
baseline + growth-law fit + non-LLM stage duration). It does **not** replay
`CostMonitorAgent`'s LLM-driven calibration cycle (golden/growth overrides)
— that needs a real LLM call each time and can't be reproduced from a log;
use `run_ablation.py` for a genuine with-agent comparison.

## Known gotchas (carried over from the 2026-07-31 session)

- Redis db must be 0-127 (this server has 128 logical dbs) — db≥128 crashes
  hydra's `redis_storage` instantiation with an `InstantiationException`.
- A non-empty redis db makes a run finish suspiciously fast (~5 min) without
  doing real work — `run_ablation.py` already checks `keys('*') == []`
  before picking a db, but if you launch manually, check this yourself.
- `stall_watchdog` stops a run early if no mutant is accepted for 300s —
  a task with a genuinely low/zero accept rate (e.g. a seed program already
  near-optimal on its own test suite) will show fewer `[CostMonitorHookJSON]`
  points than `max_mutants` would suggest. Not a bug, just fewer data points.
- `summarize()` in `build_report.py` skips any log with no CostMonitorHookJSON
  lines, no `Duration:` line, or zero actual tokens (crashed/incomplete run) —
  check the driver's stdout for "skipping ... no usable data" if a task is
  missing from the report.
