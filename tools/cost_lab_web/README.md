# Cost Lab

Web console for the cost-prediction ablation: pick tasks, launch the runs,
watch them, then read whether `CostMonitorAgent` beat the automatic
estimator — at 5%, 10%, 15% … of run progress, not just at the end.

## Run it on the summer-school server

```bash
ssh User10@82.202.156.206
cd ~/gigaevo-core
setsid nohup python3 tools/cost_lab_web/app.py > ~/cost_lab_web.log 2>&1 < /dev/null &
```

Port 8091 by default (`COST_LAB_PORT` to change it).

## Reach it from your laptop

The server's firewall only lets 80/443/8080 through, so tunnel it:

```bash
ssh -N -L 8091:127.0.0.1:8091 User10@82.202.156.206
```

Then open <http://127.0.0.1:8091/>. Leave that command running; it is the
whole connection.

## The three views

**Launch** — every runnable problem in `problems/`, grouped by family
(alphaevolve first). Pick tasks, set attempts per run and tasks per wave,
press Start. Each task is launched twice, `+cost_monitor=agentless` and
`+cost_monitor=enabled`, on auto-picked empty redis dbs. It shells out to
`tools/cost_ablation/run_ablation.py` — same launcher as the CLI.

**Live** — one card per run. The mark is the prediction interval drawn as a
ribbon, with the measured total as a dashed line through it and amber
notches where the agent woke. When the ribbon stops straddling the dashed
line, the forecast has gone wrong and you can see it without reading a
number.

**Analyze** — the comparison:

- error vs progress, median across paired tasks with an IQR band
- checkpoint ledger at 5/10/15/25/50/75/100%, with the agent−estimator delta
- **what the agent actually did**: every wakeup on the run's own attempt
  axis, filled if it moved a lever, hollow if it looked and declined. Hover
  gives the trigger sentence and the agent's reasoning.
- per-task small multiples, and the final forecast-vs-measured table

Toggle duration/tokens at the top right. "Build PNG report" runs
`build_report.py` for the same experiment and writes the static artifacts
next to the logs.

## Keeping the code fresh

The header shows branch, HEAD and how far behind `origin` it is. **Sync
code** runs `git pull --ff-only`. Python changes need the app restarted;
frontend changes are picked up on reload.

## Notes

- `--selftest` checks the one thing parsed here and nowhere else (the
  agent's adjustment log line): `python3 tools/cost_lab_web/app.py --selftest`
- Runs are started with `start_new_session=True`, so they survive an app
  restart; a run whose log has not grown in 3 minutes without a `Duration:`
  line is reported as stopped.
- Nothing is cached: every poll re-reads the logs. That is fine at this
  scale (tens of runs) and keeps the console honest about what is on disk.
