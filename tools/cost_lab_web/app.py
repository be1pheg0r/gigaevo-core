#!/usr/bin/env python3
"""Web console for launching and analysing cost-prediction experiments."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "cost_ablation"))

from build_report import parse_log  # noqa: E402  (reuse, don't reimplement)
from replay_from_log import hook_from_log  # noqa: E402
from run_ablation import find_free_dbs, launch  # noqa: E402  (one launch path)

EXPERIMENTS = REPO / "experiments"
STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.environ.get("COST_LAB_PORT", "8091"))

# Maximum log inactivity before an unfinished run is considered stopped.
STALE_AFTER_S = 180

ADJ_RE = re.compile(
    r"\[CostMonitorHook\] adjustments: cold=(\S+) golden=(\S+) growth=(\S+) conc=(\S+) "
    r"outliers=(\d+) trigger='([^']*)' reason=(.*)"
)
TRACE_RE = re.compile(r"\[CostMonitorAgentTrace\] (\{.*\})")
ATT_RE = re.compile(r"\[MUTATION_ATTEMPTED\]")
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+")
CALL_RE = re.compile(r"\[LLM_CALL\] (\{.*\})")
# Fitness records provide search progress independently of the cost model.
FIT_RE = re.compile(r"\[FetchMetrics\] \w+ metrics=(\{.*\})")

# Observer calls are reported separately from the monitored workload.
OBSERVER_STAGE = "CostMonitorAgent"

# Imports every problem gets for free — never reported as a missing dependency.
_FIRST_PARTY = {"problems", "gigaevo", "config", "tools"}
_dep_cache: dict[str, tuple[float, list[str]]] = {}


def missing_deps(problem_dir: Path) -> list[str]:
    """Return unavailable third-party modules imported by a problem."""
    key = problem_dir.as_posix()
    try:
        stamp = max(p.stat().st_mtime for p in problem_dir.rglob("*.py"))
    except ValueError:
        return []
    hit = _dep_cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]

    mods: set[str] = set()
    for py in problem_dir.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])

    # Local problem modules are added to sys.path by the executor.
    siblings = {p.stem for p in problem_dir.rglob("*.py")}

    missing = []
    for m in sorted(mods):
        if m in _FIRST_PARTY or m in siblings or m in sys.stdlib_module_names:
            continue
        try:
            if importlib.util.find_spec(m) is None:
                missing.append(m)
        except (ImportError, ValueError):
            missing.append(m)
    _dep_cache[key] = (stamp, missing)
    return missing


# --------------------------------------------------------------- repo state


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def repo_state() -> dict:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    counts = _git("rev-list", "--left-right", "--count", f"origin/{branch}...HEAD")
    behind, ahead = (counts.split() + ["0", "0"])[:2] if counts else ("0", "0")
    return {
        "branch": branch,
        "head": _git("rev-parse", "--short", "HEAD"),
        "subject": _git("log", "-1", "--format=%s"),
        "authored": _git("log", "-1", "--format=%ar"),
        "behind": int(behind),
        "ahead": int(ahead),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


# ------------------------------------------------------------------ problems


def list_problems() -> list[dict]:
    """Every runnable problem, as `problem.name` values grouped by family."""
    out = []
    for metrics in sorted((REPO / "problems").rglob("metrics.yaml")):
        name = metrics.parent.relative_to(REPO / "problems").as_posix()
        parts = name.split("/")
        out.append(
            {
                "name": name,
                "family": parts[0] if len(parts) > 1 else "standalone",
                "label": "/".join(parts[1:]) if len(parts) > 1 else name,
                "seeds": len(list((metrics.parent / "initial_programs").glob("*.py"))),
                "missing_deps": missing_deps(metrics.parent),
            }
        )
    return out


# --------------------------------------------------------------- experiments


def parse_agent_events(
    path: Path,
) -> tuple[list[dict], int, dict[str, dict], list[float]]:
    """The agent's own wakeup record: attempt, trigger, levers, reasoning.

    Returns the events, the total `[MUTATION_ATTEMPTED]` count (the fallback
    progress figure for logs written before the prediction line carried an
    ``attempts`` field), per-stage LLM-call totals, and every program's
    fitness in evaluation order — all in ONE pass, so adding the stage
    breakdown and the fitness series did not add more reads of a 10k-line log.

    A lever value of -1 means 'agent declined to move this one', so an event
    where all four are -1 is a wakeup that changed nothing — worth showing,
    because a monitor that wakes constantly and never acts is its own finding.
    """
    traces: list[dict] = []  # new format: decision + the evidence behind it
    legacy: list[dict] = []  # pre-trace logs: the decision alone
    attempts = 0
    stages: dict[str, dict] = {}
    fitness: list[float] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if ATT_RE.search(line):
                attempts += 1
                continue
            m = FIT_RE.search(line)
            if m:
                try:
                    # The metrics dict is logged as a Python repr with string
                    # values, not JSON — literal_eval, then coerce.
                    val = ast.literal_eval(m.group(1)).get("fitness")
                    fitness.append(float(val))
                except (ValueError, SyntaxError, TypeError):
                    pass
                continue
            m = CALL_RE.search(line)
            if m:
                try:
                    c = json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
                s = stages.setdefault(
                    c.get("stage") or "?",
                    {"calls": 0, "latency_ms": 0.0, "tokens": 0, "max_latency_ms": 0.0},
                )
                lat = float(c.get("latency_ms") or 0.0)
                s["calls"] += 1
                s["latency_ms"] += lat
                s["max_latency_ms"] = max(s["max_latency_ms"], lat)
                s["tokens"] += int(c.get("tokens_in") or 0) + int(
                    c.get("tokens_out") or 0
                )
                continue
            m = TRACE_RE.search(line)
            if m:
                try:
                    t = json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
                levers = {
                    k: (None if v is None or v < 0 else v)
                    for k, v in (t.get("levers") or {}).items()
                }
                ts = TS_RE.match(line)
                traces.append(
                    {
                        "attempt": t.get("attempt", attempts),
                        "levers": levers,
                        "levers_before": t.get("levers_before") or {},
                        "moved": [k for k, v in levers.items() if v is not None],
                        "actions": t.get("actions", []),
                        "outliers": len(t.get("flag_outlier_indices") or []),
                        "outlier_indices": t.get("flag_outlier_indices") or [],
                        "skip_calibration": t.get("skip_calibration", False),
                        "trigger": t.get("trigger", ""),
                        "reason": (t.get("reasoning") or "").strip(),
                        "evidence": t.get("evidence") or {},
                        "ts": ts.group(1) if ts else None,
                        "line": lineno,
                    }
                )
                continue
            m = ADJ_RE.search(line)
            if not m:
                continue
            cold, golden, growth, conc, outliers, trigger, reason = m.groups()

            def num(v: str) -> float | None:
                try:
                    f_ = float(v)
                except ValueError:
                    return None
                return None if f_ < 0 else f_

            levers = {
                "cold_start": num(cold),
                "golden_ratio": num(golden),
                "growth_rate": num(growth),
                "concurrency": num(conc),
            }
            ts = TS_RE.match(line)
            legacy.append(
                {
                    "attempt": attempts,
                    "levers": levers,
                    "moved": [k for k, v in levers.items() if v is not None],
                    "actions": [],
                    "outliers": int(outliers),
                    "outlier_indices": [],
                    "skip_calibration": False,
                    "trigger": trigger,
                    "reason": reason.strip(),
                    "evidence": {},
                    "ts": ts.group(1) if ts else None,
                    "line": lineno,
                }
            )
    # Both lines are written for the same wakeup, so a log that has traces
    # must be read only through them — the legacy line would double-count.
    return (traces or legacy), attempts, stages, fitness


def run_status(log_path: Path, has_duration: bool) -> str:
    if has_duration:
        return "done"
    if not log_path.exists():
        return "pending"
    if time.time() - log_path.stat().st_mtime > STALE_AFTER_S:
        return "stopped"
    return "running"


_fitness_spec_cache: dict[str, dict] = {}


def fitness_spec(task: str) -> dict:
    """Which way is better, and what counts as done — from the problem's own
    ``metrics.yaml``.

    Without the direction a fitness chart is unreadable: `packing_circles`
    maximises the sum of radii while `minimize_max_min_dist_ratio` minimises a
    ratio, and "best so far" means the opposite thing in each.
    """
    hit = _fitness_spec_cache.get(task)
    if hit is not None:
        return hit
    spec = {
        "higher_is_better": True,
        "target": None,
        "label": "fitness",
        "sentinel": None,
    }
    path = REPO / "problems" / task / "metrics.yaml"
    try:
        import yaml

        f = (
            (yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            .get("specs", {})
            .get("fitness", {})
        )
        spec["higher_is_better"] = bool(f.get("higher_is_better", True))
        # The bound the problem was written against is the paper's target; on a
        # minimised metric it is the lower bound instead.
        spec["target"] = (
            f.get("upper_bound") if spec["higher_is_better"] else f.get("lower_bound")
        )
        spec["label"] = f.get("description") or "fitness"
        # "this program failed", not a fitness: -1000 on a maximised metric,
        # +1000 on a minimised one. Left in the series but kept out of the
        # chart's scale, where one of them flattens everything else.
        spec["sentinel"] = f.get("sentinel_value")
    except (OSError, ImportError, AttributeError, TypeError, ValueError):
        pass
    _fitness_spec_cache[task] = spec
    return spec


def summarize_run(entry: dict) -> dict:
    log_path = REPO / entry["log"]
    series, duration, actual_tokens = ([], None, 0)
    events: list[dict] = []
    attempts = 0
    stages: dict[str, dict] = {}
    fitness: list[float] = []
    if log_path.exists():
        series, duration, actual_tokens = parse_log(log_path)
        events, attempts, stages, fitness = parse_agent_events(log_path)
    return {
        **{k: entry[k] for k in ("task", "key", "condition", "db")},
        "log": entry["log"],
        "stem": Path(entry["log"]).stem,
        "status": run_status(log_path, duration is not None),
        "series": series,
        "actual_duration": duration,
        "actual_tokens": actual_tokens or None,
        "agent_events": events,
        "stages": stages,
        "fitness": fitness,
        "fitness_spec": fitness_spec(entry["task"]),
        "attempts": (series[-1].get("attempts") if series else 0) or attempts,
        "updated": log_path.stat().st_mtime if log_path.exists() else None,
    }


def driver_state(name: str) -> dict:
    """Is the launcher alive, and what has it said so far.

    The driver spends its first minute scanning 128 redis dbs and staggering
    launches, so without this the console had nothing to show between the
    click and the first run's log appearing — which read as "nothing
    happened". A non-zero ``returncode`` is a launch that died; its tail is
    the only place the reason exists.
    """
    log = EXPERIMENTS / f"{name}.driver.log"
    launch = _launches.get(name)
    rc = launch.proc.poll() if launch else None
    tail = _peek(log)[-6000:] if log.is_file() else ""
    return {
        "known": bool(launch) or log.is_file(),
        "alive": bool(launch) and rc is None,
        "returncode": rc,
        "failed": rc is not None and rc != 0 and not launch.stopped,
        "stopped": bool(launch and launch.stopped),
        "log_lines": tail.splitlines()[-120:],
        "updated": log.stat().st_mtime if log.is_file() else None,
    }


def experiment_dirs() -> list[Path]:
    if not EXPERIMENTS.exists():
        return []
    return sorted(
        (
            d
            for d in EXPERIMENTS.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def load_manifest(d: Path) -> list[dict]:
    try:
        return json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


@dataclass
class Launch:
    name: str
    proc: subprocess.Popen
    cmd: list[str]
    stopped: bool = False  # killed on request, so a non-zero exit isn't a crash


_launches: dict[str, Launch] = {}


class NewExperiment(BaseModel):
    tasks: list[str] = Field(min_length=1)
    max_mutants: int = 100
    wave_size: int = 3
    llm: str = "summer_school_servers"
    label: str = ""


app = FastAPI(title="gigaevo cost lab")


@app.get("/api/repo")
async def api_repo() -> dict:
    return repo_state()


@app.post("/api/repo/pull")
async def api_pull() -> dict:
    branch = repo_state()["branch"]
    r = subprocess.run(
        ["git", "pull", "--ff-only", "origin", branch],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return {
        "ok": r.returncode == 0,
        "output": (r.stdout + r.stderr).strip()[-4000:],
        "repo": repo_state(),
    }


@app.get("/api/problems")
async def api_problems() -> list[dict]:
    return list_problems()


@app.get("/api/experiments")
async def api_experiments() -> list[dict]:
    out = []
    seen = set()
    for d in experiment_dirs():
        manifest = load_manifest(d)
        statuses = [
            run_status(REPO / e["log"], "Duration:" in _peek(REPO / e["log"]))
            for e in manifest
        ]
        drv = driver_state(d.name)
        seen.add(d.name)
        out.append(
            {
                "name": d.name,
                "created": d.stat().st_mtime,
                "n_runs": len(manifest),
                "tasks": sorted({e["task"] for e in manifest}),
                "running": statuses.count("running"),
                "done": statuses.count("done"),
                "launching": drv["alive"],
                "failed": drv["failed"],
            }
        )
    # A launch that died before writing its manifest has only a driver log —
    # list it anyway, otherwise a failed start is indistinguishable from a
    # button that did nothing.
    for name in _launches:
        if name in seen:
            continue
        drv = driver_state(name)
        out.append(
            {
                "name": name,
                "created": time.time(),
                "n_runs": 0,
                "tasks": [],
                "running": 0,
                "done": 0,
                "launching": drv["alive"],
                "failed": drv["failed"],
            }
        )
    return sorted(out, key=lambda e: e["created"], reverse=True)


# Tail window used for inexpensive run-status polling.
_PEEK_BYTES = 256 * 1024


def _peek(path: Path) -> str:
    """Cheap 'did this run finish' check — read the tail, not the whole log."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - _PEEK_BYTES))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


@app.get("/api/experiments/{name}")
async def api_experiment(name: str) -> dict:
    d = EXPERIMENTS / name
    drv = driver_state(name)
    # No manifest yet is normal for the first seconds of a launch — answer with
    # the driver's own output rather than 404, so the console can show progress
    # instead of an error while the launcher is still scanning redis.
    if not (d / "manifest.json").exists() and not drv["known"]:
        raise HTTPException(404, "unknown experiment")
    manifest = load_manifest(d)
    runs = [summarize_run(e) for e in manifest]
    report = (
        sorted(p.name for p in (d / "report").glob("*"))
        if (d / "report").exists()
        else []
    )
    return {
        "name": name,
        "created": d.stat().st_mtime if d.exists() else time.time(),
        "runs": runs,
        "report": report,
        "driver": drv,
        "driver_alive": drv["alive"],
    }


@app.post("/api/experiments")
async def api_launch(body: NewExperiment) -> dict:
    known = {p["name"] for p in list_problems()}
    unknown = [t for t in body.tasks if t not in known]
    if unknown:
        raise HTTPException(400, f"unknown problems: {', '.join(unknown)}")

    slug = re.sub(r"[^a-z0-9]+", "-", body.label.lower()).strip("-") or "ablation"
    name = f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        sys.executable,
        "-u",
        "tools/cost_ablation/run_ablation.py",
        "--tasks",
        *body.tasks,
        "--max-mutants",
        str(body.max_mutants),
        "--wave-size",
        str(body.wave_size),
        "--llm",
        body.llm,
        "--skip-report",
        "--out-dir",
        f"experiments/{name}",
    ]
    log = EXPERIMENTS / f"{name}.driver.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )  # outlives an app restart
    _launches[name] = Launch(name=name, proc=proc, cmd=cmd)
    return {"name": name, "cmd": " ".join(cmd)}


# Number of real attempts used to initialise a budget projection.
PROBE_ATTEMPTS = 10


class BudgetProbe(BaseModel):
    task: str
    attempts: int = Field(default=100, ge=1, le=100_000)
    budget_tokens: int = Field(default=1_000_000, ge=1)
    llm: str = "summer_school_servers"


@dataclass
class Probe:
    name: str
    task: str
    attempts: int
    budget_tokens: int
    proc: subprocess.Popen
    log: Path


_probes: dict[str, Probe] = {}


_probe_cache: dict[str, tuple[tuple, dict | None]] = {}


def probe_answer(p: Probe) -> dict | None:
    """Project cost and affordable attempts from a probe log."""
    if not p.log.exists():
        return None
    # The question is part of the key, not just the log: two probes can share
    # a log and ask about different horizons.
    stamp = (p.log.stat().st_size, p.log.stat().st_mtime, p.attempts, p.budget_tokens)
    hit = _probe_cache.get(p.name)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        hook = hook_from_log(p.log)
    except (OSError, ValueError):
        return None
    if hook is None or not hook._tokens_by_stage:
        return None
    point, (lo, hi) = hook.project_tokens(p.attempts)
    if point <= 0:
        return None
    fits = hi <= p.budget_tokens
    # Use the upper quartile for the conservative end of the planning range.
    q75 = hook.project_tokens(p.attempts, alpha=0.5)[1][1]
    n_hi = hook.affordable_attempts(p.budget_tokens)
    n_lo = hook.affordable_attempts(p.budget_tokens / max(q75 / point, 1.0))
    answer = {
        "observed_attempts": hook._attempts,
        "target_attempts": p.attempts,
        "budget_tokens": p.budget_tokens,
        "predicted_tokens": point,
        "ci": [lo, hi],
        "verdict": "fits"
        if fits
        else ("tight" if point <= p.budget_tokens else "over"),
        "affordable": [min(n_lo, n_hi), max(n_lo, n_hi)],
    }
    _probe_cache[p.name] = (stamp, answer)
    return answer


@app.post("/api/budget")
async def api_budget_start(body: BudgetProbe) -> dict:
    if body.task not in {p["name"] for p in list_problems()}:
        raise HTTPException(400, f"unknown problem: {body.task}")
    missing = missing_deps(REPO / "problems" / body.task)
    if missing:
        raise HTTPException(
            400, f"problem needs modules this box lacks: {', '.join(missing)}"
        )
    try:
        db = find_free_dbs(1)[0]
    except (RuntimeError, OSError) as exc:
        raise HTTPException(503, f"no free redis db: {exc}") from exc

    name = f"probe_{re.sub(r'[^a-z0-9]+', '-', body.task.lower()).strip('-')}_{time.strftime('%Y%m%d_%H%M%S')}"
    log = EXPERIMENTS / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Probe the underlying pipeline without observer overhead.
    proc = launch(body.task, db, PROBE_ATTEMPTS, body.llm, "agentless", log)
    _probes[name] = Probe(
        name=name,
        task=body.task,
        attempts=body.attempts,
        budget_tokens=body.budget_tokens,
        proc=proc,
        log=log,
    )
    return {"name": name, "probe_attempts": PROBE_ATTEMPTS, "db": db}


@app.get("/api/budget/{name}")
async def api_budget_state(name: str) -> dict:
    p = _probes.get(name)
    if p is None:
        raise HTTPException(404, "unknown probe")
    rc = p.proc.poll()
    return {
        "name": name,
        "task": p.task,
        "probe_attempts": PROBE_ATTEMPTS,
        "alive": rc is None,
        "failed": rc is not None and rc != 0 and probe_answer(p) is None,
        "stem": p.log.stem,
        "log_lines": _peek(p.log)[-4000:].splitlines()[-40:] if p.log.exists() else [],
        "answer": probe_answer(p),
    }


@app.post("/api/budget/{name}/stop")
async def api_budget_stop(name: str) -> dict:
    p = _probes.get(name)
    if p is None:
        raise HTTPException(404, "unknown probe")
    if p.proc.poll() is None:
        p.proc.kill()
    return {"ok": True, "answer": probe_answer(p)}


@app.post("/api/experiments/{name}/stop")
async def api_stop(name: str) -> dict:
    launch = _launches.get(name)
    killed = 0
    if launch:
        launch.stopped = True
        if launch.proc.poll() is None:
            launch.proc.kill()
            killed += 1
    # The driver's children are separate run.py processes — stop them by out-dir.
    # Killing the driver already worked at this point, so a missing pkill must
    # not turn the whole request into a 500 and hide that.
    try:
        rc = subprocess.run(
            ["pkill", "-f", f"experiments/{name}"], capture_output=True
        ).returncode
    except OSError as exc:
        return {
            "ok": True,
            "driver_killed": killed,
            "pkill_rc": None,
            "warning": str(exc),
        }
    return {"ok": True, "driver_killed": killed, "pkill_rc": rc}


@app.post("/api/experiments/{name}/report")
async def api_report(name: str) -> dict:
    d = EXPERIMENTS / name
    if not (d / "manifest.json").exists():
        raise HTTPException(404, "unknown experiment")
    r = subprocess.run(
        [
            sys.executable,
            "tools/cost_ablation/build_report.py",
            "--manifest",
            str(d / "manifest.json"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    files = (
        sorted(p.name for p in (d / "report").glob("*"))
        if (d / "report").exists()
        else []
    )
    return {
        "ok": r.returncode == 0,
        "output": (r.stdout + r.stderr)[-4000:],
        "files": files,
    }


@app.get("/api/experiments/{name}/archive")
async def api_archive(name: str) -> StreamingResponse:
    """The whole experiment as one zip in the browser's download — logs,
    manifest and any built report. Nothing is written to disk on the way
    out; the zip is assembled in memory and streamed."""
    d = (EXPERIMENTS / name).resolve()
    if not (d / "manifest.json").exists():
        raise HTTPException(404, "unknown experiment")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(d.rglob("*")):
            if path.is_file():
                z.write(path, arcname=f"{name}/{path.relative_to(d).as_posix()}")
        driver = EXPERIMENTS / f"{name}.driver.log"
        if driver.is_file():
            z.write(driver, arcname=f"{name}/{driver.name}")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )


@app.get("/api/experiments/{name}/report/{filename}")
async def api_report_file(name: str, filename: str) -> FileResponse:
    path = (EXPERIMENTS / name / "report" / filename).resolve()
    if (
        not path.is_file()
        or (EXPERIMENTS / name / "report").resolve() not in path.parents
    ):
        raise HTTPException(404, "no such report file")
    return FileResponse(path)


@app.get("/api/experiments/{name}/log/{stem}")
async def api_log(
    name: str, stem: str, tail: int = 400, around: int = 0, ctx: int = 60, q: str = ""
) -> dict:
    """Log slice. `around` centres the window on a 1-indexed line (that's
    what an agent trace carries), `q` filters, otherwise it's the tail."""
    path = (EXPERIMENTS / name / f"{stem}.log").resolve()
    if not path.is_file() or (EXPERIMENTS / name).resolve() not in path.parents:
        raise HTTPException(404, "no such log")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    if q:
        hits = [(i, ln) for i, ln in enumerate(lines, 1) if q.lower() in ln.lower()]
        return {
            "stem": stem,
            "total": len(lines),
            "first_line": 0,
            "query": q,
            "matches": len(hits),
            "numbered": [{"n": i, "text": t} for i, t in hits[:tail]],
        }

    if around > 0:
        lo = max(0, around - ctx - 1)
        hi = min(len(lines), around + ctx)
    else:
        lo, hi = max(0, len(lines) - tail), len(lines)
    return {
        "stem": stem,
        "total": len(lines),
        "first_line": lo + 1,
        "focus": around,
        "numbered": [{"n": lo + i + 1, "text": t} for i, t in enumerate(lines[lo:hi])],
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _selftest() -> None:
    """Assert-based check of the one thing parsed here and nowhere else."""
    import tempfile

    log = (
        "2026-07-31 14:00:01.000 | INFO | x | [MUTATION_ATTEMPTED] {}\n"
        "2026-07-31 14:00:02.000 | INFO | x | [MUTATION_ATTEMPTED] {}\n"
        "2026-07-31 14:00:03.000 | INFO | gigaevo.monitoring.cost_monitor_hook:_run_agent:447 | "
        "[CostMonitorHook] adjustments: cold=-1 golden=-1 growth=-1 conc=-1 outliers=0 "
        "trigger='estimate jumped above its own interval at attempt 4: 1654s vs [436, 954]s' reason=\n"
        "2026-07-31 14:00:04.000 | INFO | x | [MUTATION_ATTEMPTED] {}\n"
        "2026-07-31 14:00:05.000 | INFO | gigaevo.monitoring.cost_monitor_hook:_run_agent:447 | "
        "[CostMonitorHook] adjustments: cold=0.8 golden=-1 growth=1.15 conc=6.2 outliers=2 "
        "trigger='tail cap hit' reason=latency outliers dominate the bucket\n"
        # the search's own progress: a python repr with string values, not JSON
        "2026-07-31 14:00:06.000 | INFO | x | [FetchMetrics] 632dfa63 "
        "metrics={'fitness': '1.2424', 'is_valid': '1.0000'}\n"
        "2026-07-31 14:00:07.000 | INFO | x | [FetchMetrics] 7ac01e11 "
        "metrics={'fitness': '1.8571', 'is_valid': '1.0000'}\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write(log)
        p = Path(f.name)
    try:
        ev, attempts, stages, fit = parse_agent_events(p)
        assert len(ev) == 2, ev
        assert attempts == 3, attempts  # fallback progress for pre-`attempts` logs
        assert all(e["evidence"] == {} for e in ev)  # old format carries no evidence
        # a wakeup that declined every lever is still recorded, with nothing moved
        assert ev[0]["attempt"] == 2 and ev[0]["moved"] == [], ev[0]
        assert ev[0]["trigger"].startswith("estimate jumped"), ev[0]
        # -1 means "declined", so it must not be reported as a value
        assert ev[0]["levers"]["cold_start"] is None
        assert ev[1]["attempt"] == 3, ev[1]
        assert set(ev[1]["moved"]) == {"cold_start", "growth_rate", "concurrency"}, ev[
            1
        ]
        assert ev[1]["levers"]["golden_ratio"] is None
        assert ev[1]["outliers"] == 2
        assert ev[1]["reason"] == "latency outliers dominate the bucket"
        # fitness comes off the same single pass, in evaluation order
        assert fit == [1.2424, 1.8571], fit
        # direction and target are read from the problem, not assumed
        assert (
            fitness_spec("alphaevolve/packing_circles/n_26")["higher_is_better"] is True
        )
        assert fitness_spec("alphaevolve/packing_circles/n_26")["target"] == 2.635
        assert fitness_spec("alphaevolve/packing_circles/n_26")["sentinel"] == -1000.0
        mm = fitness_spec("alphaevolve/minimize_max_min_dist_ratio/2_dimensions")
        assert mm["higher_is_better"] is False, mm
        # a minimised metric's sentinel is +1000, not -1000: worst, not lowest
        assert mm["sentinel"] == 1000 and mm["target"] == 12.889, mm
        assert fitness_spec("no/such/problem")["higher_is_better"] is True
        # a fresh log with no Duration: line reads as running, not done
        assert run_status(p, False) == "running"
        assert run_status(p, True) == "done"
        assert run_status(Path("/nonexistent.log"), False) == "pending"

        # the new trace line: evidence + which tools fired, and it must win
        # over the plain adjustments line for the same wakeup
        trace = {
            "attempt": 7,
            "mutant": 3,
            "trigger": "CI breach",
            "actions": ["adjust_model", "flag_as_outlier"],
            "levers": {
                "cold_start_factor": -1,
                "golden_ratio": 1.2,
                "growth_rate_mult": -1,
                "concurrency_mult": 6.4,
            },
            "flag_outlier_indices": [0, 1],
            "skip_calibration": False,
            "reasoning": "two slow calls dominate the bucket",
            "evidence": {
                "get_progress": "attempts 7/100",
                "get_recent_calls": "#0 28s\n#1 31s",
            },
        }
        trace_log = (
            "2026-07-31 16:00:00.000 | INFO | x | [MUTATION_ATTEMPTED] {}\n"
            f"2026-07-31 16:00:01.000 | INFO | x | [CostMonitorAgentTrace] {json.dumps(trace)}\n"
            "2026-07-31 16:00:02.000 | INFO | x | [CostMonitorHook] adjustments: cold=-1 "
            "golden=1.2 growth=-1 conc=6.4 outliers=2 trigger='CI breach' reason=dup\n"
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".log", delete=False, encoding="utf-8"
        ) as f:
            f.write(trace_log)
            p2 = Path(f.name)
        try:
            ev2, _, _, _ = parse_agent_events(p2)
            assert len(ev2) == 1, ev2  # the adjustments line must not double-count
            e = ev2[0]
            assert e["attempt"] == 7 and e["line"] == 2, e
            assert set(e["moved"]) == {"golden_ratio", "concurrency_mult"}, e
            assert e["levers"]["cold_start_factor"] is None, e
            assert e["actions"] == ["adjust_model", "flag_as_outlier"], e
            assert (
                e["outliers"] == 2 and e["evidence"]["get_progress"] == "attempts 7/100"
            )
        finally:
            p2.unlink(missing_ok=True)

        calls = (
            "2026-07-31 17:00:00.000 | INFO | x | [LLM_CALL] "
            '{"stage": "MutationAgent", "latency_ms": 30000, "tokens_in": 3800, "tokens_out": 1000}\n'
            "2026-07-31 17:00:01.000 | INFO | x | [LLM_CALL] "
            '{"stage": "CostMonitorAgent", "latency_ms": 280660, "tokens_in": 1101, "tokens_out": 39}\n'
            "2026-07-31 17:00:02.000 | INFO | x | [LLM_CALL] "
            '{"stage": "MutationAgent", "latency_ms": 10000, "tokens_in": 100, "tokens_out": 10}\n'
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".log", delete=False, encoding="utf-8"
        ) as f:
            f.write(calls)
            p3 = Path(f.name)
        try:
            _, _, st, _ = parse_agent_events(p3)
            assert st["MutationAgent"]["calls"] == 2, st
            assert st["MutationAgent"]["tokens"] == 4910, st
            assert st[OBSERVER_STAGE]["calls"] == 1, st
            assert st[OBSERVER_STAGE]["max_latency_ms"] == 280660, st
            total = sum(v["latency_ms"] for v in st.values())
            assert st[OBSERVER_STAGE]["latency_ms"] / total > 0.8, st
        finally:
            p3.unlink(missing_ok=True)

        # a launch nobody ever started is unknown; anything else must not 404
        assert driver_state("no-such-experiment-at-all")["known"] is False

        # preflight: a problem importing a module this env lacks must say so
        import tempfile as _tf

        d = Path(_tf.mkdtemp())
        (d / "validate.py").write_text(
            "import vllm_definitely_absent\nimport json\n", encoding="utf-8"
        )
        assert missing_deps(d) == ["vllm_definitely_absent"], missing_deps(d)
        # Local problem modules must not be reported as missing dependencies.
        (d / "helper.py").write_text("import numpy\n", encoding="utf-8")
        (d / "entrypoint.py").write_text(
            "from helper import x\nfrom validate import y\n", encoding="utf-8"
        )
        assert missing_deps(d) == ["vllm_definitely_absent"], missing_deps(d)

        probe_log = "".join(
            f"2026-07-31 18:00:{i:02d}.000 | INFO | x | [LLM_CALL] "
            f'{{"stage": "MutationAgent", "endpoint": "", "model": "q", "ok": true, '
            f'"latency_ms": 20000, "tokens_in": 3000, "tokens_out": 500}}\n'
            f"2026-07-31 18:00:{i:02d}.500 | INFO | x | [MUTATION_ATTEMPTED] "
            f'{{"mutant_id": "m{i}"}}\n'
            for i in range(10)
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".log", delete=False, encoding="utf-8"
        ) as f:
            f.write(probe_log)
            p4 = Path(f.name)
        try:

            def _probe(budget: int, attempts: int = 100):
                return probe_answer(
                    Probe(
                        name="t",
                        task="t",
                        attempts=attempts,
                        budget_tokens=budget,
                        proc=None,
                        log=p4,
                    )
                )

            a = _probe(10_000_000)
            assert a["observed_attempts"] == 10, a
            assert a["ci"][0] <= a["predicted_tokens"] <= a["ci"][1], a
            assert a["verdict"] == "fits", a
            assert 200_000 < a["predicted_tokens"] < 600_000, a
            n_lo, n_hi = a["affordable"]
            assert 0 < n_lo <= n_hi, a

            over = _probe(1_000)
            assert over["verdict"] == "over", over
            assert over["affordable"] == [0, 0], over  # already overspent

            tight = _probe(int(a["predicted_tokens"]) + 1)
            assert tight["verdict"] == "tight", tight
            t_lo, t_hi = tight["affordable"]
            assert 0 < t_lo <= t_hi, tight

            assert (
                probe_answer(
                    Probe(
                        name="t",
                        task="t",
                        attempts=100,
                        budget_tokens=1,
                        proc=None,
                        log=Path("/nope.log"),
                    )
                )
                is None
            )
        finally:
            p4.unlink(missing_ok=True)
        print("selftest OK")
    finally:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        import uvicorn

        # 0.0.0.0: the server is reached from the operator's laptop over the LAN.
        uvicorn.run(app, host="0.0.0.0", port=PORT)
