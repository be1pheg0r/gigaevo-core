#!/usr/bin/env python3
"""Cost Lab — web console for the cost-prediction ablation.

Launch `tools/cost_ablation/run_ablation.py` from a browser, watch the runs
while they happen, and read afterwards whether CostMonitorAgent actually
improved on the automatic estimator.

Backend is deliberately thin: it shells out to the same run_ablation.py the
CLI uses (no second launch path to keep in sync), reuses build_report.py's
log parser, and ships raw prediction series to the frontend, which computes
the error grid itself. The only thing parsed here that build_report.py does
not already parse is the agent's own adjustment log line — the record of
when the agent woke, what tripped it, and which lever it moved.

Run on the summer-school server:
    cd ~/gigaevo-core && nohup python3 tools/cost_lab_web/app.py > ~/cost_lab_web.log 2>&1 &
"""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "cost_ablation"))

from build_report import parse_log  # noqa: E402  (reuse, don't reimplement)

EXPERIMENTS = REPO / "experiments"
STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.environ.get("COST_LAB_PORT", "8091"))

# A run whose log has not grown in this long, and which has no Duration:
# line, is treated as dead rather than running — survives an app restart
# without having to track pids across processes.
STALE_AFTER_S = 180

ADJ_RE = re.compile(
    r"\[CostMonitorHook\] adjustments: cold=(\S+) golden=(\S+) growth=(\S+) conc=(\S+) "
    r"outliers=(\d+) trigger='([^']*)' reason=(.*)"
)
TRACE_RE = re.compile(r"\[CostMonitorAgentTrace\] (\{.*\})")
ATT_RE = re.compile(r"\[MUTATION_ATTEMPTED\]")
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+")

# Imports every problem gets for free — never reported as a missing dependency.
_FIRST_PARTY = {"problems", "gigaevo", "config", "tools"}
_dep_cache: dict[str, tuple[float, list[str]]] = {}


def missing_deps(problem_dir: Path) -> list[str]:
    """Third-party modules a problem imports that this environment lacks.

    Catches the trap that `prompts/sudoku` walked into: the problem exists and
    launches fine, then every single mutant dies in the validator because the
    box has no `vllm`. Cheaper to say so before spending an hour of GPU.
    """
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

    missing = []
    for m in sorted(mods):
        if m in _FIRST_PARTY or m in sys.stdlib_module_names:
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
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, timeout=120).stdout.strip()


def repo_state() -> dict:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    counts = _git("rev-list", "--left-right", "--count", f"origin/{branch}...HEAD")
    behind, ahead = (counts.split() + ["0", "0"])[:2] if counts else ("0", "0")
    return {
        "branch": branch,
        "head": _git("rev-parse", "--short", "HEAD"),
        "subject": _git("log", "-1", "--format=%s"),
        "authored": _git("log", "-1", "--format=%ar"),
        "behind": int(behind), "ahead": int(ahead),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


# ------------------------------------------------------------------ problems

def list_problems() -> list[dict]:
    """Every runnable problem, as `problem.name` values grouped by family."""
    out = []
    for metrics in sorted((REPO / "problems").rglob("metrics.yaml")):
        name = metrics.parent.relative_to(REPO / "problems").as_posix()
        parts = name.split("/")
        out.append({
            "name": name,
            "family": parts[0] if len(parts) > 1 else "standalone",
            "label": "/".join(parts[1:]) if len(parts) > 1 else name,
            "seeds": len(list((metrics.parent / "initial_programs").glob("*.py"))),
            "missing_deps": missing_deps(metrics.parent),
        })
    return out


# --------------------------------------------------------------- experiments

def parse_agent_events(path: Path) -> tuple[list[dict], int]:
    """The agent's own wakeup record: attempt, trigger, levers, reasoning.

    Returns the events plus the total `[MUTATION_ATTEMPTED]` count, which is
    the fallback progress figure for logs written before the prediction line
    carried an ``attempts`` field.

    A lever value of -1 means 'agent declined to move this one', so an event
    where all four are -1 is a wakeup that changed nothing — worth showing,
    because a monitor that wakes constantly and never acts is its own finding.
    """
    traces: list[dict] = []   # new format: decision + the evidence behind it
    legacy: list[dict] = []   # pre-trace logs: the decision alone
    attempts = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if ATT_RE.search(line):
                attempts += 1
                continue
            m = TRACE_RE.search(line)
            if m:
                try:
                    t = json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
                levers = {k: (None if v is None or v < 0 else v)
                          for k, v in (t.get("levers") or {}).items()}
                ts = TS_RE.match(line)
                traces.append({
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
                })
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

            levers = {"cold_start": num(cold), "golden_ratio": num(golden),
                      "growth_rate": num(growth), "concurrency": num(conc)}
            ts = TS_RE.match(line)
            legacy.append({
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
            })
    # Both lines are written for the same wakeup, so a log that has traces
    # must be read only through them — the legacy line would double-count.
    return (traces or legacy), attempts


def run_status(log_path: Path, has_duration: bool) -> str:
    if has_duration:
        return "done"
    if not log_path.exists():
        return "pending"
    if time.time() - log_path.stat().st_mtime > STALE_AFTER_S:
        return "stopped"
    return "running"


def summarize_run(entry: dict) -> dict:
    log_path = REPO / entry["log"]
    series, duration, actual_tokens = ([], None, 0)
    events: list[dict] = []
    attempts = 0
    if log_path.exists():
        series, duration, actual_tokens = parse_log(log_path)
        events, attempts = parse_agent_events(log_path)
    return {
        **{k: entry[k] for k in ("task", "key", "condition", "db")},
        "log": entry["log"],
        "stem": Path(entry["log"]).stem,
        "status": run_status(log_path, duration is not None),
        "series": series,
        "actual_duration": duration,
        "actual_tokens": actual_tokens or None,
        "agent_events": events,
        "attempts": (series[-1].get("attempts") if series else 0) or attempts,
        "updated": log_path.stat().st_mtime if log_path.exists() else None,
    }


def experiment_dirs() -> list[Path]:
    if not EXPERIMENTS.exists():
        return []
    return sorted((d for d in EXPERIMENTS.iterdir()
                   if d.is_dir() and (d / "manifest.json").exists()),
                  key=lambda d: d.stat().st_mtime, reverse=True)


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
    r = subprocess.run(["git", "pull", "--ff-only", "origin", branch],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    return {"ok": r.returncode == 0,
            "output": (r.stdout + r.stderr).strip()[-4000:],
            "repo": repo_state()}


@app.get("/api/problems")
async def api_problems() -> list[dict]:
    return list_problems()


@app.get("/api/experiments")
async def api_experiments() -> list[dict]:
    out = []
    for d in experiment_dirs():
        manifest = load_manifest(d)
        statuses = [run_status(REPO / e["log"], "Duration:" in _peek(REPO / e["log"]))
                    for e in manifest]
        out.append({
            "name": d.name,
            "created": d.stat().st_mtime,
            "n_runs": len(manifest),
            "tasks": sorted({e["task"] for e in manifest}),
            "running": statuses.count("running"),
            "done": statuses.count("done"),
            "launching": d.name in _launches and _launches[d.name].proc.poll() is None,
        })
    return out


# ponytail: fixed-size tail instead of parsing the whole log on every poll —
# run.py's `Duration:` line is followed by shutdown chatter, so the window has
# to be generous. If a run ever logs more than this after finishing, the list
# will call it "stopped" while the detail view (full parse) says "done"; move
# to an index file if that shows up.
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
    if not (d / "manifest.json").exists():
        raise HTTPException(404, "unknown experiment")
    manifest = load_manifest(d)
    runs = [summarize_run(e) for e in manifest]
    report = sorted(p.name for p in (d / "report").glob("*")) if (d / "report").exists() else []
    launch = _launches.get(name)
    return {
        "name": name,
        "created": d.stat().st_mtime,
        "runs": runs,
        "report": report,
        "driver_alive": bool(launch and launch.proc.poll() is None),
    }


@app.post("/api/experiments")
async def api_launch(body: NewExperiment) -> dict:
    known = {p["name"] for p in list_problems()}
    unknown = [t for t in body.tasks if t not in known]
    if unknown:
        raise HTTPException(400, f"unknown problems: {', '.join(unknown)}")

    slug = re.sub(r"[^a-z0-9]+", "-", body.label.lower()).strip("-") or "ablation"
    name = f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
    cmd = [sys.executable, "-u", "tools/cost_ablation/run_ablation.py",
           "--tasks", *body.tasks,
           "--max-mutants", str(body.max_mutants),
           "--wave-size", str(body.wave_size),
           "--llm", body.llm,
           "--skip-report",
           "--out-dir", f"experiments/{name}"]
    log = EXPERIMENTS / f"{name}.driver.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)  # outlives an app restart
    _launches[name] = Launch(name=name, proc=proc, cmd=cmd)
    return {"name": name, "cmd": " ".join(cmd)}


@app.post("/api/experiments/{name}/stop")
async def api_stop(name: str) -> dict:
    launch = _launches.get(name)
    killed = 0
    if launch and launch.proc.poll() is None:
        launch.proc.kill()
        killed += 1
    # the driver's children are separate run.py processes — stop them by out-dir
    r = subprocess.run(["pkill", "-f", f"experiments/{name}"], capture_output=True)
    return {"ok": True, "driver_killed": killed, "pkill_rc": r.returncode}


@app.post("/api/experiments/{name}/report")
async def api_report(name: str) -> dict:
    d = EXPERIMENTS / name
    if not (d / "manifest.json").exists():
        raise HTTPException(404, "unknown experiment")
    r = subprocess.run([sys.executable, "tools/cost_ablation/build_report.py",
                        "--manifest", str(d / "manifest.json")],
                       cwd=REPO, capture_output=True, text=True, timeout=900)
    files = sorted(p.name for p in (d / "report").glob("*")) if (d / "report").exists() else []
    return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr)[-4000:], "files": files}


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
        buf, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})


@app.get("/api/experiments/{name}/report/{filename}")
async def api_report_file(name: str, filename: str) -> FileResponse:
    path = (EXPERIMENTS / name / "report" / filename).resolve()
    if not path.is_file() or (EXPERIMENTS / name / "report").resolve() not in path.parents:
        raise HTTPException(404, "no such report file")
    return FileResponse(path)


@app.get("/api/experiments/{name}/log/{stem}")
async def api_log(name: str, stem: str, tail: int = 400, around: int = 0,
                  ctx: int = 60, q: str = "") -> dict:
    """Log slice. `around` centres the window on a 1-indexed line (that's
    what an agent trace carries), `q` filters, otherwise it's the tail."""
    path = (EXPERIMENTS / name / f"{stem}.log").resolve()
    if not path.is_file() or (EXPERIMENTS / name).resolve() not in path.parents:
        raise HTTPException(404, "no such log")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    if q:
        hits = [(i, ln) for i, ln in enumerate(lines, 1) if q.lower() in ln.lower()]
        return {"stem": stem, "total": len(lines), "first_line": 0, "query": q,
                "matches": len(hits),
                "numbered": [{"n": i, "text": t} for i, t in hits[:tail]]}

    if around > 0:
        lo = max(0, around - ctx - 1)
        hi = min(len(lines), around + ctx)
    else:
        lo, hi = max(0, len(lines) - tail), len(lines)
    return {"stem": stem, "total": len(lines), "first_line": lo + 1, "focus": around,
            "numbered": [{"n": lo + i + 1, "text": t} for i, t in enumerate(lines[lo:hi])]}


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
    )
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
        f.write(log)
        p = Path(f.name)
    try:
        ev, attempts = parse_agent_events(p)
        assert len(ev) == 2, ev
        assert attempts == 3, attempts  # fallback progress for pre-`attempts` logs
        assert all(e["evidence"] == {} for e in ev)  # old format carries no evidence
        # a wakeup that declined every lever is still recorded, with nothing moved
        assert ev[0]["attempt"] == 2 and ev[0]["moved"] == [], ev[0]
        assert ev[0]["trigger"].startswith("estimate jumped"), ev[0]
        # -1 means "declined", so it must not be reported as a value
        assert ev[0]["levers"]["cold_start"] is None
        assert ev[1]["attempt"] == 3, ev[1]
        assert set(ev[1]["moved"]) == {"cold_start", "growth_rate", "concurrency"}, ev[1]
        assert ev[1]["levers"]["golden_ratio"] is None
        assert ev[1]["outliers"] == 2
        assert ev[1]["reason"] == "latency outliers dominate the bucket"
        # a fresh log with no Duration: line reads as running, not done
        assert run_status(p, False) == "running"
        assert run_status(p, True) == "done"
        assert run_status(Path("/nonexistent.log"), False) == "pending"

        # the new trace line: evidence + which tools fired, and it must win
        # over the plain adjustments line for the same wakeup
        trace = {
            "attempt": 7, "mutant": 3, "trigger": "CI breach",
            "actions": ["adjust_model", "flag_as_outlier"],
            "levers": {"cold_start_factor": -1, "golden_ratio": 1.2,
                       "growth_rate_mult": -1, "concurrency_mult": 6.4},
            "flag_outlier_indices": [0, 1], "skip_calibration": False,
            "reasoning": "two slow calls dominate the bucket",
            "evidence": {"get_progress": "attempts 7/100", "get_recent_calls": "#0 28s\n#1 31s"},
        }
        trace_log = (
            "2026-07-31 16:00:00.000 | INFO | x | [MUTATION_ATTEMPTED] {}\n"
            f"2026-07-31 16:00:01.000 | INFO | x | [CostMonitorAgentTrace] {json.dumps(trace)}\n"
            "2026-07-31 16:00:02.000 | INFO | x | [CostMonitorHook] adjustments: cold=-1 "
            "golden=1.2 growth=-1 conc=6.4 outliers=2 trigger='CI breach' reason=dup\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write(trace_log)
            p2 = Path(f.name)
        try:
            ev2, _ = parse_agent_events(p2)
            assert len(ev2) == 1, ev2  # the adjustments line must not double-count
            e = ev2[0]
            assert e["attempt"] == 7 and e["line"] == 2, e
            assert set(e["moved"]) == {"golden_ratio", "concurrency_mult"}, e
            assert e["levers"]["cold_start_factor"] is None, e
            assert e["actions"] == ["adjust_model", "flag_as_outlier"], e
            assert e["outliers"] == 2 and e["evidence"]["get_progress"] == "attempts 7/100"
        finally:
            p2.unlink(missing_ok=True)

        # preflight: a problem importing a module this env lacks must say so
        import tempfile as _tf
        d = Path(_tf.mkdtemp())
        (d / "validate.py").write_text("import vllm_definitely_absent\nimport json\n", encoding="utf-8")
        assert missing_deps(d) == ["vllm_definitely_absent"], missing_deps(d)
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
