#!/usr/bin/env python3
"""Public, resource-bounded token-budget playground.

This service deliberately exposes only the token projection from CostMonitor.
It has no route for launching an experiment, choosing an LLM, changing the
probe size, reading logs, or stopping arbitrary processes. Every new measurement
runs exactly ``PROBE_ATTEMPTS`` mutations for a curated task, with one probe
allowed at a time.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

REPO = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
PROBE_DIR = REPO / "experiments" / "token_playground"
sys.path.insert(0, str(REPO / "tools" / "cost_ablation"))

from replay_from_log import hook_from_log  # noqa: E402
from run_ablation import find_free_dbs, launch  # noqa: E402

PORT = int(os.environ.get("TOKEN_PLAYGROUND_PORT", "8092"))
LLM_CONFIG = os.environ.get("TOKEN_PLAYGROUND_LLM", "single")

# These are product safety limits, not user-configurable defaults.
PROBE_ATTEMPTS = 10
PROBE_TIMEOUT_S = 15 * 60
MAX_TARGET_ATTEMPTS = 1_000
MAX_BUDGET_TOKENS = 20_000_000
MAX_ACTIVE_PROBES = 1
MAX_DAILY_PROBES = 12
IP_COOLDOWN_S = 30
MAX_REQUEST_BYTES = 4_096
MAX_JOBS = 500
JOB_TTL_S = 60 * 60
API_RATE_WINDOW_S = 60
MAX_API_REQUESTS_PER_WINDOW = 180

TASK_CATALOG = {
    "alphaevolve/packing_circles/n_26": {
        "label": "Packing Circles · 26",
        "family": "Geometry",
        "note": "Упаковка 26 равных окружностей в единичный квадрат.",
    },
    "alphaevolve/packing_circles/n_32": {
        "label": "Packing Circles · 32",
        "family": "Geometry",
        "note": "Упаковка 32 равных окружностей в единичный квадрат.",
    },
    "alphaevolve/heilbronn_convex/points_13": {
        "label": "Heilbronn · 13 points",
        "family": "Geometry",
        "note": "Максимизация площади минимального треугольника.",
    },
    "alphaevolve/heilbronn_convex/points_14": {
        "label": "Heilbronn · 14 points",
        "family": "Geometry",
        "note": "Размещение 14 точек с максимальной площадью минимального треугольника.",
    },
    "alphaevolve/minimize_max_min_dist_ratio/2_dimensions": {
        "label": "Distance Ratio · 2D",
        "family": "Geometry",
        "note": "Минимизация отношения максимального расстояния к минимальному на плоскости.",
    },
    "alphaevolve/minimize_max_min_dist_ratio/3_dimensions": {
        "label": "Distance Ratio · 3D",
        "family": "Geometry",
        "note": "Минимизация отношения максимального расстояния к минимальному в пространстве.",
    },
    "alphaevolve/erdos_minimum_overlap": {
        "label": "Erdős Minimum Overlap",
        "family": "Combinatorics",
        "note": "Поиск конструкции с минимальным перекрытием.",
    },
    "alphaevolve/sums_diffs_finite_sets": {
        "label": "Sums & Differences of Finite Sets",
        "family": "Combinatorics",
        "note": "Оптимизация соотношения сумм и разностей конечных множеств.",
    },
    "alphaevolve/matrix_multiplication/2_4_5": {
        "label": "Matrix Multiplication · 2×4×5",
        "family": "Algebra",
        "note": "Поиск эффективной схемы матричного умножения размера 2×4×5.",
    },
    "alphaevolve/first_autocorr_ineq": {
        "label": "First Autocorrelation Inequality",
        "family": "Analysis",
        "note": "Улучшение первой автокорреляционной оценки.",
    },
    "alphaevolve/second_autocorr_ineq": {
        "label": "Second Autocorrelation Inequality",
        "family": "Analysis",
        "note": "Улучшение второй автокорреляционной оценки.",
    },
    "alphaevolve/second_autocorr_ineq_improver": {
        "label": "Second Autocorrelation · Improver",
        "family": "Analysis",
        "note": "Улучшение найденных решений второй автокорреляционной задачи.",
    },
    "alphaevolve/third_autocorr_ineq": {
        "label": "Third Autocorrelation Inequality",
        "family": "Analysis",
        "note": "Улучшение третьей автокорреляционной оценки.",
    },
    "alphaevolve/uncertainty_inequality": {
        "label": "Uncertainty Inequality",
        "family": "Analysis",
        "note": "Поиск усиленной конструкции для неравенства неопределённости.",
    },
}


class EstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    attempts: int = Field(default=250, ge=25, le=MAX_TARGET_ATTEMPTS)
    budget_tokens: int = Field(default=3_000_000, ge=100_000, le=MAX_BUDGET_TOKENS)


@dataclass
class ProbeRun:
    task: str
    process: subprocess.Popen
    log: Path
    started_at: float
    completed_at: float | None = None
    hook: Any | None = None
    timed_out: bool = False


@dataclass
class EstimateJob:
    job_id: str
    task: str
    attempts: int
    budget_tokens: int
    probe: ProbeRun
    answer_stamp: tuple[int, int] | None = None
    answer: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)


app = FastAPI(
    title="GigaEvo Token Playground",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_start_lock = asyncio.Lock()
_active: ProbeRun | None = None
_jobs: dict[str, EstimateJob] = {}
_starts_24h: deque[float] = deque()
_ip_last_start: dict[str, float] = {}
_api_hits: dict[str, deque[float]] = {}


def _client_ip(request: Request) -> str:
    # nginx must overwrite, not append, X-Forwarded-For (see README).
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",", 1)[0].strip() or (
        request.client.host if request.client else "unknown"
    )


def _problem_exists(task: str) -> bool:
    problem = REPO / "problems" / task
    return (problem / "validate.py").is_file() and (problem / "metrics.yaml").is_file()


def _probe_log_path(task: str, run_id: str) -> Path:
    task_slug = task.replace("/", "__").replace("\\", "__")
    return PROBE_DIR / f"probe_{task_slug}_{run_id}.log"


def _load_hook(probe: ProbeRun) -> Any | None:
    if probe.hook is not None:
        return probe.hook
    if not probe.log.exists() or probe.log.stat().st_size == 0:
        return None
    try:
        hook = hook_from_log(probe.log)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if hook is None or not hook._tokens_by_stage:  # noqa: SLF001 - projection seam
        return None
    if probe.process.poll() is not None:
        probe.hook = hook
    return hook


def _refresh_active() -> ProbeRun | None:
    global _active
    if _active is None:
        return None
    now = time.time()
    if _active.process.poll() is None and now - _active.started_at > PROBE_TIMEOUT_S:
        _active.process.terminate()
        _active.timed_out = True
    if _active.process.poll() is None:
        return _active
    _active.completed_at = _active.completed_at or now
    finished = _active
    _active = None
    return finished


def _projection(job: EstimateJob) -> dict[str, Any] | None:
    hook = _load_hook(job.probe)
    if hook is None:
        return None
    stat = job.probe.log.stat()
    stamp = (stat.st_size, stat.st_mtime_ns)
    if job.answer is not None and job.answer_stamp == stamp:
        return job.answer

    point, (lo, hi) = hook.project_tokens(job.attempts)
    if point <= 0:
        return None
    q75 = hook.project_tokens(job.attempts, alpha=0.5)[1][1]
    n_hi = hook.affordable_attempts(job.budget_tokens, hi=MAX_TARGET_ATTEMPTS)
    pessimism = max(q75 / point, 1.0)
    n_lo = hook.affordable_attempts(
        job.budget_tokens / pessimism, hi=MAX_TARGET_ATTEMPTS
    )
    fits = hi <= job.budget_tokens
    answer = {
        "observed_attempts": int(hook._attempts),  # noqa: SLF001 - UI provenance
        "target_attempts": job.attempts,
        "budget_tokens": job.budget_tokens,
        "predicted_tokens": round(point),
        "interval": [round(lo), round(hi)],
        "verdict": "fits"
        if fits
        else ("tight" if point <= job.budget_tokens else "over"),
        "affordable_attempts": [min(n_lo, n_hi), max(n_lo, n_hi)],
    }
    job.answer_stamp = stamp
    job.answer = answer
    return answer


def _prune_limits(now: float) -> None:
    while _starts_24h and now - _starts_24h[0] >= 24 * 60 * 60:
        _starts_24h.popleft()
    stale_ips = [
        ip for ip, stamp in _ip_last_start.items() if now - stamp >= 24 * 60 * 60
    ]
    for ip in stale_ips:
        _ip_last_start.pop(ip, None)
    stale_jobs = [
        job_id for job_id, job in _jobs.items() if now - job.created_at >= JOB_TTL_S
    ]
    for job_id in stale_jobs:
        _jobs.pop(job_id, None)


def _api_rate_ok(ip: str, now: float) -> bool:
    hits = _api_hits.setdefault(ip, deque())
    while hits and now - hits[0] >= API_RATE_WINDOW_S:
        hits.popleft()
    if not hits:
        for stale_ip in [key for key, values in _api_hits.items() if not values]:
            if stale_ip != ip:
                _api_hits.pop(stale_ip, None)
    if len(hits) >= MAX_API_REQUESTS_PER_WINDOW:
        return False
    hits.append(now)
    return True


@app.middleware("http")
async def public_safety_headers(request: Request, call_next):
    content_length = request.headers.get("content-length")
    try:
        too_large = bool(content_length) and int(content_length) > MAX_REQUEST_BYTES
    except ValueError:
        too_large = True
    if too_large:
        return JSONResponse({"detail": "request is too large"}, status_code=413)
    if request.url.path.startswith("/api/") and not _api_rate_ok(
        _client_ip(request), time.time()
    ):
        return JSONResponse(
            {"detail": "Слишком много запросов. Попробуйте через минуту."},
            status_code=429,
            headers={"Retry-After": str(API_RATE_WINDOW_S)},
        )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
async def health() -> dict[str, Any]:
    _refresh_active()
    return {
        "ok": True,
        "active_probes": 1 if _active else 0,
        "probe_attempts": PROBE_ATTEMPTS,
    }


@app.get("/api/catalog")
async def catalog() -> dict[str, Any]:
    _refresh_active()
    now = time.time()
    _prune_limits(now)
    tasks = [
        {"name": name, **meta}
        for name, meta in TASK_CATALOG.items()
        if _problem_exists(name)
    ]
    return {
        "tasks": tasks,
        "limits": {
            "probe_attempts": PROBE_ATTEMPTS,
            "target_attempts_max": MAX_TARGET_ATTEMPTS,
            "budget_tokens_max": MAX_BUDGET_TOKENS,
            "one_probe_at_a_time": MAX_ACTIVE_PROBES == 1,
            "starts_remaining_today": max(0, MAX_DAILY_PROBES - len(_starts_24h)),
        },
        "busy": _active is not None,
        "active_task": _active.task if _active else None,
    }


@app.post("/api/estimate", status_code=202)
async def start_estimate(body: EstimateRequest, request: Request) -> dict[str, Any]:
    global _active
    if body.task not in TASK_CATALOG or not _problem_exists(body.task):
        raise HTTPException(400, "Эта задача недоступна в публичном playground.")

    async with _start_lock:
        _refresh_active()
        now = time.time()
        _prune_limits(now)
        if len(_jobs) >= MAX_JOBS:
            raise HTTPException(503, "Очередь playground заполнена. Попробуйте позже.")

        probe = None
        if _active is not None and _active.task == body.task:
            probe = _active
        if probe is None and _active is not None:
            raise HTTPException(
                429, "Сейчас идёт другая прикидка. Попробуйте через несколько минут."
            )

        if probe is None:
            ip = _client_ip(request)
            last = _ip_last_start.get(ip, 0.0)
            if now - last < IP_COOLDOWN_S:
                wait_s = round(IP_COOLDOWN_S - (now - last))
                raise HTTPException(
                    429, f"Для нового замера с этого адреса подождите {wait_s} с."
                )
            if len(_starts_24h) >= MAX_DAILY_PROBES:
                raise HTTPException(429, "Дневной лимит новых замеров исчерпан.")
            try:
                db = find_free_dbs(1)[0]
            except (RuntimeError, OSError) as exc:
                raise HTTPException(
                    503, "Нет свободного слота для короткой прикидки."
                ) from exc

            PROBE_DIR.mkdir(parents=True, exist_ok=True)
            run_id = secrets.token_hex(6)
            log = _probe_log_path(body.task, run_id)
            process = launch(
                body.task, db, PROBE_ATTEMPTS, LLM_CONFIG, "agentless", log
            )
            probe = ProbeRun(task=body.task, process=process, log=log, started_at=now)
            _active = probe
            _starts_24h.append(now)
            _ip_last_start[ip] = now

        job_id = secrets.token_urlsafe(12)
        job = EstimateJob(
            job_id=job_id,
            task=body.task,
            attempts=body.attempts,
            budget_tokens=body.budget_tokens,
            probe=probe,
        )
        _jobs[job_id] = job
        return {
            "job_id": job_id,
            "probe_attempts": PROBE_ATTEMPTS,
            "source": "shared_probe"
            if probe is _active and probe.started_at < now
            else "new_probe",
        }


@app.get("/api/estimate/{job_id}")
async def estimate_state(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Неизвестная прикидка.")
    _refresh_active()
    rc = job.probe.process.poll()
    answer = _projection(job)
    alive = rc is None and not job.probe.timed_out
    failed = not alive and answer is None
    return {
        "job_id": job_id,
        "task": job.task,
        "alive": alive,
        "failed": failed,
        "timed_out": job.probe.timed_out,
        "answer": answer,
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def selftest() -> None:
    post_paths = {
        route.path for route in app.routes if "POST" in getattr(route, "methods", set())
    }
    public_paths = {route.path for route in app.routes}
    assert post_paths == {"/api/estimate"}
    forbidden_segments = {"experiment", "experiments", "launch", "stop", "log", "logs"}
    assert not any(
        forbidden_segments.intersection(path.strip("/").split("/"))
        for path in public_paths
    )
    assert PROBE_ATTEMPTS == 10 and MAX_ACTIVE_PROBES == 1 and IP_COOLDOWN_S == 30
    assert TASK_CATALOG and all(_problem_exists(task) for task in TASK_CATALOG)
    alphaevolve_root = REPO / "problems" / "alphaevolve"
    discovered_tasks = {
        "alphaevolve/" + metrics.parent.relative_to(alphaevolve_root).as_posix()
        for metrics in alphaevolve_root.rglob("metrics.yaml")
        if (metrics.parent / "validate.py").is_file()
    }
    assert set(TASK_CATALOG) == discovered_tasks
    probe_log = _probe_log_path("alphaevolve/packing_circles/n_26", "abc123")
    assert (
        probe_log.parent == PROBE_DIR
        and probe_log.name == "probe_alphaevolve__packing_circles__n_26_abc123.log"
    )
    assert (
        EstimateRequest(task="alphaevolve/packing_circles/n_26").attempts
        <= MAX_TARGET_ATTEMPTS
    )
    try:
        EstimateRequest(task="x", attempts=MAX_TARGET_ATTEMPTS + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("target-attempt cap is not enforced")
    print(
        json.dumps(
            {
                "ok": True,
                "post_routes": sorted(post_paths),
                "probe_attempts": PROBE_ATTEMPTS,
            }
        )
    )


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=PORT, access_log=False)
