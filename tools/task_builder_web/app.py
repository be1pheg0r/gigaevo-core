#!/usr/bin/env python3
"""Minimal demo web GUI for the request -> gigaevo-task pipeline.

FastAPI + a single static page (vanilla JS, Plotly via CDN) -- no frontend
framework. Flow per job: guard -> build -> scaffold (all three via the
existing gigaevo.problems.task_creator.create_task_from_request) -> launch
a small seed run (`run.py ... +cost_monitor=enabled`) -> tail its log,
parsing the [CostMonitorHookJSON] lines CostMonitorHook already emits
into a live predicted-vs-actual series for the frontend's Plotly charts.

Run: SUMMER_SCHOOL_LLM_KEY_A=... python tools/task_builder_web/app.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from gigaevo.llm.agents.factories import (  # noqa: E402
    create_code_writer_agent,
    create_task_builder_agent,
    create_task_guard_agent,
)
from gigaevo.problems.context import ProblemContext  # noqa: E402
from gigaevo.problems.layout import ProblemLayout  # noqa: E402
from gigaevo.problems.task_creator import _validate_problem_name  # noqa: E402

CATEGORIES_PATH = REPO / "config" / "task_builder" / "categories.yaml"
LOGS_DIR = REPO / "experiments" / "task_builder_web"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MAX_MUTANTS = 12
SEED_TIMEOUT_S = 900
REDIS_DB_POOL = list(range(50, 60))

_JSON_RE = re.compile(r"\[CostMonitorHookJSON\] (\{.*\})")
_LLM_CALL_RE = re.compile(r"\[LLM_CALL\] (\{.*\})")


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen3.5-9b",
        api_key=os.environ["SUMMER_SCHOOL_LLM_KEY_A"],
        base_url="http://82.202.157.243:8080/v1",
        temperature=1.0,
        max_tokens=4096,
        request_timeout=120,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def _make_code_llm() -> ChatOpenAI:
    """Separate, higher-max_tokens LLM for CodeWriterAgent.

    Confirmed live: a verbose validate.py response hit the shared
    max_tokens=4096 mid-generation (openai.LengthFinishReasonError) --
    real code files with comments routinely run longer than a
    ProblemConfig/guard classification response.
    """
    return ChatOpenAI(
        model="qwen3.5-9b",
        api_key=os.environ["SUMMER_SCHOOL_LLM_KEY_A"],
        base_url="http://82.202.157.243:8080/v1",
        temperature=1.0,
        max_tokens=8192,
        request_timeout=180,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


@dataclass
class MutantPoint:
    mutant: int
    predicted_tokens: int
    token_ci_low: int
    token_ci_high: int
    predicted_duration_s: float
    ci_low_s: float
    ci_high_s: float
    elapsed_s: float
    actual_tokens_so_far: int


STAGES = ["guarding", "building", "scaffolding", "writing_code", "seeding", "estimating", "done"]


@dataclass
class Job:
    id: str
    request: str
    stage: str = "guarding"
    accepted: bool | None = None
    reason: str = ""
    domain_hint: str | None = None
    config_summary: dict | None = None
    problem_name: str | None = None
    problem_dir: str | None = None
    redis_db: int | None = None
    initial_estimate: dict | None = None  # first CostMonitorHookJSON point
    points: list[MutantPoint] = field(default_factory=list)
    final_actual_tokens: int | None = None
    final_actual_duration_s: float | None = None
    log_tail: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    _proc: asyncio.subprocess.Process | None = None

    def public_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["points"] = [p.__dict__ for p in self.points]
        d["log_tail"] = self.log_tail[-60:]
        return d


_jobs: dict[str, Job] = {}
_db_cursor = 0


def _next_redis_db() -> int:
    global _db_cursor
    db = REDIS_DB_POOL[_db_cursor % len(REDIS_DB_POOL)]
    _db_cursor += 1
    return db


async def _tail_seed_run(job: Job, log_path: Path) -> None:
    """Poll the run.py log file until the process exits, updating job.points
    from [CostMonitorHookJSON] lines and actual token spend from
    [LLM_CALL] lines."""
    actual_tokens = 0
    t0 = time.monotonic()
    pos = 0
    proc = job._proc
    assert proc is not None

    while True:
        await asyncio.sleep(1.5)
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            new, pos = text[pos:], len(text)
            for line in new.splitlines():
                job.log_tail.append(line)
                m = _LLM_CALL_RE.search(line)
                if m:
                    try:
                        d = json.loads(m.group(1))
                        actual_tokens += d.get("tokens_in", 0) + d.get("tokens_out", 0)
                    except Exception:
                        pass
                m = _JSON_RE.search(line)
                if m:
                    try:
                        d = json.loads(m.group(1))
                    except Exception:
                        continue
                    point = MutantPoint(
                        mutant=d["mutant"],
                        predicted_tokens=d["predicted_tokens"],
                        token_ci_low=d["token_ci_low"],
                        token_ci_high=d["token_ci_high"],
                        predicted_duration_s=d["predicted_duration_s"],
                        ci_low_s=d["ci_low_s"],
                        ci_high_s=d["ci_high_s"],
                        elapsed_s=time.monotonic() - t0,
                        actual_tokens_so_far=actual_tokens,
                    )
                    job.points.append(point)
                    if job.initial_estimate is None:
                        job.initial_estimate = d
                        job.stage = "estimating"
        job.log_tail[:] = job.log_tail[-200:]
        if proc.returncode is not None:
            break

    job.final_actual_tokens = actual_tokens
    job.final_actual_duration_s = time.monotonic() - t0
    job.stage = "done"


async def _write_real_code(config, target_dir: Path) -> None:
    """Replace every scaffolded stub (``# TODO`` + ``pass``) with a real
    implementation.

    ProblemLayout's jinja templates only emit a signature + docstring +
    TODO body -- a freshly scaffolded problem is not runnable yet. Left
    as-is, every seed program returns ``None``, CallProgramFunction/
    CallValidatorFunction AUTO-SKIP, ``is_valid=0`` always, and the
    archive never gets a single accepted mutant (confirmed live: a demo
    job stalled with 0 accepted after its full attempt budget for exactly
    this reason).
    """
    writer = create_code_writer_agent(_make_code_llm())
    task_description = config.task_description.objective

    for prog in config.initial_programs:
        path = target_dir / "initial_programs" / f"{prog.name}.py"
        stub = path.read_text(encoding="utf-8")
        code = await writer.arun(
            problem_name=config.name,
            task_description=task_description,
            file_kind="initial_program",
            file_purpose=f"Seed strategy '{prog.name}': {prog.description}",
            stub_code=stub,
        )
        path.write_text(code, encoding="utf-8")

    if config.add_context:
        context_path = target_dir / "context.py"
        stub = context_path.read_text(encoding="utf-8")
        fields_desc = "\n".join(
            f"- {k}: {v}" for k, v in (config.context_spec.fields if config.context_spec else {}).items()
        ) or "(no field breakdown given -- use your judgement from the docstring)"
        code = await writer.arun(
            problem_name=config.name,
            task_description=task_description,
            file_kind="context",
            file_purpose=(
                "Build the read-only data entrypoint()/validate() receive as "
                f"`context`. Return a real, non-empty dict -- an empty dict "
                "means every downstream call sees no test data and is always "
                f"rejected as invalid. Fields:\n{fields_desc}"
            ),
            stub_code=stub,
        )
        context_path.write_text(code, encoding="utf-8")

    validate_path = target_dir / "validate.py"
    stub = validate_path.read_text(encoding="utf-8")
    metrics_desc = "\n".join(
        f"- {name}: {spec.description} (primary={spec.is_primary}, "
        f"higher_is_better={spec.higher_is_better})"
        for name, spec in config.metrics.items()
    )
    code = await writer.arun(
        problem_name=config.name,
        task_description=task_description,
        file_kind="validate",
        file_purpose=f"Score a solution and report validity. Metrics:\n{metrics_desc}",
        stub_code=stub,
    )
    validate_path.write_text(code, encoding="utf-8")


async def _run_job(job: Job) -> None:
    llm = _make_llm()
    try:
        job.stage = "guarding"
        guard = create_task_guard_agent(llm, CATEGORIES_PATH)
        builder = create_task_builder_agent(llm)

        classification = await guard.arun(job.request)
        job.accepted = classification.accepted
        job.reason = classification.reason
        if not classification.accepted:
            job.stage = "refused"
            return

        job.stage = "building"
        job.domain_hint = classification.domain_hint
        config = await builder.arun(job.request, classification.domain_hint)

        job.stage = "scaffolding"
        problems_root = REPO / "problems"
        _validate_problem_name(config.name, problems_root)
        target_dir = problems_root / config.name
        ProblemLayout.scaffold(target_dir, config, problem_type="programs")
        ProblemContext(target_dir).validate(add_context=config.add_context)

        job.problem_name = config.name
        job.problem_dir = str(target_dir)
        job.config_summary = {
            "description": config.description,
            "metrics": list(config.metrics.keys()),
            "initial_programs": [p.name for p in config.initial_programs],
        }

        job.stage = "writing_code"
        await _write_real_code(config, target_dir)

        job.stage = "seeding"
        db = _next_redis_db()
        job.redis_db = db
        import redis as redis_lib

        redis_lib.Redis(db=db).flushdb()

        log_path = LOGS_DIR / f"{job.id}.log"
        cmd = [
            sys.executable, str(REPO / "run.py"),
            f"problem.name={job.problem_name}",
            f"max_mutants={MAX_MUTANTS}",
            "llm=summer_school_servers",
            f"redis.db={db}",
            "redis.resume=true",
            "+cost_monitor=enabled",
            "stall_watchdog.stall_timeout_s=120",
        ]
        with open(log_path, "w") as lf:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=REPO, stdout=lf, stderr=asyncio.subprocess.STDOUT
            )
        job._proc = proc
        await asyncio.wait_for(_tail_seed_run(job, log_path), timeout=SEED_TIMEOUT_S)
    except TimeoutError:
        job.stage = "error"
        job.error = f"Seed run exceeded {SEED_TIMEOUT_S}s"
        if job._proc:
            job._proc.kill()
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        job.stage = "error"
        job.error = str(exc)


class CreateJobRequest(BaseModel):
    request: str


app = FastAPI(title="gigaevo task-builder demo")


@app.post("/api/jobs")
async def create_job(body: CreateJobRequest) -> dict:
    if not body.request.strip():
        raise HTTPException(400, "request must not be empty")
    job = Job(id=uuid.uuid4().hex[:12], request=body.request.strip())
    _jobs[job.id] = job
    asyncio.create_task(_run_job(job))
    return {"job_id": job.id}


@app.get("/api/jobs")
async def list_jobs() -> list[dict]:
    return [
        {"id": j.id, "request": j.request, "stage": j.stage, "created_at": j.created_at}
        for j in sorted(_jobs.values(), key=lambda j: -j.created_at)
    ]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    return job.public_dict()


@app.post("/api/jobs/{job_id}/abort")
async def abort_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    if job._proc and job._proc.returncode is None:
        job._proc.kill()
        job.stage = "error"
        job.error = "Aborted by user"
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download")
async def download_problem(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if job is None or not job.problem_dir:
        raise HTTPException(404, "no scaffolded problem for this job")
    archive_base = LOGS_DIR / f"{job_id}_problem"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=job.problem_dir)
    return FileResponse(archive_path, filename=f"{job.problem_name}.zip")


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8090)
