"""End-to-end pipeline: natural-language request -> scaffolded gigaevo problem.

guard (TaskGuardAgent, deny-list + keyword backstop)
  -> generation (TaskBuilderAgent -> ProblemConfig)
  -> filesystem-safety validation (this module)
  -> scaffold (ProblemLayout.scaffold)
  -> post-scaffold structural validation (ProblemContext.validate)

Nothing is written to disk unless the guard accepts the request. Every
stage can independently reject; the first rejection short-circuits the
rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from loguru import logger

from gigaevo.llm.agents.task_builder import TaskBuilderAgent
from gigaevo.llm.agents.task_guard import TaskGuardAgent
from gigaevo.problems.config import ProblemConfig
from gigaevo.problems.context import ProblemContext
from gigaevo.problems.layout import ProblemLayout

# Lowercase snake_case, 3-64 chars, must start with a letter. The LLM's
# `name` becomes a directory name — ProblemConfig itself does not check
# this, so it's enforced here before anything touches the filesystem.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


@dataclass
class TaskCreationResult:
    accepted: bool
    reason: str
    problem_dir: Path | None = None
    config: ProblemConfig | None = None
    domain_hint: str | None = None


def _validate_problem_name(name: str, problems_root: Path) -> None:
    """Filesystem-safety guard the generation step doesn't cover.

    Raises ValueError on anything that isn't a safe, non-colliding
    directory name — path separators, ``..``, uppercase, leading digits,
    or an existing problem with the same name.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Problem name '{name}' must be lowercase snake_case, 3-64 chars, "
            "starting with a letter."
        )
    if (problems_root / name).exists():
        raise ValueError(f"Problem directory '{name}' already exists.")


async def create_task_from_request(
    request: str,
    *,
    guard_agent: TaskGuardAgent,
    builder_agent: TaskBuilderAgent,
    problems_root: Path,
) -> TaskCreationResult:
    """Run the full request -> problem-directory pipeline.

    Never raises for a refused/prohibited request — returns
    ``accepted=False`` with a reason instead. Does raise if the accepted
    request produces a config or filesystem state that fails validation,
    since at that point it's a bug worth surfacing, not a normal refusal.
    """
    classification = await guard_agent.arun(request)
    if not classification.accepted:
        logger.info(
            "[task_creator] refused (task_request={}, prohibited={}, category={}): {}",
            classification.is_task_request,
            classification.prohibited,
            classification.prohibited_category,
            classification.reason,
        )
        return TaskCreationResult(accepted=False, reason=classification.reason)

    config = await builder_agent.arun(request, classification.domain_hint)
    _validate_problem_name(config.name, problems_root)

    target_dir = problems_root / config.name
    ProblemLayout.scaffold(target_dir, config, problem_type="programs")

    # Re-validate what actually landed on disk, not just the config object
    # -- catches template-rendering issues the config-level validators
    # can't see.
    ProblemContext(target_dir).validate(add_context=config.add_context)

    logger.info(
        "[task_creator] scaffolded '{}' (domain={})",
        config.name,
        classification.domain_hint,
    )
    return TaskCreationResult(
        accepted=True,
        reason=classification.reason,
        problem_dir=target_dir,
        config=config,
        domain_hint=classification.domain_hint,
    )
