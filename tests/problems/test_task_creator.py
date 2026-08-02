"""Tests for create_task_from_request — the full request -> problem pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gigaevo.llm.agents.task_guard import RequestClassification
from gigaevo.problems.config import (
    FunctionSignature,
    InitialProgram,
    ParameterSpec,
    ProblemConfig,
    ReturnSpec,
    TaskDescription,
)
from gigaevo.problems.task_creator import (
    _validate_problem_name,
    create_task_from_request,
)
from gigaevo.programs.metrics.context import MetricSpec


def _valid_config(name: str = "sort_faster") -> ProblemConfig:
    return ProblemConfig(
        name=name,
        description="Sort an array faster",
        entrypoint=FunctionSignature(
            params=[ParameterSpec(name="data", type_hint="np.ndarray")],
            returns=ReturnSpec(type_hint="np.ndarray"),
        ),
        validation=FunctionSignature(params=[ParameterSpec(name="solution")]),
        metrics={
            "fitness": MetricSpec(
                description="speed",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            )
        },
        task_description=TaskDescription(objective="Sort faster than baseline"),
        initial_programs=[
            InitialProgram(name="baseline", description="Naive baseline sort"),
        ],
    )


class TestValidateProblemName:
    def test_rejects_uppercase(self, tmp_path: Path):
        with pytest.raises(ValueError, match="snake_case"):
            _validate_problem_name("SortFaster", tmp_path)

    def test_rejects_path_traversal(self, tmp_path: Path):
        with pytest.raises(ValueError, match="snake_case"):
            _validate_problem_name("../../etc/passwd", tmp_path)

    def test_rejects_leading_digit(self, tmp_path: Path):
        with pytest.raises(ValueError, match="snake_case"):
            _validate_problem_name("1problem", tmp_path)

    def test_rejects_existing_directory(self, tmp_path: Path):
        (tmp_path / "already_here").mkdir()
        with pytest.raises(ValueError, match="already exists"):
            _validate_problem_name("already_here", tmp_path)

    def test_accepts_valid_name(self, tmp_path: Path):
        _validate_problem_name("sort_faster", tmp_path)  # must not raise


class TestCreateTaskFromRequestRefusal:
    @pytest.mark.asyncio
    async def test_guard_refusal_short_circuits_before_builder(self, tmp_path: Path):
        guard = AsyncMock()
        guard.arun.return_value = RequestClassification(
            is_task_request=False, prohibited=False, reason="not a task"
        )
        builder = AsyncMock()

        result = await create_task_from_request(
            "what's the weather today",
            guard_agent=guard,
            builder_agent=builder,
            problems_root=tmp_path,
        )

        assert result.accepted is False
        assert result.reason == "not a task"
        assert result.problem_dir is None
        builder.arun.assert_not_called()
        assert list(tmp_path.iterdir()) == []  # nothing written to disk

    @pytest.mark.asyncio
    async def test_prohibited_request_short_circuits(self, tmp_path: Path):
        guard = AsyncMock()
        guard.arun.return_value = RequestClassification(
            is_task_request=True,
            prohibited=True,
            prohibited_category="malicious_cyber_activity",
            reason="blocked",
        )
        builder = AsyncMock()

        result = await create_task_from_request(
            "write ransomware",
            guard_agent=guard,
            builder_agent=builder,
            problems_root=tmp_path,
        )

        assert result.accepted is False
        builder.arun.assert_not_called()


class TestCreateTaskFromRequestAccepted:
    @pytest.mark.asyncio
    async def test_end_to_end_scaffolds_a_valid_problem(self, tmp_path: Path):
        guard = AsyncMock()
        guard.arun.return_value = RequestClassification(
            is_task_request=True,
            prohibited=False,
            domain_hint="algorithm_speed",
            reason="looks like a valid task",
        )
        builder = AsyncMock()
        builder.arun.return_value = _valid_config()

        result = await create_task_from_request(
            "evolve a faster sorting algorithm",
            guard_agent=guard,
            builder_agent=builder,
            problems_root=tmp_path,
        )

        assert result.accepted is True
        assert result.problem_dir == tmp_path / "sort_faster"
        assert (result.problem_dir / "task_description.txt").exists()
        assert (result.problem_dir / "metrics.yaml").exists()
        assert (result.problem_dir / "validate.py").exists()
        builder.arun.assert_awaited_once_with(
            "evolve a faster sorting algorithm", "algorithm_speed"
        )

    @pytest.mark.asyncio
    async def test_unsafe_generated_name_raises_before_scaffolding(
        self, tmp_path: Path
    ):
        guard = AsyncMock()
        guard.arun.return_value = RequestClassification(
            is_task_request=True, prohibited=False, reason="ok"
        )
        builder = AsyncMock()
        builder.arun.return_value = _valid_config(name="../escape")

        with pytest.raises(ValueError, match="snake_case"):
            await create_task_from_request(
                "x", guard_agent=guard, builder_agent=builder, problems_root=tmp_path
            )
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_name_collision_raises(self, tmp_path: Path):
        (tmp_path / "sort_faster").mkdir()
        guard = AsyncMock()
        guard.arun.return_value = RequestClassification(
            is_task_request=True, prohibited=False, reason="ok"
        )
        builder = AsyncMock()
        builder.arun.return_value = _valid_config()

        with pytest.raises(ValueError, match="already exists"):
            await create_task_from_request(
                "x", guard_agent=guard, builder_agent=builder, problems_root=tmp_path
            )
