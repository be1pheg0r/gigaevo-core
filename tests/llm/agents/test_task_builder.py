"""Tests for TaskBuilderAgent — request -> ProblemConfig generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gigaevo.llm.agents.task_builder import TaskBuilderAgent
from gigaevo.problems.config import (
    FunctionSignature,
    ParameterSpec,
    ProblemConfig,
    ReturnSpec,
    TaskDescription,
)
from gigaevo.programs.metrics.context import MetricSpec


def _mock_llm():
    m = MagicMock()
    m.with_structured_output.return_value = m
    return m


def _make_agent() -> TaskBuilderAgent:
    return TaskBuilderAgent(
        llm=_mock_llm(),
        system_prompt="sys",
        user_prompt_template="Domain: {domain_hint}\nRequest: {request}",
    )


def _valid_config(**overrides) -> ProblemConfig:
    kwargs = dict(
        name="sort_faster",
        description="Sort an array faster",
        entrypoint=FunctionSignature(
            params=[ParameterSpec(name="data", type_hint="np.ndarray")],
            returns=ReturnSpec(type_hint="np.ndarray"),
        ),
        validation=FunctionSignature(params=[ParameterSpec(name="solution")]),
        metrics={
            "fitness": MetricSpec(
                description="speed", is_primary=True, higher_is_better=True,
                lower_bound=0.0, upper_bound=1.0,
            )
        },
        task_description=TaskDescription(objective="Sort faster than baseline"),
    )
    kwargs.update(overrides)
    return ProblemConfig(**kwargs)


class TestBuildPrompt:
    def test_domain_hint_and_request_rendered(self):
        agent = _make_agent()
        state = {
            "request": "evolve a faster sort",
            "domain_hint": "algorithm_speed",
            "messages": [],
            "llm_response": None,
            "config": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        user_text = result["messages"][1].content
        assert "algorithm_speed" in user_text
        assert "evolve a faster sort" in user_text

    def test_missing_domain_hint_falls_back_to_placeholder(self):
        agent = _make_agent()
        state = {
            "request": "evolve a faster sort",
            "domain_hint": "",
            "messages": [],
            "llm_response": None,
            "config": None,
            "metadata": {},
        }
        # build_prompt reads state["domain_hint"] directly -- the "unspecified"
        # fallback lives in arun(), not build_prompt, so this just confirms
        # an empty string doesn't crash formatting.
        result = agent.build_prompt(state)
        assert "Domain: " in result["messages"][1].content


class TestParseResponse:
    def test_valid_config_passed_through(self):
        agent = _make_agent()
        cfg = _valid_config()
        state = {
            "request": "x",
            "domain_hint": "y",
            "messages": [],
            "llm_response": cfg,
            "config": None,
            "metadata": {},
        }
        result = agent.parse_response(state)
        assert result["config"] is cfg

    def test_wrong_response_type_raises(self):
        agent = _make_agent()
        state = {
            "request": "x",
            "domain_hint": "y",
            "messages": [],
            "llm_response": "not a ProblemConfig",
            "config": None,
            "metadata": {},
        }
        with pytest.raises(ValueError, match="Expected ProblemConfig"):
            agent.parse_response(state)
