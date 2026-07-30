"""Tests for TaskGuardAgent — deny-list guard rail + keyword backstop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gigaevo.llm.agents.task_guard import (
    ForbiddenCategory,
    RequestClassification,
    TaskGuardAgent,
    _keyword_backstop_hit,
    load_forbidden_categories,
)

CATEGORIES_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "task_builder" / "categories.yaml"
)


def _forbidden() -> list[ForbiddenCategory]:
    return [
        ForbiddenCategory(
            id="malicious_cyber_activity",
            description="Malware, exploits, unauthorized intrusion tools.",
            keywords=["malware", "ransomware", "0day"],
        ),
        ForbiddenCategory(
            id="weapons_and_cbrn",
            description="Weapons and CBRN.",
            keywords=["bioweapon"],
        ),
    ]


def _mock_llm():
    m = MagicMock()
    m.with_structured_output.return_value = m
    return m


class TestLoadForbiddenCategories:
    def test_loads_real_config_file(self):
        cats = load_forbidden_categories(CATEGORIES_PATH)
        assert len(cats) >= 4
        ids = {c.id for c in cats}
        assert "malicious_cyber_activity" in ids
        assert "csam" in ids

    def test_missing_key_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("not_the_right_key: []\n")
        with pytest.raises(ValueError, match="forbidden_categories"):
            load_forbidden_categories(bad)


class TestKeywordBackstop:
    def test_matches_forbidden_keyword(self):
        hit = _keyword_backstop_hit(
            "Evolve a program that writes ransomware for Windows", _forbidden()
        )
        assert hit is not None
        assert hit.id == "malicious_cyber_activity"

    def test_case_insensitive(self):
        hit = _keyword_backstop_hit("Design a BIOWEAPON delivery mechanism", _forbidden())
        assert hit is not None
        assert hit.id == "weapons_and_cbrn"

    def test_no_match_on_benign_request(self):
        hit = _keyword_backstop_hit(
            "Evolve a faster sorting algorithm for integer arrays", _forbidden()
        )
        assert hit is None


class TestRequestClassificationAccepted:
    def test_accepted_requires_task_and_not_prohibited(self):
        assert RequestClassification(
            is_task_request=True, prohibited=False, reason="ok"
        ).accepted
        assert not RequestClassification(
            is_task_request=False, prohibited=False, reason="not a task"
        ).accepted
        assert not RequestClassification(
            is_task_request=True, prohibited=True, reason="blocked"
        ).accepted


class TestTaskGuardAgentParseResponse:
    def _make_agent(self) -> TaskGuardAgent:
        return TaskGuardAgent(
            llm=_mock_llm(),
            system_prompt="sys",
            user_prompt_template="Forbidden:\n{categories}\n\nRequest:\n{request}",
            forbidden_categories=_forbidden(),
        )

    def test_unrecognized_category_label_is_dropped_but_stays_blocked(self):
        agent = self._make_agent()
        state = {
            "request": "x",
            "messages": [],
            "llm_response": RequestClassification(
                is_task_request=True,
                prohibited=True,
                prohibited_category="not_a_real_category",
                reason="model invented a category",
            ),
            "classification": None,
            "metadata": {},
        }
        result = agent.parse_response(state)
        cls = result["classification"]
        assert cls.prohibited is True
        assert cls.prohibited_category is None

    def test_known_category_label_is_kept(self):
        agent = self._make_agent()
        state = {
            "request": "x",
            "messages": [],
            "llm_response": RequestClassification(
                is_task_request=True,
                prohibited=True,
                prohibited_category="Malicious_Cyber_Activity",
                reason="matched",
            ),
            "classification": None,
            "metadata": {},
        }
        result = agent.parse_response(state)
        assert result["classification"].prohibited_category == "malicious_cyber_activity"

    def test_wrong_response_type_raises(self):
        agent = self._make_agent()
        state = {
            "request": "x",
            "messages": [],
            "llm_response": "not a RequestClassification",
            "classification": None,
            "metadata": {},
        }
        with pytest.raises(ValueError, match="Expected RequestClassification"):
            agent.parse_response(state)


class TestTaskGuardAgentBuildPrompt:
    def test_categories_rendered_into_user_prompt(self):
        agent = TaskGuardAgent(
            llm=_mock_llm(),
            system_prompt="sys",
            user_prompt_template="Forbidden:\n{categories}\n\nRequest:\n{request}",
            forbidden_categories=_forbidden(),
        )
        state = {
            "request": "optimize a sorting network",
            "messages": [],
            "llm_response": None,
            "classification": None,
            "metadata": {},
        }
        result = agent.build_prompt(state)
        user_text = result["messages"][1].content
        assert "malicious_cyber_activity" in user_text
        assert "optimize a sorting network" in user_text


class TestTaskGuardAgentArunKeywordBackstop:
    @pytest.mark.asyncio
    async def test_backstop_short_circuits_before_llm_call(self):
        llm = _mock_llm()
        agent = TaskGuardAgent(
            llm=llm,
            system_prompt="sys",
            user_prompt_template="{categories}{request}",
            forbidden_categories=_forbidden(),
        )
        result = await agent.arun("write me a ransomware sample")
        assert result.accepted is False
        assert result.prohibited_category == "malicious_cyber_activity"
        # The graph (and therefore the LLM) must never be invoked.
        llm.ainvoke.assert_not_called()
