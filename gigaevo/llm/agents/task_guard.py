"""Guard-rail agent: decide whether a free-text request is a legitimate
evolutionary task request, and whether it matches a forbidden category.

This is a deny-list, not an allow-list: any request describing a scorable
optimization/search task is accepted UNLESS it matches one of the
forbidden categories loaded from config/task_builder/categories.yaml. This
is the first stage of the request -> gigaevo-problem pipeline (see
gigaevo/problems/task_creator.py). Nothing downstream runs unless this
agent accepts the request.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator
import yaml

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter


class ForbiddenCategory(BaseModel):
    """One deny-list entry, as loaded from categories.yaml."""

    id: str
    description: str
    keywords: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_lowercase(cls, v: str) -> str:
        return v.strip().lower()


class RequestClassification(BaseModel):
    """Structured guard-rail decision."""

    is_task_request: bool = Field(
        description="True only if the request describes a scorable optimization/search task."
    )
    prohibited: bool = Field(
        default=False,
        description="True if the request matches any forbidden category, regardless of is_task_request.",
    )
    prohibited_category: str | None = Field(
        default=None,
        description="Which forbidden category id it matches, if prohibited=True.",
    )
    domain_hint: str | None = Field(
        default=None,
        description=(
            "Free-text domain label for framing the task description "
            "(e.g. 'control systems', 'NLP reasoning chain') — not a "
            "restricted category, purely descriptive."
        ),
    )
    reason: str = Field(description="One-sentence explanation of the decision, for the user.")

    @field_validator("prohibited_category")
    @classmethod
    def _normalize_category(cls, v: str | None) -> str | None:
        return v.strip().lower() if v else v

    @property
    def accepted(self) -> bool:
        return self.is_task_request and not self.prohibited


def load_forbidden_categories(path: str | Path) -> list[ForbiddenCategory]:
    """Load the deny-list from config/task_builder/categories.yaml."""
    data = yaml.safe_load(Path(path).read_text())
    entries = (data or {}).get("forbidden_categories")
    if not entries:
        raise ValueError(f"{path} has no 'forbidden_categories' entries")
    return [ForbiddenCategory(**e) for e in entries]


def _keyword_backstop_hit(
    request: str, forbidden: list[ForbiddenCategory]
) -> ForbiddenCategory | None:
    """Deterministic pre-check, independent of the LLM's own judgment.

    A confused or adversarially-prompted classifier can still miss an
    obvious match; this regex pass runs regardless of what the model says
    and can only push a decision toward *more* restrictive, never less.
    """
    lowered = request.lower()
    for cat in forbidden:
        for kw in cat.keywords:
            if re.search(re.escape(kw.lower()), lowered):
                return cat
    return None


class TaskGuardState(TypedDict):
    request: str
    messages: list[BaseMessage]
    llm_response: AIMessage | RequestClassification | None
    classification: RequestClassification | None
    metadata: dict


class TaskGuardAgent(LangGraphAgent):
    """Classifies a free-text request: legitimate task request, and whether
    it matches a forbidden category.

    Two independent layers, either can block: the keyword backstop (regex,
    no LLM involved) runs first in ``arun``; the LLM's own judgment runs
    second. Both must clear for a request to be accepted — the keyword
    backstop can force a rejection the LLM missed, but the LLM's more
    permissive judgment can never override a keyword hit.
    """

    StateSchema = TaskGuardState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
        forbidden_categories: list[ForbiddenCategory],
    ):
        if not forbidden_categories:
            raise ValueError("forbidden_categories must be non-empty")
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.forbidden_categories = forbidden_categories
        self._forbidden_ids = {c.id for c in forbidden_categories}
        structured_llm = llm.with_structured_output(RequestClassification)
        super().__init__(structured_llm)

    def build_prompt(self, state: TaskGuardState) -> TaskGuardState:
        categories_block = "\n".join(
            f"- {c.id}: {c.description}" for c in self.forbidden_categories
        )
        user_prompt = self.user_prompt_template.format(
            request=state["request"], categories=categories_block
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return state

    def parse_response(self, state: TaskGuardState) -> TaskGuardState:
        resp = state["llm_response"]
        if not isinstance(resp, RequestClassification):
            raise ValueError(f"Expected RequestClassification, got {type(resp)}")

        # If the model claims a match outside the configured deny-list,
        # it's not a valid category id — still trust that it wanted to
        # block (conservative), just drop the unrecognized label.
        if resp.prohibited and resp.prohibited_category not in self._forbidden_ids:
            resp = resp.model_copy(update={"prohibited_category": None})

        state["classification"] = resp
        return state

    async def arun(self, request: str) -> RequestClassification:
        backstop_hit = _keyword_backstop_hit(request, self.forbidden_categories)
        if backstop_hit is not None:
            return RequestClassification(
                is_task_request=False,
                prohibited=True,
                prohibited_category=backstop_hit.id,
                reason=f"Keyword guard matched forbidden category '{backstop_hit.id}'.",
            )

        initial_state: TaskGuardState = {
            "request": request,
            "messages": [],
            "llm_response": None,
            "classification": None,
            "metadata": {},
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["classification"]
