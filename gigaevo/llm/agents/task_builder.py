"""Generation agent: turn an approved request into a scaffoldable
ProblemConfig.

Second stage of the request -> gigaevo-problem pipeline (see
gigaevo/problems/task_creator.py). Only reached after TaskGuardAgent has
accepted the request and picked a category.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.config import ProblemConfig


class TaskBuilderState(TypedDict):
    request: str
    domain_hint: str
    messages: list[BaseMessage]
    llm_response: AIMessage | ProblemConfig | None
    config: ProblemConfig | None
    metadata: dict


class TaskBuilderAgent(LangGraphAgent):
    """Generates a ProblemConfig for an approved request.

    Validation is layered, not a single check: LangChain's structured
    output instantiates ``ProblemConfig`` from the model's JSON, which runs
    every field-level Pydantic validator (types, bounds) AND
    ``ProblemConfig``'s own ``_run_validations`` (exactly-one-primary-metric,
    context signature consistency, helper/context config consistency) —
    a malformed response raises before this method ever sees it.
    """

    StateSchema = TaskBuilderState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        structured_llm = llm.with_structured_output(ProblemConfig)
        super().__init__(structured_llm)

    def build_prompt(self, state: TaskBuilderState) -> TaskBuilderState:
        user_prompt = self.user_prompt_template.format(
            request=state["request"],
            domain_hint=state["domain_hint"] or "unspecified",
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return state

    def parse_response(self, state: TaskBuilderState) -> TaskBuilderState:
        resp = state["llm_response"]
        if not isinstance(resp, ProblemConfig):
            raise ValueError(f"Expected ProblemConfig, got {type(resp)}")
        state["config"] = resp
        return state

    async def arun(self, request: str, domain_hint: str | None = None) -> ProblemConfig:
        initial_state: TaskBuilderState = {
            "request": request,
            "domain_hint": domain_hint or "",
            "messages": [],
            "llm_response": None,
            "config": None,
            "metadata": {},
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["config"]
