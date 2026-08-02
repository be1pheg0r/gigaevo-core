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
from loguru import logger
from pydantic import ValidationError

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.config import ProblemConfig


class TaskBuilderState(TypedDict):
    request: str
    domain_hint: str
    retry_note: str
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
        if state.get("retry_note"):
            user_prompt += (
                "\n\nYour previous attempt was rejected by validation:\n"
                f"{state['retry_note']}\n"
                "Fix exactly this and produce a corrected, complete configuration."
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

    async def arun(
        self, request: str, domain_hint: str | None = None, max_attempts: int = 2
    ) -> ProblemConfig:
        """Generate a ProblemConfig, retrying once with the validation
        error fed back into the prompt.

        Small models frequently produce internally-inconsistent configs
        (e.g. ``add_context=True`` without a matching ``context`` param) —
        ``ProblemConfig``'s own validators catch this reliably, but a hard
        failure on the first miss wastes an otherwise-recoverable attempt.
        """
        retry_note = ""
        last_error: ValidationError | None = None
        for attempt in range(max_attempts):
            initial_state: TaskBuilderState = {
                "request": request,
                "domain_hint": domain_hint or "",
                "retry_note": retry_note,
                "messages": [],
                "llm_response": None,
                "config": None,
                "metadata": {},
            }
            try:
                final_state = await self.graph.ainvoke(initial_state)
                return final_state["config"]
            except ValidationError as exc:
                last_error = exc
                retry_note = str(exc)
                logger.warning(
                    "[TaskBuilderAgent] attempt {}/{} failed validation: {}",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
        raise last_error
