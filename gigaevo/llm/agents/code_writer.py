"""Stub-to-implementation agent: fill in a scaffolded problem's TODO stubs
(initial_programs/*.py, validate.py) with a real, runnable implementation.

Third stage of the request -> runnable gigaevo problem pipeline, after
TaskBuilderAgent (see gigaevo/problems/task_creator.py). ProblemLayout's
jinja templates only emit a signature + docstring + `# TODO` body -- a
scaffolded problem is not runnable until this stage fills in real logic
(otherwise every seed program returns ``None``, evolutionary stages
AUTO-SKIP, ``is_valid=0`` always, and the archive never gets a single
accepted mutant to build on).
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter


class CodeFile(BaseModel):
    code: str = Field(description="Complete Python source for the file")


class CodeWriterState(TypedDict):
    problem_name: str
    task_description: str
    file_kind: str
    file_purpose: str
    stub_code: str
    messages: list[BaseMessage]
    llm_response: AIMessage | CodeFile | None
    code: str | None
    metadata: dict


class CodeWriterAgent(LangGraphAgent):
    """Turns one stub file into a real implementation."""

    StateSchema = CodeWriterState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        structured_llm = llm.with_structured_output(CodeFile)
        super().__init__(structured_llm)

    def build_prompt(self, state: CodeWriterState) -> CodeWriterState:
        user_prompt = self.user_prompt_template.format(
            problem_name=state["problem_name"],
            task_description=state["task_description"],
            file_kind=state["file_kind"],
            file_purpose=state["file_purpose"],
            stub_code=state["stub_code"],
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return state

    def parse_response(self, state: CodeWriterState) -> CodeWriterState:
        resp = state["llm_response"]
        if not isinstance(resp, CodeFile):
            raise ValueError(f"Expected CodeFile, got {type(resp)}")
        state["code"] = resp.code
        return state

    async def arun(
        self,
        *,
        problem_name: str,
        task_description: str,
        file_kind: str,
        file_purpose: str,
        stub_code: str,
    ) -> str:
        """Return the complete implemented file content."""
        initial_state: CodeWriterState = {
            "problem_name": problem_name,
            "task_description": task_description,
            "file_kind": file_kind,
            "file_purpose": file_purpose,
            "stub_code": stub_code,
            "messages": [],
            "llm_response": None,
            "code": None,
            "metadata": {},
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["code"]
