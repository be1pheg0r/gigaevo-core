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

import ast
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter

# Small/quantized models occasionally double-escape the multi-line `code`
# field of the structured JSON output, producing a single-line file with
# literal backslash-n/backslash-t instead of real newlines/tabs -- valid
# JSON, invalid Python (SyntaxError, confirmed live: a task_builder_web
# validate.py came back as one line and failed ast.parse). Try the raw
# text first (the common, correct case) before assuming escaping.
_ESCAPE_CANDIDATES = (
    lambda s: s,
    lambda s: s.replace("\\n", "\n").replace("\\t", "\t"),
)


def _best_parseable(code: str) -> str | None:
    """Return the first candidate transform of ``code`` that is valid
    Python source, or None if none of them parse."""
    for transform in _ESCAPE_CANDIDATES:
        candidate = transform(code)
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate
    return None


def _looks_like_unimplemented_stub(code: str) -> bool:
    """Heuristic: the model echoed the stub back instead of implementing it.

    A real implementation is never a bare ``pass`` body under the leftover
    ``# TODO: Implement strategy`` marker -- catches the model silently
    declining to write logic (confirmed live: 2 of 3 initial_programs came
    back byte-for-byte the original stub).
    """
    return "# TODO: Implement strategy" in code and "pass" in code.split(
        "# TODO: Implement strategy", 1
    )[1]


class CodeFile(BaseModel):
    code: str = Field(description="Complete Python source for the file")


class CodeWriterState(TypedDict):
    problem_name: str
    task_description: str
    file_kind: str
    file_purpose: str
    stub_code: str
    retry_note: str
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
        if state.get("retry_note"):
            user_prompt += f"\n\nYour previous attempt was rejected:\n{state['retry_note']}\nFix exactly this and return the complete corrected file."
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
        max_attempts: int = 3,
    ) -> str:
        """Return the complete implemented file content.

        Retries (feeding the concrete failure back into the prompt) when
        the response is unparseable Python or is just the stub echoed
        back unimplemented -- both observed live from small/quantized
        models on this pipeline.
        """
        retry_note = ""
        last_problem = "no valid response"
        for attempt in range(max_attempts):
            initial_state: CodeWriterState = {
                "problem_name": problem_name,
                "task_description": task_description,
                "file_kind": file_kind,
                "file_purpose": file_purpose,
                "stub_code": stub_code,
                "retry_note": retry_note,
                "messages": [],
                "llm_response": None,
                "code": None,
                "metadata": {},
            }
            final_state = await self.graph.ainvoke(initial_state)
            code = final_state["code"]

            parseable = _best_parseable(code)
            if parseable is None:
                last_problem = "Response was not valid Python (SyntaxError)."
                retry_note = last_problem
                logger.warning(
                    "[CodeWriterAgent] attempt {}/{} for {} ({}): {}",
                    attempt + 1, max_attempts, file_kind, problem_name, last_problem,
                )
                continue
            if _looks_like_unimplemented_stub(parseable):
                last_problem = (
                    "Response left the body as `pass` under the "
                    "'# TODO: Implement strategy' comment instead of writing "
                    "a real implementation."
                )
                retry_note = last_problem
                logger.warning(
                    "[CodeWriterAgent] attempt {}/{} for {} ({}): {}",
                    attempt + 1, max_attempts, file_kind, problem_name, last_problem,
                )
                continue
            return parseable

        raise ValueError(
            f"[CodeWriterAgent] {file_kind} for '{problem_name}' failed after "
            f"{max_attempts} attempts: {last_problem}"
        )
