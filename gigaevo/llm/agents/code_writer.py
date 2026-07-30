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
import re
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


_GIGAEVO_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(gigaevo(?:\.\w+)*)", re.MULTILINE)


def _new_gigaevo_import(stub_code: str, code: str) -> str | None:
    """Return a hallucinated ``gigaevo.*`` import the model added that
    wasn't already in the stub, or None.

    The exec sandbox for these files does not have the ``gigaevo`` package
    importable, and there is no shared utils library to draw extra
    functions from -- confirmed live: the model invented
    ``from gigaevo.problems.types.programs.utils import get_test_case,
    get_ground_truth`` (neither function exists), crashing with
    ModuleNotFoundError before entrypoint()/build_context() ever ran. The
    system prompt already says not to do this; this is the deterministic
    backstop for when it does anyway.
    """
    allowed = set(_GIGAEVO_IMPORT_RE.findall(stub_code))
    for found in _GIGAEVO_IMPORT_RE.findall(code):
        if found not in allowed:
            return found
    return None


def _looks_like_unimplemented_stub(code: str) -> bool:
    """Heuristic: the model echoed the stub back instead of implementing it.

    Any leftover ``# TODO:`` marker means a real implementation was never
    written for that spot -- every jinja stub template
    (initial_program/validate/context) leaves at least one such marker,
    and a genuine implementation has no reason to keep it. Also flags a
    context.py that still returns a bare empty dict, the other stub
    signature (no TODO text left, but no real data either) confirmed live.
    """
    if "# TODO:" in code or "# TODO :" in code:
        return True
    if "def build_context" in code and re.search(r"return\s*\{\s*\}", code):
        return True
    return False


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
            try:
                final_state = await self.graph.ainvoke(initial_state)
            except Exception as exc:  # noqa: BLE001 - LLM/parsing boundary, retry-able
                # E.g. openai.LengthFinishReasonError: the response was cut
                # off mid-JSON by max_tokens -- confirmed live on a verbose
                # validate.py response. Any failure here is a transient LLM
                # call, not a config bug, so retry with the same treatment
                # as a bad response rather than crashing the whole pipeline.
                last_problem = f"{type(exc).__name__}: {exc}"
                retry_note = (
                    f"{last_problem}\nKeep the implementation concise (fewer "
                    "comments, no restating the docstring) so the full file "
                    "fits in the response."
                )
                logger.warning(
                    "[CodeWriterAgent] attempt {}/{} for {} ({}): {}",
                    attempt + 1, max_attempts, file_kind, problem_name, last_problem,
                )
                continue
            code = final_state["code"]

            parseable = _best_parseable(code)
            if parseable is None:
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    last_problem = f"Response was not valid Python: {e}"
                else:
                    last_problem = "Response was not valid Python (SyntaxError)."
                retry_note = last_problem
                logger.warning(
                    "[CodeWriterAgent] attempt {}/{} for {} ({}): {}\n--- raw response ---\n{}",
                    attempt + 1, max_attempts, file_kind, problem_name, last_problem, code,
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
            bad_import = _new_gigaevo_import(stub_code, parseable)
            if bad_import is not None:
                last_problem = (
                    f"Response added `import {bad_import}` which was not in "
                    "the stub -- that module/those names are not real and "
                    "the sandbox cannot import gigaevo.* at all."
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
