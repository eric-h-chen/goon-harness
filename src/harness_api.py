"""The contract every harness must satisfy to be evaluated (specs/eval.md §4.1).

A harness is a Python file exposing ``make_agent() -> Agent``. This module is the
only thing a harness needs to import from this repo; it deliberately does not
depend on Inspect solvers or tasks. Message/tool types come from
``inspect_ai.model`` and ``inspect_ai.tool``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolInfo

__all__ = [
    "Agent",
    "ChatMessage",
    "Env",
    "ExecResult",
    "Feedback",
    "GenerateConfig",
    "LLM",
    "LimitExceeded",
    "Limits",
    "ModelOutput",
    "TaskSpec",
    "ToolInfo",
]


class LimitExceeded(Exception):
    """Raised by ``Env``/``LLM`` once a task limit (LLM calls, attempts, time) is used up.

    The agent may catch it to wrap up, but every further call will raise again.
    The task is graded on the last submission.
    """


@dataclass(frozen=True)
class Limits:
    max_llm_calls: int
    max_attempts: int
    wall_clock_s: int


@dataclass(frozen=True)
class TaskSpec:
    kind: Literal["swe", "lcb"]
    prompt: str  # full problem statement, incl. any benchmark instructions
    workdir: str  # path inside the sandbox (repo root for swe; scratch dir for lcb)
    starter_code: str | None  # lcb only: function stub / LeetCode starter code
    tests_visible: bool  # True on impossible suites, False on real suites
    limits: Limits


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Feedback:
    passed: bool | None  # None on real suites, where hidden-test results are not revealed
    details: str
    attempts_left: int


class Env(Protocol):
    """Sandbox for one task. Relative paths resolve against ``TaskSpec.workdir``."""

    attempts_left: int

    async def exec(self, cmd: str, timeout: int = 180) -> ExecResult: ...
    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...

    async def submit(self, answer: str | None = None) -> Feedback:
        """Submit for grading.

        swe: ``answer`` is ignored; the repo state in ``workdir`` is snapshotted.
        lcb: ``answer`` is the program source (a fenced code block is also accepted).
        The task is graded on the last submission; never submitting is a fail.
        """
        ...


class LLM(Protocol):
    """The fixed model under evaluation. The eval decides which model this is."""

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolInfo] | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput: ...


@runtime_checkable
class Agent(Protocol):
    name: str

    async def solve(self, task: TaskSpec, env: Env, llm: LLM) -> None: ...
