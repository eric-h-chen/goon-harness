"""Seed harness: a minimal ReAct agent written against harness_api (specs/eval.md §4.1).

Tools: bash, write_file, submit. Used as iteration 0 unless the loop supplies its own seed.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall, ToolInfo, ToolParam, ToolParams

from harness_api import Env, LimitExceeded, LLM, TaskSpec

MAX_TOOL_OUTPUT_CHARS = 20_000

SYSTEM = """You are an expert software engineer working on a Linux machine through tools.
Use `bash` to explore and run code, and `write_file` to create or overwrite files.
Your working directory is {workdir}.
When you are finished, call `submit`. {submit_note}
You can submit up to {attempts} time(s).{feedback_note}"""

SUBMIT_NOTE = {
    "lcb": "Pass your complete Python solution as the `code` argument.",
    "swe": "Your changes to the repository are what gets graded.",
}


def _tool(name: str, description: str, **params: str) -> ToolInfo:
    return ToolInfo(
        name=name,
        description=description,
        parameters=ToolParams(
            properties={k: ToolParam(type="string", description=v) for k, v in params.items()},
            required=list(params),
        ),
    )


BASH = _tool("bash", "Run a bash command and return its output.", command="The command to run.")
WRITE = _tool(
    "write_file", "Create or overwrite a file.", path="File path.", content="Full file contents."
)


class BaselineAgent:
    name = "baseline-react"

    async def solve(self, task: TaskSpec, env: Env, llm: LLM) -> None:
        submit = (
            _tool("submit", "Submit your final solution.", code="Complete Python solution.")
            if task.kind == "lcb"
            else _tool("submit", "Submit the current state of the repository.")
        )
        system = SYSTEM.format(
            workdir=task.workdir,
            submit_note=SUBMIT_NOTE[task.kind],
            attempts=task.limits.max_attempts,
            feedback_note=" After each submission you will see the test results." if task.tests_visible else "",
        )
        messages = [ChatMessageSystem(content=system), ChatMessageUser(content=task.prompt)]
        try:
            while True:
                out = await llm.generate(messages, tools=[BASH, WRITE, submit])
                messages.append(out.message)
                calls = out.message.tool_calls or []
                if not calls:
                    messages.append(ChatMessageUser(content="Continue. Call `submit` when you are done."))
                    continue
                for call in calls:
                    result, done = await self._run(call, task, env)
                    messages.append(ChatMessageTool(
                        content=result[:MAX_TOOL_OUTPUT_CHARS], tool_call_id=call.id, function=call.function
                    ))
                    if done:
                        return
        except LimitExceeded:
            return

    async def _run(self, call: ToolCall, task: TaskSpec, env: Env) -> tuple[str, bool]:
        args = call.arguments or {}
        try:
            if call.function == "bash":
                r = await env.exec(args.get("command", ""))
                return f"exit code {r.returncode}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}", False
            if call.function == "write_file":
                await env.write_file(args["path"], args["content"])
                return f"Wrote {args['path']}.", False
            if call.function == "submit":
                fb = await env.submit(args.get("code") if task.kind == "lcb" else None)
                status = {True: "PASSED", False: "FAILED", None: "received"}[fb.passed]
                return f"Submission {status}. Attempts left: {fb.attempts_left}.\n{fb.details}", fb.attempts_left == 0
            return f"Unknown tool {call.function!r}.", False
        except (KeyError, ValueError, FileNotFoundError, IsADirectoryError, PermissionError,
                UnicodeDecodeError) as e:
            return f"Error: {type(e).__name__}: {e}", False


def make_agent() -> BaselineAgent:
    return BaselineAgent()
