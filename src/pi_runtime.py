"""Trusted bridge from actual Pi agent-core to the evaluator's Env and LLM.

Candidate files may configure PiAgent; the runtime and RPC implementation remain
fixed. Every solve gets a fresh Node process with no inherited API credentials.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall, ToolInfo, ToolParams

from harness_api import LimitExceeded

RUNTIME = Path(__file__).resolve().parents[1] / "runtimes" / "pi"


def preflight():
    node = shutil.which("node")
    if not node or not (RUNTIME / "node_modules/@earendil-works/pi-agent-core/package.json").exists():
        raise RuntimeError("Pi requires Node >=22 and `npm ci --ignore-scripts --prefix runtimes/pi`")
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if int(version.lstrip("v").split(".")[0]) < 22:
        raise RuntimeError("Pi requires Node >=22")
    expected = json.loads((RUNTIME / "package.json").read_text())["dependencies"]
    for package, pinned in expected.items():
        installed = json.loads((RUNTIME / "node_modules" / package / "package.json").read_text())["version"]
        if installed != pinned:
            raise RuntimeError(f"Pi dependency {package} differs from its pin; run npm ci")
    return {"node": version, "lock_sha256": hashlib.sha256((RUNTIME / "package-lock.json").read_bytes()).hexdigest(),
            "bridge_sha256": hashlib.sha256((RUNTIME / "bridge.mjs").read_bytes()).hexdigest(),
            "python_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


def inspect_messages(messages):
    """Pi's normal transcript → Inspect, preserving original assistant metadata.

    Opaque provider reasoning/signatures stay in _inspect on each Pi assistant
    message and are restored verbatim rather than flattened into text.
    """
    result = []
    for message in messages:
        role = message["role"]
        if role == "assistant":
            if "_inspect" in message:
                result.append(ChatMessageAssistant.model_validate(message["_inspect"]))
            else:
                result.append(ChatMessageAssistant(content=_text(message["content"]), tool_calls=[
                    ToolCall(id=b["id"], function=b["name"], arguments=b["arguments"])
                    for b in message["content"] if b["type"] == "toolCall"]))
        elif role == "system":
            result.append(ChatMessageSystem(content=_text(message["content"])))
        elif role == "user":
            result.append(ChatMessageUser(content=_text(message["content"])))
        elif role == "toolResult":
            result.append(ChatMessageTool(content=_text(message["content"]), tool_call_id=message["toolCallId"],
                                          function=message["toolName"]))
        else:
            raise ValueError(f"Unsupported Pi message role: {role}")
    return result


def pi_message(output):
    message = output.message
    content = [{"type": "text", "text": message.text}] if message.text else []
    content += [{"type": "toolCall", "id": c.id, "name": c.function, "arguments": c.arguments}
                for c in message.tool_calls or []]
    return {"role": "assistant", "content": content, "api": "inspect", "provider": "inspect",
            "model": "inspect-solver", "stopReason": "toolUse" if message.tool_calls else "stop",
            "timestamp": int(time.time() * 1000), "_inspect": message.model_dump(mode="json"),
            # Inspect owns real usage accounting; Pi never independently bills a provider.
            "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}}}


def tool_specs(kind, enabled):
    def tool(name, description, **params):
        return {"name": name, "description": description, "parameters": {
            "type": "object", "properties": {k: {"type": "string"} for k in params},
            "required": list(params), "additionalProperties": False}}
    specs = [tool("bash", "Execute a command in the task working directory.", command=""),
             tool("read_file", "Read a task file.", path=""),
             tool("write_file", "Create or overwrite a task file.", path="", content="")]
    specs = [s for s in specs if s["name"] in enabled]
    specs.append(tool("submit", "Submit for grading.", **({"code": ""} if kind == "lcb" else {})))
    return specs


class PiAgent:
    """Configurable Pi agent-core harness; model/tool transports are evaluator-owned."""
    def __init__(self, system_prompt="You are an expert software engineer. Solve the task using the tools.",
                 name="pi-agent-core", tools=("bash", "read_file", "write_file"),
                 tool_output_chars=20000, bash_timeout=180,
                 continue_prompt="Continue working. Call submit when your solution is ready."):
        if not set(tools) <= {"bash", "read_file", "write_file"}:
            raise ValueError("Unknown Pi tool")
        if not 1 <= tool_output_chars <= 1_000_000 or not 1 <= bash_timeout <= 180:
            raise ValueError("Invalid Pi tool output or command timeout limit")
        self.name, self.system_prompt, self.tools = name, system_prompt, tuple(tools)
        self.tool_output_chars, self.bash_timeout = tool_output_chars, bash_timeout
        self.continue_prompt = continue_prompt
        self.runtime_info = preflight()

    async def _tool(self, name, args, env, task):
        if name not in (*self.tools, "submit"):
            raise ValueError(f"Unknown Pi tool {name}")
        done = False
        if name == "bash":
            result = json.dumps(asdict(await env.exec(args["command"], timeout=self.bash_timeout)))
        elif name == "read_file":
            result = await env.read_file(args["path"])
        elif name == "write_file":
            await env.write_file(args["path"], args["content"])
            result = f"Wrote {args['path']}"
        else:
            feedback = await env.submit(args["code"] if task.kind == "lcb" else None)
            result = json.dumps(asdict(feedback))
            done = feedback.attempts_left == 0
        return {"text": result[:self.tool_output_chars], "done": done}

    async def solve(self, task, env, llm):
        specs = tool_specs(task.kind, self.tools)
        tools = [ToolInfo(name=s["name"], description=s["description"],
                          parameters=ToolParams.model_validate(s["parameters"])) for s in specs]
        node = shutil.which("node")
        if not node:
            raise RuntimeError("Pi requires Node >=22; see README")
        with tempfile.TemporaryDirectory(prefix="goon-pi-") as scratch:
            # The trusted bridge receives no credentials, user settings or workspace
            # environment. All task operations are RPCs, never Node filesystem tools.
            with tempfile.TemporaryFile() as stderr:
                process = await asyncio.create_subprocess_exec(
                    node, str(RUNTIME / "bridge.mjs"), cwd=scratch,
                    env={"PATH": os.environ.get("PATH", ""), "HOME": scratch, "LANG": "C.UTF-8"},
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr,
                    limit=32 * 1024 * 1024)
                async def send(message):
                    process.stdin.write((json.dumps(message) + "\n").encode())
                    await process.stdin.drain()
                try:
                    note = ("Submit complete Python code with submit(code)." if task.kind == "lcb"
                            else "Modify the repository and call submit() to submit its current state.")
                    await send({"type": "start", "tools": specs,
                                "system_prompt": self.system_prompt + f"\nWorking directory: {task.workdir}. "
                                + note + f" You have {task.limits.max_attempts} submission attempts.",
                                "prompt": task.prompt, "continue_prompt": self.continue_prompt})
                    while line := await process.stdout.readline():
                        request = json.loads(line)
                        if request["type"] == "done":
                            break
                        if request["type"] == "fatal":
                            raise RuntimeError("Pi runtime: " + request["error"])
                        if request["type"] != "request":
                            raise RuntimeError("Invalid Pi bridge message")
                        method, params = request["method"], request["params"]
                        if method == "generate":
                            result = pi_message(await llm.generate(inspect_messages(params["messages"]), tools=tools))
                        elif method == "tool":
                            try:
                                result = await self._tool(params["name"], params["arguments"], env, task)
                            except (KeyError, ValueError, FileNotFoundError, IsADirectoryError,
                                    PermissionError, UnicodeDecodeError) as exc:
                                result = {"text": str(exc), "done": False, "isError": True}
                        else:
                            raise RuntimeError(f"Invalid Pi bridge method {method}")
                        await send({"id": request["id"], "result": result})
                    await asyncio.wait_for(process.wait(), 10)
                    if process.returncode:
                        stderr.seek(0)
                        raise RuntimeError("Pi exited: " + stderr.read()[-4000:].decode(errors="replace"))
                except LimitExceeded:
                    return  # Same final-submission semantics as the Python baseline.
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
