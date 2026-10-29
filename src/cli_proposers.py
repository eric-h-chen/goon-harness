"""Claude Code / Codex proposers in disposable Docker containers.

Only a regular-file snapshot of training history and a scratch workspace are
mounted. No checkout, evaluator results, home directory, or Docker socket enters
these containers. Provider credentials are passed by environment name, never argv.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from train import write_json

DEFAULT_IMAGE = "goon-proposers:local"
SCHEMA = {"type": "object", "properties": {"source": {"type": "string"},
           "reflection": {"type": "string"}}, "required": ["source", "reflection"],
          "additionalProperties": False}


def credentials(backend):
    if backend == "codex":
        key = os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Codex proposer requires CODEX_API_KEY or OPENAI_API_KEY")
        return {"CODEX_API_KEY": key}
    if backend == "claude-code":
        for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
            if os.environ.get(name):
                return {name: os.environ[name]}
        raise ValueError("Claude Code proposer requires ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN")
    raise ValueError(f"Unknown CLI proposer {backend}")


def preflight(backend, image):
    credentials(backend)
    if not shutil.which("docker"):
        raise RuntimeError("Docker is required for CLI proposers; see README setup")
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Proposer image unavailable. Build runtimes/proposers/Dockerfile first. "
                           + result.stderr[-1000:])
    return result.stdout.strip()


def snapshot_history(history, destination):
    """Never follow links, including links within history, into mounted snapshots."""
    root = Path(history).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix not in (".py", ".json", ".jsonl", ".txt", ".eval"):
            continue
        if any((root / p).is_symlink() for p in path.relative_to(root).parents if p != Path(".")):
            continue
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def cli_command(backend, model, max_calls):
    if backend == "codex":
        # The external Docker boundary isolates shell execution. The inner CLI
        # sandbox cannot be relied on inside a container (nested sandbox support).
        return ["codex", "exec", "--model", model, "--json", "--ephemeral",
                "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
                "--output-schema", "/workspace/schema.json",
                "--output-last-message", "/workspace/answer.json", "-"]
    if backend == "claude-code":
        return ["claude", "--print", "--model", model, "--output-format", "stream-json",
                "--verbose", "--json-schema", json.dumps(SCHEMA), "--max-turns", str(max_calls),
                "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config",
                "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands",
                "--permission-mode", "dontAsk", "--tools", "Bash,Read,Glob,Grep",
                "--allowedTools", "Bash,Read,Glob,Grep"]
    raise ValueError(f"Unknown CLI proposer {backend}")


def parse_proposal(backend, workspace, stdout):
    if backend == "codex":
        path = workspace / "answer.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError("Codex did not produce a regular, bounded answer.json")
        value = json.loads(path.read_text())
    else:
        result = None
        with stdout.open() as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("type") == "result":
                    result = event
        if not result or result.get("is_error") or result.get("subtype") != "success":
            raise ValueError("Claude Code did not report a successful structured result; inspect proposer logs")
        value = result.get("structured_output")
    if not isinstance(value, dict) or set(value) != {"source", "reflection"}:
        raise ValueError("Proposer output must contain exactly source and reflection")
    if not all(isinstance(value[k], str) and value[k].strip() for k in value):
        raise ValueError("Proposer source and reflection must be nonempty strings")
    return value


async def propose_cli(history, model, prompt, out_dir, *, backend, system_prompt,
                      max_calls=30, timeout=1800, image=DEFAULT_IMAGE):
    auth = credentials(backend)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    attempt = len(list(out.glob("cli_*.json")))
    prefix = out / f"cli_{attempt:03d}"
    stdout_path, stderr_path = prefix.with_suffix(".jsonl"), prefix.with_suffix(".txt")
    container = "goon-proposer-" + uuid.uuid4().hex
    instruction = (system_prompt + "\n\n" + prompt + "\n\n"
                   "All training history is mounted read-only at /history. Use your shell/read tools "
                   "to inspect it. Paths in the task description are relative to /history. "
                   "Work only in /workspace. Return a JSON object with source (complete Python "
                   "harness) and reflection (your reasoning about the change). Do not run benchmark "
                   "evaluations yourself. Do not retrieve benchmark data from the network.")
    with tempfile.TemporaryDirectory(prefix="goon-proposer-") as temporary:
        temp = Path(temporary)
        snapshot, workspace = temp / "history", temp / "workspace"
        snapshot_history(history, snapshot)
        workspace.mkdir(mode=0o777)
        workspace.chmod(0o777)  # node uid in the container must be able to create answer.json
        write_json(workspace / "schema.json", SCHEMA)
        command = ["docker", "run", "--rm", "--init", "--name", container, "-i",
                   "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "--pids-limit=256", "--memory=4g", "--cpus=2",
                   "--tmpfs", "/tmp:rw,nosuid,size=512m",
                   "--tmpfs", "/home/node:rw,nosuid,uid=1000,gid=1000,size=256m",
                   "--mount", f"type=bind,src={snapshot},dst=/history,readonly",
                   "--mount", f"type=bind,src={workspace},dst=/workspace",
                   "--workdir", "/workspace"]
        for name in auth:
            command += ["--env", name]
        command += [image, *cli_command(backend, model, max_calls)]
        write_json(prefix.with_suffix(".json"), {"backend": backend, "model": model,
                   "command": command, "prompt": instruction, "timeout_s": timeout})
        proc = None
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                proc = await asyncio.create_subprocess_exec(
                    *command, stdin=asyncio.subprocess.PIPE, stdout=stdout, stderr=stderr,
                    env={**os.environ, **auth})
                await asyncio.wait_for(proc.communicate(instruction.encode()), timeout=timeout)
            if proc.returncode:
                raise RuntimeError(f"{backend} exited {proc.returncode}; see {stderr_path}")
            return parse_proposal(backend, workspace, stdout_path)
        finally:
            # Also runs on timeout/cancellation. Killing only docker's client can
            # leave its container making paid requests, so explicitly remove it.
            cleanup = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)
            try:
                await asyncio.wait_for(cleanup.wait(), 15)
            except asyncio.TimeoutError:
                cleanup.kill()
                await cleanup.wait()
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
