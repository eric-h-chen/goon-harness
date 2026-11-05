"""Evaluator-owned RPC server for an untrusted, editable JavaScript agent loop.

Candidate code is never imported into Python or run on the host. Every sample
gets a fresh non-root Docker container with no network, credentials, evaluator
files, task filesystem mount, or Docker socket. Only its source is mounted.
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
import uuid

from inspect_ai.model import GenerateConfig
from inspect_ai.tool import ToolInfo, ToolParams

from harness_api import LimitExceeded
from pi_runtime import inspect_messages, pi_message

DEFAULT_IMAGE = 'goon-pi-loop:local'
MAX_SOURCE_BYTES = 2_000_000
MAX_LINE_BYTES = 16 * 1024 * 1024


def preflight():
    if not shutil.which('docker'):
        raise RuntimeError('Editable Pi loops require Docker')
    image = os.environ.get('GOON_PI_LOOP_IMAGE', DEFAULT_IMAGE)
    result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Build the editable Pi runtime: docker build -t goon-pi-loop:local runtimes/pi')
    return {'image_id': result.stdout.strip(),
            'python_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'message_bridge_sha256': hashlib.sha256(Path(__file__).with_name('pi_runtime.py').read_bytes()).hexdigest()}


def validate_source(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('Loop must be a regular .mjs file of at most 2 MB')
    source = path.read_text()
    if not source.strip():
        raise ValueError('Loop source is empty')
    # Parsing only; node --check does not execute imports or candidate code.
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node is required to syntax-check editable loops')
    result = subprocess.run([node, '--check', str(path.resolve())], capture_output=True,
                            text=True, timeout=15)
    if result.returncode:
        raise ValueError('Invalid JavaScript loop: ' + result.stderr[-2000:])


class EditablePiAgent:
    def __init__(self, source, runtime_info=None):
        self.source = source
        self.name = 'pi-editable-loop'
        self.runtime_info = runtime_info or preflight()
        self.harness_id = hashlib.sha256(json.dumps(
            {'source': source, 'runtime': self.runtime_info}, sort_keys=True).encode()).hexdigest()[:12]

    async def solve(self, task, env, llm):
        container = 'goon-pi-loop-' + uuid.uuid4().hex
        done = False
        process = None
        stderr_task = None
        stderr_tail = bytearray()
        with tempfile.TemporaryDirectory(prefix='goon-pi-loop-') as tmp:
            directory = Path(tmp)
            source = directory / 'harness.mjs'
            source.write_text(self.source)
            source.chmod(0o444)
            command = ['docker', 'run', '--rm', '--init', '-i', '--name', container,
                       '--network=none', '--read-only', '--user=1000:1000', '--cap-drop=ALL',
                       '--security-opt=no-new-privileges', '--pids-limit=64', '--memory=512m',
                       '--cpus=1', '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m,mode=1777',
                       '--env', 'HOME=/tmp', '--workdir', '/tmp',
                       '--mount', f'type=bind,src={source},dst=/candidate/harness.mjs,readonly',
                       self.runtime_info['image_id']]
            try:
                process = await asyncio.create_subprocess_exec(
                    *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, limit=MAX_LINE_BYTES)
                async def drain_stderr():
                    while chunk := await process.stderr.read(8192):
                        stderr_tail.extend(chunk)
                        del stderr_tail[:-8192]
                stderr_task = asyncio.create_task(drain_stderr())
                async def send(value):
                    process.stdin.write((json.dumps(value) + '\n').encode())
                    await process.stdin.drain()
                await send({'type': 'start', 'task': asdict(task)})
                requests = 0
                while line := await process.stdout.readline():
                    message = json.loads(line)
                    if message.get('type') == 'done':
                        break
                    if message.get('type') == 'fatal':
                        raise RuntimeError('Editable loop: ' + str(message.get('error'))[:4000])
                    if message.get('type') != 'request' or not isinstance(message.get('params'), dict):
                        raise ValueError('Invalid editable-loop RPC')
                    requests += 1
                    if requests > 10000:
                        raise LimitExceeded('editable-loop RPC request limit')
                    if done:
                        raise LimitExceeded('Task already submitted; no further operations allowed')
                    method, params = message.get('method'), message['params']
                    try:
                        if method == 'generate':
                            specs = params.get('tools', [])
                            if not isinstance(specs, list) or len(specs) > 128:
                                raise ValueError('At most 128 tool definitions allowed')
                            tools = [ToolInfo(name=s['name'], description=s.get('description', ''),
                                parameters=ToolParams.model_validate(s['parameters'])) for s in specs]
                            config = params.get('config', {})
                            if not isinstance(config, dict) or set(config) - {'max_tokens', 'temperature', 'reasoning_effort'}:
                                raise ValueError('Unsupported generation config')
                            if 'max_tokens' in config and not 1 <= config['max_tokens'] <= 32768:
                                raise ValueError('max_tokens must be 1..32768')
                            result = pi_message(await llm.generate(inspect_messages(params['messages']),
                                tools=tools, config=GenerateConfig(**config)))
                        elif method == 'exec':
                            timeout = params.get('timeout', 180)
                            if not isinstance(timeout, (int, float)) or not 0 < timeout <= 180:
                                raise ValueError('Command timeout must be greater than 0 and at most 180 seconds')
                            result = asdict(await env.exec(params['command'], timeout=timeout))
                        elif method == 'read_file':
                            result = await env.read_file(params['path'])
                        elif method == 'write_file':
                            await env.write_file(params['path'], params['content'])
                            result = None
                        elif method == 'submit':
                            feedback = await env.submit(params.get('answer') if task.kind == 'lcb' else None)
                            result = asdict(feedback)
                            done = feedback.attempts_left == 0
                        else:
                            raise ValueError(f'Unknown RPC method: {method}')
                        await send({'id': message['id'], 'result': result})
                    except (KeyError, ValueError, TypeError, FileNotFoundError, IsADirectoryError,
                            PermissionError, UnicodeDecodeError) as exc:
                        await send({'id': message['id'], 'error': str(exc)[:4000]})
                await asyncio.wait_for(process.wait(), 10)
                if process.returncode:
                    raise RuntimeError('Editable loop exited: ' + stderr_tail.decode(errors='replace'))
            finally:
                # Cancel the actual container as well as the docker client on timeout/budget/error.
                cleanup = await asyncio.create_subprocess_exec('docker', 'rm', '-f', container,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                try:
                    await asyncio.wait_for(cleanup.wait(), 15)
                except asyncio.TimeoutError:
                    cleanup.kill()
                    await cleanup.wait()
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
                if stderr_task:
                    await stderr_task
