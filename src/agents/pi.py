"""Seed harness backed by the pinned Pi agent-core runtime.

The optimizer may change this configuration; pi_runtime owns the model/tool bridge.
"""
from pi_runtime import PiAgent


def make_agent():
    return PiAgent(
        name="pi-baseline",
        system_prompt="""You are an expert software engineer working through tools.
Explore the task, implement a solution, and test it when useful.
Use read_file, write_file, and bash to work in the task environment.
Call submit when your solution is ready. If feedback reports failure and you have
attempts remaining, investigate the failure and try again.""",
        tools=("bash", "read_file", "write_file"),
        tool_output_chars=20000,
        bash_timeout=180,
    )
