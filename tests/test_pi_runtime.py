import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.tool import ToolCall

from harness_api import ExecResult, Feedback, LimitExceeded, Limits, TaskSpec
from pi_runtime import PiAgent, inspect_messages, pi_message, preflight


def output(*calls, text=""):
    return ModelOutput(model="mock", choices=[{"message": ChatMessageAssistant(
        content=text, tool_calls=[ToolCall(id=f"call-{i}", function=name, arguments=args)
                                  for i, (name, args) in enumerate(calls)]),
        "stop_reason": "tool_calls" if calls else "stop"}])


def task(kind="lcb"):
    return TaskSpec(kind, "Solve the synthetic task", "/task", None, True, Limits(30, 3, 900))


class Env:
    def __init__(self):
        self.calls = []
        self.attempts_left = 3

    async def exec(self, command, timeout):
        self.calls.append(("bash", command, timeout))
        return ExecResult(0, "command result", "")

    async def read_file(self, path):
        self.calls.append(("read", path))
        return "file content"

    async def write_file(self, path, content):
        self.calls.append(("write", path, content))

    async def submit(self, answer):
        self.calls.append(("submit", answer))
        passed = answer in ("correct", None)
        self.attempts_left = 0 if passed else self.attempts_left - 1
        return Feedback(passed, "feedback", self.attempts_left)


@pytest.fixture(autouse=True)
def require_pi():
    try:
        preflight()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.asyncio
async def test_real_pi_runtime_tools_and_retry():
    """Real installed Pi loop and Node/Python bridge; only the model/env are fake."""
    env = Env()
    responses = [output(text="I should inspect the files."),
                 output(("bash", {"command": "pwd"}), ("read_file", {"path": "input.py"}),
                        ("write_file", {"path": "answer.py", "content": "solution"})),
                 output(("submit", {"code": "wrong"})), output(("submit", {"code": "correct"}))]
    conversations = []
    class LLM:
        async def generate(self, messages, tools):
            assert {t.name for t in tools} == {"bash", "read_file", "write_file", "submit"}
            conversations.append(messages)
            return responses.pop(0)
    await asyncio.wait_for(PiAgent().solve(task(), env, LLM()), 15)
    assert env.calls == [("bash", "pwd", 180), ("read", "input.py"),
                         ("write", "answer.py", "solution"), ("submit", "wrong"), ("submit", "correct")]
    assert len(conversations) == 4  # No model call after the final submission.
    assert "Continue working" in conversations[1][-1].text
    assert '"passed": false' in conversations[3][-1].text
    assert any(m.role == "tool" and m.text == "file content" for m in conversations[2])


@pytest.mark.asyncio
async def test_pi_limits_stop_child_and_no_cross_task_state():
    agent = PiAgent()
    class LimitedLLM:
        async def generate(self, *args, **kwargs):
            raise LimitExceeded("budget")
    await asyncio.wait_for(agent.solve(task(), Env(), LimitedLLM()), 10)
    class LLM:
        async def generate(self, messages, **kwargs):
            assert len(messages) == 2  # Fresh system + task, even on a shared Agent.
            return output(("submit", {}))
    environments = [Env(), Env()]
    await asyncio.wait_for(asyncio.gather(*(agent.solve(task("swe"), env, LLM()) for env in environments)), 15)
    assert all(env.calls == [("submit", None)] for env in environments)


def test_assistant_roundtrip_preserves_inspect_fields():
    original = output(("bash", {"command": "echo hi"}), text="Checking")
    restored = inspect_messages([pi_message(original)])[0]
    assert restored.model_dump() == original.message.model_dump()


def test_pi_base_selected_and_configuration_is_in_history(monkeypatch, tmp_path):
    import optimize as opt
    from pathlib import Path
    from train import write_json
    rewards = iter([0, 1])
    def score(path, model, batch, directory, **kwargs):
        assert "PiAgent" in Path(path).read_text()
        value = {"valid": True, "reward": next(rewards)}
        write_json(Path(directory) / "reward.json", value)
        return value
    async def propose(history, model, prompt, directory, calls, **kwargs):
        assert "PiAgent" in kwargs["system_prompt"]
        assert (history / "pi_api.txt").exists()
        return {"source": "from pi_runtime import PiAgent\ndef make_agent(): return PiAgent()",
                "reflection": "Adjust prompt"}
    monkeypatch.setattr(opt, "score_training", score)
    monkeypatch.setattr(opt, "propose", propose)
    result = opt.optimize(tmp_path, "mock", "mock", base_harness="pi",
                          iterations=1, candidates=1, batch_size=1)
    config = json.loads((tmp_path / "optimization.json").read_text())
    assert result["reward"] == 1 and config["base_harness"] == "pi"
    assert config["pi_runtime"]["lock_sha256"]


def test_pi_harness_identity_includes_configuration():
    from adapter import load_agent
    _, first = load_agent(PiAgent(system_prompt="first"))
    _, second = load_agent(PiAgent(system_prompt="second"))
    assert first != second


def test_pi_through_inspect_adapter_and_local_sandbox(tmp_path):
    """Real Pi + Inspect solver/metering/env/scorer, with a mock provider and no dataset."""
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import get_model
    from inspect_ai.scorer import match
    from adapter import graded_on_submission, harness_solver

    responses = [output(("bash", {"command": "printf bridge-ok"})),
                 output(("submit", {"code": "print(1)"}))]
    def model_response(messages, tools, tool_choice, config):
        if len(responses) == 1:
            assert "bridge-ok" in messages[-1].text
        return responses.pop(0)
    model = get_model("mockllm/model", custom_outputs=model_response)
    benchmark = Task(dataset=[Sample(input="Print one", target="print(1)", id="synthetic")],
                     solver=harness_solver(PiAgent(), "lcb_real"),
                     scorer=graded_on_submission(match()), sandbox="local")
    [log] = inspect_eval(benchmark, model=model, log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    [sample] = log.samples
    assert sample.error is None
    assert sample.scores["pass"].value == 1
    assert sample.metadata["harness"]["llm_calls"] == 2
    assert sample.metadata["harness"]["submitted"]
