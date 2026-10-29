import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import optimize as opt
import train
from splits import HeldOutLeakError, heldout_ids, train_ids


def test_training_batch_is_seeded_and_validated():
    batch = train.training_batch(["lcb_impossible"], 3, 42)
    assert batch == train.training_batch(["lcb_impossible"], 3, 42)
    assert batch != train.training_batch(["lcb_impossible"], 3, 43)
    with pytest.raises(HeldOutLeakError):
        train.validate_batch({"lcb_impossible": heldout_ids("impossible_livecodebench")[:1]})
    with pytest.raises(ValueError):
        train.validate_batch({"lcb_real": batch["lcb_impossible"]})
    with pytest.raises(ValueError):
        train.training_batch(["lcb_impossible"], 0)
    with pytest.raises(ValueError):
        train.training_batch(["lcb_impossible"], 100000)


def sample(sid, passed=1, error=None, agent_error=None):
    return NS(id=sid, error=error, metadata={"harness": {"agent_error": agent_error}},
              scores={"pass": NS(value=passed)}, model_dump_json=lambda **kw: json.dumps({"id": sid}))


def test_training_scores_original_only_and_invalidates_missing_samples(monkeypatch, tmp_path):
    import inspect_ai
    batch = train.training_batch(["lcb_impossible", "lcb_real"], 2)
    calls = []
    monkeypatch.setattr(train, "load_agent", lambda _: (object(), "hash"))
    def build(suite, ids, agent, **kwargs):
        calls.append((suite, kwargs["split"]))
        return suite
    monkeypatch.setattr(train, "build_task", build)
    def run(suite, **kwargs):
        ids = batch[suite]
        samples = [sample(ids[0]), sample(ids[1], passed=0)] if suite == "lcb_impossible" else [sample(ids[0])]
        return [NS(samples=samples, location="log.eval", status="success")]
    monkeypatch.setattr(inspect_ai, "eval", run)
    result = train.score_training("unused", "mock", batch, tmp_path)
    assert calls == [("lcb_impossible", "original"), ("lcb_real", None)]
    assert result["reward"] == .5  # Missing sample is NOT dropped from denominator.
    assert not result["valid"]
    assert result["suites"]["lcb_real"]["errors"] == 1
    assert len(list((tmp_path / "traces").rglob("*.json"))) == 3
    with pytest.raises(FileExistsError):
        train.score_training("unused", "mock", batch, tmp_path)


def test_reject_heldout_before_loading_harness(monkeypatch, tmp_path):
    def forbidden(*args):
        pytest.fail("must reject IDs before executing candidate")
    monkeypatch.setattr(train, "load_agent", forbidden)
    with pytest.raises(HeldOutLeakError):
        train.score_training("unused", "mock", {"lcb_real": heldout_ids("livecodebench")[:1]}, tmp_path)


def test_history_blocks_traversal_and_symlinks(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    secret = tmp_path / "eval.json"
    secret.write_text("HELDOUT SECRET")
    (history / "trace.json").write_text("training evidence\nsecond line")
    (history / "leak.json").symlink_to(secret)
    view = opt.History(history)
    assert view.files() == ["trace.json"]
    for path in ("../eval.json", str(secret), "leak.json"):
        with pytest.raises(ValueError):
            view.call("read_file", {"path": path, "offset": 0, "length": 100})
    assert view.call("search", {"query": "SECRET", "offset": 0}) == ""
    assert "training evidence" in view.call("search", {"query": "evidence", "offset": 0})
    assert view.call("read_file", {"path": "trace.json", "offset": 9, "length": 8}) == "evidence"


def test_search_selection_resume_and_eval_separation(monkeypatch, tmp_path):
    proposed, scored, evaluated = [], [], []
    rewards = iter([.4, .7, .99, .7, .8])  # Invalid .99, tie .7, strict improvement .8.
    def score(path, model, batch, out, **kwargs):
        reward = next(rewards)
        scored.append(str(path))
        result = {"reward": reward, "valid": reward != .99}
        train.write_json(Path(out) / "reward.json", result)
        return result
    async def propose(history, model, prompt, out, max_calls):
        assert "eval" not in opt.History(history).files()
        assert "HELDOUT SECRET" not in opt.History(history).call("search", {"query": "HELDOUT SECRET"})
        proposed.append(prompt)
        return {"source": "# candidate\ndef make_agent(): pass\n", "reflection": "training evidence"}
    def evaluate(harness, iteration, run, model, **kwargs):
        evaluated.append(iteration)
        train.write_json(Path(run) / "eval" / f"iter_{iteration:03d}" / "metrics.json", {"secret": "HELDOUT SECRET"})
    monkeypatch.setattr(opt, "score_training", score)
    monkeypatch.setattr(opt, "propose", propose)
    monkeypatch.setattr(opt, "evaluate", evaluate)
    args = dict(model="mock", proposer_model="mock", iterations=2, candidates=2, batch_size=2, eval_every=1)
    result = opt.optimize(tmp_path, **args)
    assert result["reward"] == .8
    assert result["winner"] == "iter_002_candidate_01"
    assert evaluated == [0, 1, 2]
    assert len(proposed) == 4 and len(scored) == 5
    assert opt.optimize(tmp_path, **args) == result
    assert len(proposed) == 4 and len(scored) == 5 and evaluated == [0, 1, 2]
    with pytest.raises(ValueError, match="configuration"):
        opt.optimize(tmp_path, **{**args, "model": "changed"})


@pytest.mark.asyncio
async def test_proposer_tools_and_transcript(monkeypatch, tmp_path):
    from inspect_ai.model import ChatMessageAssistant, ModelOutput
    from inspect_ai.tool import ToolCall
    history = tmp_path / "history"
    history.mkdir()
    (history / "evidence.txt").write_text("a useful trace")
    calls = 0
    class Model:
        async def generate(self, messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                tool = ToolCall(id="read", function="read_file", arguments={"path": "evidence.txt", "offset": 0, "length": 100})
            else:
                assert messages[-1].content == "a useful trace"
                tool = ToolCall(id="submit", function="submit", arguments={"source": "x = 1", "reflection": "why"})
            return ModelOutput(model="mock", choices=[{"message": ChatMessageAssistant(content="", tool_calls=[tool]), "stop_reason": "tool_calls"}])
    monkeypatch.setattr(opt, "get_model", lambda _: Model())
    result = await opt.propose(history, "mock", "improve", tmp_path)
    assert result["source"] == "x = 1"
    assert len(json.loads((tmp_path / "proposer.json").read_text())["outputs"]) == 2


def test_prepare_lcb_can_include_training(monkeypatch):
    import eval as ev
    import lcb_real
    calls = []
    monkeypatch.setattr(lcb_real, "build_cache", lambda keep_tests_for: calls.append(set(keep_tests_for)))
    ev.main(["prepare-lcb", "--partition", "all"])
    assert calls == [set(train_ids("livecodebench")) | set(heldout_ids("livecodebench"))]


def test_loop_through_real_inspect_evaluator(monkeypatch, tmp_path):
    """Exercise agent loading, execution, Inspect scoring/logging and loop selection offline."""
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import Score, mean, scorer
    from inspect_ai.solver import solver

    @scorer(metrics=[mean()], name="pass")
    def tiny_score():
        async def score(state, target):
            return Score(value=float(state.output.completion == "correct"))
        return score

    @solver
    def tiny_solver(agent):
        async def solve(state, generate):
            class Env:
                async def submit(self, answer):
                    state.output = ModelOutput.from_content("mockllm/model", answer)
            await agent.solve(None, Env(), None)
            return state
        return solve

    def build(suite, ids, agent, **kwargs):
        assert kwargs["split"] == "original"
        return Task(dataset=[Sample(id=sid, input="synthetic training task") for sid in ids],
                    solver=tiny_solver(agent), scorer=tiny_score(), name="tiny_training")

    def source(answer):
        return ("class Agent:\n    name = 'tiny'\n"
                f"    async def solve(self, task, env, llm):\n        await env.submit({answer!r})\n"
                "def make_agent(): return Agent()\n")

    seed = tmp_path / "seed.py"
    seed.write_text(source("wrong"))
    async def proposer(*args):
        return {"source": source("correct"), "reflection": "Fix the synthetic answer"}
    monkeypatch.setattr(train, "build_task", build)
    monkeypatch.setattr(opt, "propose", proposer)
    run = tmp_path / "run"
    result = opt.optimize(run, "mockllm/model", "mockllm/model", iterations=1,
                          candidates=1, batch_size=2, seed_harness=seed, display="none")
    assert result["reward"] == 1
    assert result["winner"] == "iter_001_candidate_00"
    baseline = json.loads((run / "history/baseline/training/reward.json").read_text())
    assert baseline["reward"] == 0 and baseline["valid"]
    assert len(list((run / "history").rglob("*.eval"))) == 2
    traces = list((run / "history/iter_001_candidate_00/training/traces").rglob("*.json"))
    assert len(traces) == 2
    assert json.loads(traces[0].read_text())["output"]["completion"] == "correct"
