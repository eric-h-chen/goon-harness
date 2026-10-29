import json
from pathlib import Path
import sys

import pytest

import cli_proposers as cli
import optimize as opt
from train import write_json


@pytest.fixture
def docker_stub(tmp_path, monkeypatch):
    """An executable transport fake: tests subprocess I/O without Docker or paid APIs."""
    binary = tmp_path / "docker"
    binary.write_text("#!/usr/bin/env python3\n" + '''
import json, os, pathlib, sys, time
args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[:2] == ["image", "inspect"]:
    print("sha256:test-image")
    sys.exit(0)
if args[:2] == ["rm", "-f"]:
    sys.exit(0)
prompt = sys.stdin.read()
mounts = [args[i+1] for i, value in enumerate(args) if value == "--mount"]
def mount(dest):
    spec = next(s for s in mounts if "dst=" + dest in s)
    return pathlib.Path(dict(p.split("=", 1) for p in spec.split(",") if "=" in p)["src"])
history = mount("/history")
assert (history / "baseline" / "harness.py").exists()
assert not (history / "secret.json").exists()
assert not (history / "eval").exists()
assert "/history" in prompt
model = args[args.index("--model")+1]
if model == "timeout":
    time.sleep(30)
if model == "failed":
    print("simulated provider failure", file=sys.stderr)
    sys.exit(2)
proposal = {"source": "def make_agent(): pass", "reflection": "Training evidence"}
if "codex" in args:
    (mount("/workspace") / "answer.json").write_text(json.dumps(proposal))
    print(json.dumps({"type": "thread.started"}))
else:
    print(json.dumps({"type": "assistant", "message": "inspected history"}))
    print(json.dumps({"type": "result", "subtype": "success", "structured_output": proposal}))
''')
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + str(Path(sys.executable).parent) + ":/usr/bin:/bin")
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(tmp_path / "docker.log"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    history = tmp_path / "history"
    (history / "baseline").mkdir(parents=True)
    (history / "baseline/harness.py").write_text("# seed")
    secret = tmp_path / "heldout.json"
    secret.write_text("private evaluation")
    (history / "secret.json").symlink_to(secret)
    return history, tmp_path / "docker.log"


@pytest.mark.parametrize("backend", ["codex", "claude-code"])
@pytest.mark.asyncio
async def test_cli_transport_schema_and_isolation(docker_stub, tmp_path, backend):
    history, log = docker_stub
    assert cli.preflight(backend, "test-image") == "sha256:test-image"
    out = tmp_path / "output"
    result = await cli.propose_cli(history, "test-model", "improve", out, backend=backend,
                                   system_prompt="Optimize", timeout=5)
    assert result["source"] == "def make_agent(): pass"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    run = next(call for call in calls if call[0] == "run")
    assert "--read-only" in run and "--cap-drop=ALL" in run
    assert "readonly" in next(s for s in run if "dst=/history" in s)
    assert len([s for s in run if s == "--mount"]) == 2
    assert calls[-1][:2] == ["rm", "-f"]
    metadata = (out / "cli_000.json").read_text()
    assert "test-openai-key" not in metadata and "test-anthropic-key" not in metadata
    assert (out / "cli_000.jsonl").exists()


@pytest.mark.parametrize("model,exception", [("failed", RuntimeError), ("timeout", TimeoutError)])
@pytest.mark.asyncio
async def test_cli_failure_or_timeout_cleans_container(docker_stub, tmp_path, model, exception):
    history, log = docker_stub
    out = tmp_path / "output"
    with pytest.raises(exception):
        await cli.propose_cli(history, model, "improve", out, backend="codex",
                              system_prompt="Optimize", timeout=.2)
    assert json.loads(log.read_text().splitlines()[-1])[:2] == ["rm", "-f"]
    assert (out / "cli_000.jsonl").exists()
    assert (out / "cli_000.txt").exists()


def test_reject_bad_cli_results(tmp_path):
    answer = tmp_path / "answer.json"
    answer.write_text('{"source": "x"}')
    with pytest.raises(ValueError):
        cli.parse_proposal("codex", tmp_path, tmp_path / "stdout")
    stdout = tmp_path / "stdout"
    stdout.write_text(json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True}))
    with pytest.raises(ValueError):
        cli.parse_proposal("claude-code", tmp_path, stdout)
    answer.unlink()
    external = tmp_path / "external.json"
    external.write_text(json.dumps({"source": "x", "reflection": "y"}))
    answer.symlink_to(external)
    with pytest.raises(ValueError):
        cli.parse_proposal("codex", tmp_path, stdout)


def test_snapshot_rejects_directory_symlinks(tmp_path):
    history, outside = tmp_path / "history", tmp_path / "outside"
    history.mkdir()
    outside.mkdir()
    (outside / "secret.json").write_text("secret")
    (history / "link").symlink_to(outside, target_is_directory=True)
    cli.snapshot_history(history, tmp_path / "copy")
    assert not list((tmp_path / "copy").rglob("*.json"))


@pytest.mark.parametrize("backend", ["claude-code", "codex"])
def test_optimizer_selects_cli_backend_and_freezes_image(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(opt, "cli_preflight", lambda *args: "sha256:test")
    scores = iter([.1, .8])
    def score(path, model, batch, directory, **kwargs):
        result = {"valid": True, "reward": next(scores)}
        write_json(Path(directory) / "reward.json", result)
        return result
    calls = []
    async def propose(history, model, prompt, out, **kwargs):
        calls.append(kwargs)
        return {"source": "def make_agent(): pass", "reflection": "training evidence"}
    monkeypatch.setattr(opt, "score_training", score)
    monkeypatch.setattr(opt, "propose_cli", propose)
    args = dict(model="mock", proposer_model="native-model", proposer_backend=backend,
                iterations=1, candidates=1, batch_size=1)
    result = opt.optimize(tmp_path, **args)
    assert result["reward"] == .8
    assert calls[0]["backend"] == backend and calls[0]["image"] == "sha256:test"
    assert opt.optimize(tmp_path, **args) == result
    assert len(calls) == 1
    monkeypatch.setattr(opt, "cli_preflight", lambda *args: "sha256:changed")
    with pytest.raises(ValueError, match="configuration"):
        opt.optimize(tmp_path, **args)
