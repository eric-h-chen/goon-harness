# goon-harness

This project measures whether optimizing an agent harness leads to emergent reward hacking.
The proposal is in `project_proposal/`, and the eval design is in [`specs/eval.md`](specs/eval.md).

## Setup

```bash
uv sync                          # Python ≥ 3.11, installs into .venv
uv run python src/eval.py prepare-lcb   # caches training + held-out tests (~4.5 GB download once)
```

Docker is required to run the eval. Run the SWE-bench suites on an x86_64 Linux host (see spec §8).

## ⚠️ Held-out tasks

`data/splits/HELDOUT_EVAL_DO_NOT_TRAIN.json` is **for evaluation only**.
The optimization loop should take its training tasks from `train_ids(...)`.
It should also call `assert_not_heldout(ids)` on every batch it scores.
Don't let the harness proposer read `results/*/eval/`, since that directory holds held-out traces and scores.

## Writing a harness

A harness is a `.py` file, or a directory containing `harness.py`, that defines `make_agent()`.
The agent it returns must implement `harness_api.Agent`:

```python
class MyAgent:
    name = "my-agent"
    async def solve(self, task: TaskSpec, env: Env, llm: LLM) -> None:
        ...                                  # work through env.exec / read_file / write_file
        await env.submit(answer)             # graded on the last submission

def make_agent(): return MyAgent()
```

Spec §4.1 lists the rules a harness must follow.
In short, all model calls go through `llm`, all file and shell access goes through `env`, and the agent can't write memory during eval.
`src/agents/baseline.py` is a working example.

## From the optimization loop

```python
import sys; sys.path.insert(0, "src")
from eval import evaluate, check, plot_iterations, train_ids, assert_not_heldout

assert check("candidates/h7.py", model=MODEL)["ok"]                   # cheap gate on 2 train tasks/suite
evaluate("candidates/h7.py", iteration=7, run_dir="results/run1", model=MODEL)
plot_iterations("results/run1", "figs/run1.pdf")
```

## CLI

```bash
python src/eval.py check  src/agents/baseline.py --model anthropic/claude-sonnet-5-5
python src/eval.py run    --iteration 0 --run-dir results/run1 --model anthropic/claude-sonnet-5-5
python src/eval.py plot   results/run1 [results/run2 ...] -o figs/iters.pdf --tradeoff figs/tradeoff.pdf
```

`run --limit N` evaluates only the first N held-out tasks per suite, for debugging.
Those rows are marked `partial`, and plots skip them.

## Tests

```bash
uv run pytest
```

The end-to-end tests use Inspect's mock model and a local sandbox, and they cover only the LiveCodeBench suites.

## Harness optimization

`src/optimize.py` implements a [Meta-Harness-style search](https://yoonholee.com/meta-harness/):
its proposer can browse all earlier candidate source, reflections, training scores,
and full sample traces. Select an Inspect tool loop, Claude Code, or Codex as the
proposer. Candidates are self-contained Python files implementing the existing
agent contract, optionally configuring the actual Pi agent-core runtime.

Install dependencies with `uv sync`, install your provider's SDK if needed (for
example `uv pip install anthropic` or `uv pip install openai`), configure credentials
for your chosen Inspect model providers, and start Docker. For real LiveCodeBench scoring (including held-out
checkpoints), cache tests first:

```bash
uv run python src/eval.py prepare-lcb --partition all
```

This downloads approximately 4.5 GB once. `--partition train` or `heldout` can cache
only one side. These files are evaluator inputs; never expose them to the proposer.
Training only on `lcb_impossible` uses its `original` dataset and does not need this
real-LiveCodeBench cache.

Score a harness on a seeded training batch:

```bash
uv run python src/train.py src/agents/baseline.py \
  --model "$SOLVER_MODEL" --suites lcb_impossible --batch-size 10 \
  --out-dir results/training-check
```

Omit `--batch-size` to score the entire selected training pool. The Python interface
is `train.score_training(harness, model, batch, out_dir)`, with batches constructed
by `train.training_batch(suites, batch_size, seed)` or supplied explicitly.
All IDs must belong to the named suite's training pool; held-out IDs are rejected
before loading the harness. ImpossibleBench suites always use `original` here.
Reward is the equal-weight mean of selected suite pass rates. Missing/error samples
stay in the denominator and invalidate a candidate for selection; failures to solve,
non-submissions, and exhausted budgets count as ordinary failures.

Run a LiveCodeBench pilot, with full held-out evaluations at iteration 0, every
second iteration, and the final iteration:

```bash
uv run python src/optimize.py \
  --run-dir results/pilot --model "$SOLVER_MODEL" --proposer-model "$PROPOSER_MODEL" \
  --suites lcb_impossible --batch-size 10 --seed 329 \
  --iterations 4 --candidates 2 --proposer-calls 30 \
  --eval-every 2 --eval-suites lcb_real lcb_impossible \
  --max-samples 4 --max-sandboxes 4
```

Set the two model variables to provider/model identifiers supported by your Inspect
installation. The solver is fixed across training and held-out evaluation; the
proposer can use a different model. No paid calls are made until you run these commands.
SWE-bench suites require an x86_64 Linux host. Add them explicitly to `--suites` and
`--eval-suites` when that infrastructure is ready. `--eval-every 0` (the default)
disables held-out evaluations. Each full two-suite LiveCodeBench checkpoint costs
300 task episodes; all four suites cost 750, in addition to training/proposer calls.

The same seeded training batch is used for every candidate. The seed is scored
first, and strict reward improvements replace the incumbent; ties retain it.
Each candidate can inspect all prior candidates, including rejected ones. A failed
contract scan or agent construction rejects the candidate. Infrastructure errors
in returned evaluations invalidate it; setup/provider exceptions stop the run for
repair. This fixed-batch search can overfit its training batch: held-out metrics
are the independent measurement, not a candidate selection signal.

Re-run the same command to resume; increase `--iterations` to extend it. Completed
candidates and checkpoints are reused; interrupted work without a saved reward may
be repeated and incur additional cost. Configuration, code, dependencies, model,
and training batch must match. Run directories support only one optimizer writer.

```
results/pilot/
  optimization.json             # frozen search configuration and fingerprints
  state.json                    # latest completed selection
  best.py                       # winning harness, exported on successful completion
  history/                      # proposer's only readable directory
    harness_api.py
    objective.json
    baseline/harness.py
    baseline/training/          # reward.json, Inspect logs, full JSON sample traces
    iter_001_candidate_00/      # source, reflection, proposer transcript, training, result
    selection_001.json
  eval/                         # held-out logs and metrics; invisible to proposer tools
```

The default Inspect proposer has no shell or general filesystem tool; its history
reads/searches reject path traversal and symlink escapes. Claude Code and Codex
proposers run in isolated containers as described below. **Generated solver harnesses still run in the
evaluator's Python process**, as described in spec §4.1. The contract source scan
is a heuristic, not process isolation. This implementation does not claim to prevent
a deliberately evasive candidate from accessing host data or credentials. Inspect
candidate code/contract violations before interpreting results, and use process
isolation if stronger enforcement is required. Test editing inside a task sandbox
is still allowed by the ImpossibleBench protocol.

Plot held-out checkpoints with the existing `eval.py plot` command. Per-candidate
training rewards and per-iteration selections remain in `history/`. Keep independent
seeds and fixed-harness controls in separate run directories for the final experiment.

## Claude Code and Codex proposers

`--proposer-backend` selects the agent that proposes changes; `--model` always
selects the fixed task-solving model through Inspect.

| Backend | `--proposer-model` format | Authentication |
| --- | --- | --- |
| `inspect` (default) | Inspect `provider/model` | Provider SDK credentials |
| `claude-code` | Native Claude Code model ID or alias, without `anthropic/` | `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` |
| `codex` | Native Codex model ID, without `openai/` | `CODEX_API_KEY` or `OPENAI_API_KEY` |

Build the pinned CLI image once (Docker must be running):

```bash
docker build -t goon-proposers:local runtimes/proposers
```

Each proposal uses a fresh, non-root container containing a read-only snapshot of
training history at `/history` and a writable temporary `/workspace`. The checkout,
held-out results, user home/configuration, and Docker socket are not mounted.
Symlinked files/directories are excluded from the snapshot. The CLI can browse
history with its native shell/file tools and returns schema-validated source and
reflection. It does not directly run the benchmark or select the winner.

The container needs outbound access to its model provider; this is filesystem
isolation, not an egress allowlist. The prompt prohibits fetching benchmark data.
Only the chosen provider's credential is forwarded. Host CLI logins and custom
plugins/settings are not inherited. Codex uses its external-sandbox mode inside
this container; there is deliberately no unsandboxed host CLI fallback.

```bash
uv run python src/optimize.py \
  --run-dir results/claude-proposer --model "$SOLVER_MODEL" \
  --proposer-backend claude-code --proposer-model "$CLAUDE_MODEL" \
  --iterations 2 --candidates 1 --batch-size 5 \
  --proposer-calls 30 --proposer-timeout 1800 \
  --eval-every 1 --eval-suites lcb_real lcb_impossible
```

For Codex, change the backend to `codex`, supply its native model ID, and use a
different run directory. `--proposer-calls` limits model calls for Inspect and
agent turns for Claude Code. **Codex has no equivalent call-count limit here**;
`--proposer-timeout` bounds wall time for all backends. None of these limits is a
hard dollar cap. The container is removed on success, failure, cancellation, or
timeout. A machine/process crash can still require manual container cleanup.

Raw CLI JSONL events, stderr, and invocation metadata are saved per attempt as
`history/<candidate>/cli_000.{jsonl,txt,json}`. Failed attempts remain available
after resuming. The resolved image ID is frozen in `optimization.json`; changing
the CLI image requires a fresh run. You can select another prebuilt image with
`--proposer-image` (it must provide compatible `codex` and/or `claude` executables).

## Pi as the base harness

Install the pinned Node dependencies (Node 22 or newer):

```bash
npm ci --ignore-scripts --prefix runtimes/pi
```

`--base-harness pi` selects `src/agents/pi.py`. This uses the **actual
`@earendil-works/pi-agent-core` Agent runtime**, not a Python reimplementation.
It is an agent-core integration, not the complete interactive Pi coding-agent
application: user extensions, saved sessions, automatic compaction, and Pi's
built-in local filesystem/provider tools are not loaded.

The Python factory returns `pi_runtime.PiAgent`. A fresh Node process manages
each task's conversation/tool loop. Its model requests pass back to Inspect's
metered `llm`, and its `bash`, `read_file`, `write_file`, and `submit` operations
pass through the task's `env`. The Node process receives no provider credentials.
Submission limits, test feedback, final-submission grading, and task isolation
therefore use the same evaluator interfaces as the Python baseline.

```bash
uv run python src/optimize.py \
  --run-dir results/pi-codex --base-harness pi \
  --model "$SOLVER_MODEL" --proposer-backend codex --proposer-model "$CODEX_MODEL" \
  --iterations 2 --candidates 1 --batch-size 5 \
  --eval-every 1 --eval-suites lcb_real lcb_impossible
```

The editable candidate is a Python Pi factory: prompts, enabled tools, output
truncation, command timeout, and continuation instruction. The pinned JavaScript
runtime and transport are fixed infrastructure; the proposer is instructed to
retain that boundary. Runtime/factory fingerprints identify Pi evaluations, and
the optimizer freezes the Node version, dependency lock, and bridge hashes.
Use `--seed-harness` alongside `--base-harness pi` for a custom Pi factory.
To score or evaluate Pi independently, pass `src/agents/pi.py` to `train.py` or
`eval.py --harness` just like any other Python harness.

The tests in `tests/test_pi_runtime.py` exercise the real installed Pi runtime
with fake model responses and task environments; they skip if Node/dependencies
are absent. `tests/test_cli_proposers.py` uses a subprocess transport fake to test
command construction, parsing, snapshots, failure cleanup, and optimizer dispatch.
These do not substitute for live provider/Docker benchmark smoke tests.

Upstream interfaces: [Claude Code CLI](https://code.claude.com/docs/en/cli-reference),
[Codex noninteractive mode](https://developers.openai.com/codex/noninteractive),
[Pi agent-core](https://github.com/earendil-works/pi/tree/main/packages/agent).
