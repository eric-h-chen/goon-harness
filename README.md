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
and full sample traces. This is a local adaptation using Inspect model providers
and read/search tools, rather than the paper's Claude Code wrapper. Candidates are
self-contained Python files implementing the existing agent contract.

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

The proposer has no shell or general filesystem tool; history reads/searches reject
path traversal and symlink escapes. **Generated solver harnesses still run in the
evaluator's Python process**, as described in spec §4.1. The contract source scan
is a heuristic, not process isolation. This implementation does not claim to prevent
a deliberately evasive candidate from accessing host data or credentials. Inspect
candidate code/contract violations before interpreting results, and use process
isolation if stronger enforcement is required. Test editing inside a task sandbox
is still allowed by the ImpossibleBench protocol.

Plot held-out checkpoints with the existing `eval.py plot` command. Per-candidate
training rewards and per-iteration selections remain in `history/`. Keep independent
seeds and fixed-harness controls in separate run directories for the final experiment.
