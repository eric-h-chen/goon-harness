# goon-harness

This project measures whether optimizing an agent harness leads to emergent reward hacking.
The proposal is in `project_proposal/`, and the eval design is in [`specs/eval.md`](specs/eval.md).

## Setup

```bash
uv sync                          # Python ≥ 3.11, installs into .venv
python src/eval.py prepare-lcb   # caches held-out LiveCodeBench tests (downloads ~4.5 GB once)
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
