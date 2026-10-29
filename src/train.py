"""Training-only reward evaluation; never uses impossible mutations or held-out IDs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from adapter import SUITES, build_task, load_agent
from splits import assert_not_heldout, train_ids


class HarnessLoadError(ValueError):
    """Candidate failed to import or construct its agent."""


def training_batch(suites, batch_size: int | None = None, seed: int = 329) -> dict[str, list[str]]:
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("Select at least one suite, without duplicates")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch = {}
    for suite in suites:
        pool = sorted(train_ids(SUITES[suite].split_key))
        if batch_size is not None and batch_size > len(pool):
            raise ValueError(f"{suite}: requested {batch_size}, only {len(pool)} training tasks")
        batch[suite] = random.Random(f"{seed}:{suite}").sample(pool, batch_size or len(pool))
    validate_batch(batch)
    return batch


def validate_batch(batch):
    if not batch:
        raise ValueError("Empty training batch")
    for suite, ids in batch.items():
        if suite not in SUITES or not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid suite or empty/duplicate IDs: {suite}")
        assert_not_heldout(ids)
        if not set(ids) <= set(train_ids(SUITES[suite].split_key)):
            raise ValueError(f"{suite}: IDs outside its training pool")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def score_training(harness, model: str, batch: dict[str, list[str]], out_dir,
                   *, sandbox=None, max_samples=None, max_sandboxes=None, display=None) -> dict:
    """Macro-average suite pass rates on a specified training batch.

    All requested tasks stay in the denominator. Missing samples, infrastructure
    errors, or agent exceptions invalidate selection (valid=False). Non-submission
    and budget exhaustion are ordinary failures. Full sample traces are exported.
    """
    validate_batch(batch)  # Before importing/executing candidate code or constructing tasks.
    from inspect_ai import eval as inspect_eval

    out = Path(out_dir)
    if (out / "reward.json").exists():
        raise FileExistsError(f"{out} already contains a training reward")
    try:
        agent, harness_id = load_agent(harness)
    except Exception as exc:
        raise HarnessLoadError(f"{type(exc).__name__}: {exc}") from exc
    cells, log_files = {}, []
    for suite, ids in batch.items():
        task = build_task(suite, ids, agent,
                          split="original" if SUITES[suite].impossible else None,
                          sandbox_type=sandbox)
        [log] = inspect_eval(task, model=model, log_dir=str(out / "logs" / suite),
                             fail_on_error=False, max_samples=max_samples,
                             max_sandboxes=max_sandboxes, display=display)
        log_files.append(log.location)
        samples = list(log.samples or [])
        seen, passed, errors = set(), 0, 0
        traces = out / "traces" / suite
        traces.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            sid = str(sample.id)
            invalid = sid not in ids or sid in seen
            seen.add(sid)
            h = (sample.metadata or {}).get("harness", {})
            score = (sample.scores or {}).get("pass")
            error = invalid or sample.error is not None or score is None or bool(h.get("agent_error"))
            errors += bool(error)
            passed += not error and float(score.value) == 1.0
            # Numbered filenames avoid interpreting dataset IDs as paths.
            (traces / f"{index:05d}.json").write_text(sample.model_dump_json(indent=2))
        errors += len(set(ids) - seen)
        if log.status != "success":
            errors = max(1, errors)
        cells[suite] = {"passed": passed, "n": len(ids), "errors": errors,
                        "reward": passed / len(ids), "status": log.status}
    result = {"harness_id": harness_id, "model": model, "batch": batch,
              "reward": sum(c["reward"] for c in cells.values()) / len(cells),
              "valid": all(c["errors"] == 0 for c in cells.values()),
              "suites": cells, "log_files": log_files}
    write_json(out / "reward.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("harness")
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--suites", nargs="+", choices=tuple(SUITES), default=["lcb_impossible"])
    p.add_argument("--batch-size", type=int, help="default: entire training pool")
    p.add_argument("--seed", type=int, default=329)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-sandboxes", type=int)
    args = p.parse_args()
    result = score_training(args.harness, args.model,
                            training_batch(args.suites, args.batch_size, args.seed), args.out_dir,
                            max_samples=args.max_samples, max_sandboxes=args.max_sandboxes)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
