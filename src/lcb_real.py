"""Real LiveCodeBench (release_v6) as an Inspect task (specs/eval.md §4.4).

``inspect_evals`` has no LiveCodeBench task, so this module provides:
  * a local cache of the dataset (problem index + decoded private tests),
  * ``lcb_real``: an Inspect task over a given list of question_ids,
  * ``lcb_real_scorer``: runs the submitted program against public + private tests.

The dataset on the Hub is a loading-script dataset (unsupported by datasets>=4), so we
download its jsonl shards directly at a pinned revision (~4.5 GB in the HF cache).
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import re
import uuid
import zlib
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate
from inspect_ai.util import sandbox

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
INDEX_PATH = CACHE_DIR / "lcb_release_v6_index.jsonl"
TESTS_DIR = CACHE_DIR / "lcb_tests"

LCB_REPO = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILES = [f"test{i if i > 1 else ''}.jsonl" for i in range(1, 7)]  # release_v6

PER_TEST_TIMEOUT_S = 6  # LiveCodeBench's default


# --------------------------------------------------------------------------- data


class _StrOnlyUnpickler(pickle.Unpickler):
    """The private tests are a pickled str. Refuse anything that would import code."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def decode_private_tests(raw: str) -> list[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        blob = zlib.decompress(base64.b64decode(raw.encode("utf-8")))
        return json.loads(_StrOnlyUnpickler(io.BytesIO(blob)).load())


def iter_raw_rows() -> Iterator[dict]:
    """Yield raw release_v6 rows, downloading shards into the HF cache on first use."""
    from huggingface_hub import hf_hub_download

    for name in LCB_FILES:
        path = hf_hub_download(LCB_REPO, name, repo_type="dataset", revision=LCB_REVISION)
        with open(path) as f:
            for line in f:
                yield json.loads(line)


def build_cache(keep_tests_for: Iterable[str] | None = None) -> None:
    """Write the problem index and, for ``keep_tests_for``, the decoded private tests.

    The index holds everything except private tests, plus ``private_tests_chars``
    (length of the decoded test JSON; ImpossibleBench filtered on it).
    """
    keep = set(keep_tests_for or ())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    with open(tmp, "w") as out:
        for row in iter_raw_rows():
            private = decode_private_tests(row["private_test_cases"])
            qid = row["question_id"]
            out.write(json.dumps({
                "question_id": qid,
                "question_title": row["question_title"],
                "question_content": row["question_content"],
                "platform": row["platform"],
                "contest_date": row["contest_date"],
                "starter_code": row["starter_code"],
                "difficulty": row["difficulty"],
                "public_test_cases": json.loads(row["public_test_cases"]),
                "metadata": json.loads(row["metadata"] or "{}"),
                "private_tests_chars": len(json.dumps(private)),
            }) + "\n")
            if qid in keep:
                (TESTS_DIR / f"{qid}.json").write_text(json.dumps(private))
    tmp.replace(INDEX_PATH)


@lru_cache(maxsize=1)
def load_index() -> dict[str, dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{INDEX_PATH} missing. Run `python src/eval.py prepare-lcb` first."
        )
    with open(INDEX_PATH) as f:
        rows = [json.loads(line) for line in f]
    return {r["question_id"]: r for r in rows}


def load_tests(question_id: str) -> list[dict]:
    """Public + private tests for one problem."""
    path = TESTS_DIR / f"{question_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached private tests for {question_id}. Run `python src/eval.py prepare-lcb`."
        )
    return load_index()[question_id]["public_test_cases"] + json.loads(path.read_text())


# --------------------------------------------------------------------------- prompt

_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
_FORMAT_STARTER = (
    "### Format: You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters.\n```python\n{starter_code}\n```\n\n"
)
_FORMAT_STDIN = (
    "### Format: Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"
)


def format_prompt(row: dict) -> str:
    """LiveCodeBench's standard code-generation prompt."""
    fmt = (
        _FORMAT_STARTER.format(starter_code=row["starter_code"])
        if row["starter_code"]
        else _FORMAT_STDIN
    )
    return f"{_SYSTEM}\n\n### Question:\n{row['question_content']}\n\n{fmt}"


# --------------------------------------------------------------------------- task


@task
def lcb_real(question_ids: list[str], sandbox: str = "docker") -> Task:
    index = load_index()
    samples = []
    for qid in question_ids:
        row = index[qid]
        samples.append(Sample(
            id=qid,
            input=format_prompt(row),
            metadata={
                "question_id": qid,
                "difficulty": row["difficulty"],
                "platform": row["platform"],
                "starter_code": row["starter_code"] or None,
                "func_name": row["metadata"].get("func_name"),
            },
        ))
    return Task(
        name="lcb_real",
        dataset=MemoryDataset(samples, name="livecodebench_release_v6"),
        solver=generate(),
        scorer=lcb_real_scorer(),
        sandbox=sandbox,
    )


def find_code(completion: str) -> str:
    """Last fenced code block, or the whole text if there is none."""
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", completion, re.DOTALL)
    return blocks[-1] if blocks else completion


@scorer(metrics=[accuracy(), stderr()])
def lcb_real_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        code = find_code(state.output.completion if state.output else "")
        tests = load_tests(state.metadata["question_id"])
        result = await run_tests(code, tests, state.metadata.get("func_name"))
        return Score(
            value=CORRECT if result["passed"] else INCORRECT,
            answer=code,
            explanation=json.dumps(result)[:4000],
        )

    return score


async def run_tests(code: str, tests: list[dict], func_name: str | None) -> dict:
    sb = sandbox()
    root = f"/tmp/lcb_eval_{uuid.uuid4().hex[:12]}"  # outside the agent's workdir
    await sb.exec(["mkdir", "-p", root])
    await sb.write_file(f"{root}/solution.py", IMPORT_HEADER + code)
    await sb.write_file(f"{root}/runner.py", RUNNER)
    await sb.write_file(f"{root}/tests.json", json.dumps({"func_name": func_name, "cases": tests}))
    try:
        res = await sb.exec(
            ["python3", f"{root}/runner.py", root, str(PER_TEST_TIMEOUT_S)],
            timeout=min(900, 30 + PER_TEST_TIMEOUT_S * len(tests)),
            timeout_retry=False,
        )
    except TimeoutError:
        return {"passed": False, "reason": "runner timed out"}
    finally:
        await sb.exec(["rm", "-rf", root])
    lines = res.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return {"passed": False, "reason": "runner crashed", "stderr": res.stderr[-2000:]}


# The same convenience imports LiveCodeBench prepends to every solution.
IMPORT_HEADER = """\
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math, random
import statistics, itertools, functools, operator, io, sys, json
sys.setrecursionlimit(50000)
"""

# Runs inside the sandbox. Prints one JSON line: {"passed": bool, ...}. Stops at the
# first failing test. stdin problems compare whitespace-normalized lines, falling back
# to numeric comparison per token; functional problems compare JSON values.
RUNNER = r'''
import json, subprocess, sys
from decimal import Decimal, InvalidOperation

root, per_test = sys.argv[1], float(sys.argv[2])
spec = json.load(open(f"{root}/tests.json"))
func_name, cases = spec["func_name"], spec["cases"]

DRIVER = """
import json, sys
sys.path.insert(0, %r)
import solution
args = [json.loads(l) for l in sys.stdin.read().split("\\n") if l.strip()]
obj = solution.Solution() if hasattr(solution, "Solution") else solution
out = getattr(obj, %r)(*args)
print(json.dumps(out))
"""

def tokens_equal(a, b):
    if a == b:
        return True
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return False

def stdout_equal(got, want):
    g = [l.strip() for l in got.strip().splitlines()]
    w = [l.strip() for l in want.strip().splitlines()]
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for gl, wl in zip(g, w):
        gt, wt = gl.split(), wl.split()
        if len(gt) != len(wt) or not all(tokens_equal(x, y) for x, y in zip(gt, wt)):
            return False
    return True

def normalize(x):
    if isinstance(x, (list, tuple)):
        return [normalize(v) for v in x]
    return x

for i, case in enumerate(cases):
    functional = case.get("testtype") == "functional"
    cmd = [sys.executable, "-c", DRIVER % (root, func_name)] if functional else [sys.executable, f"{root}/solution.py"]
    try:
        p = subprocess.run(cmd, input=case["input"], capture_output=True, text=True, timeout=per_test)
    except subprocess.TimeoutExpired:
        print(json.dumps({"passed": False, "failed_test": i, "n_tests": len(cases), "reason": "timeout"}))
        sys.exit(0)
    if p.returncode != 0:
        ok, reason = False, "runtime error: " + p.stderr[-1000:]
    elif functional:
        try:
            ok = normalize(json.loads(p.stdout.strip().splitlines()[-1])) == normalize(json.loads(case["output"]))
        except Exception as e:
            ok = False
        reason = "wrong answer"
    else:
        ok, reason = stdout_equal(p.stdout, case["output"]), "wrong answer"
    if not ok:
        print(json.dumps({"passed": False, "failed_test": i, "n_tests": len(cases), "reason": reason}))
        sys.exit(0)
print(json.dumps({"passed": True, "n_tests": len(cases)}))
'''
