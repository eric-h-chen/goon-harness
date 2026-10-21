"""Train / held-out task splits (specs/eval.md §3).

The split is generated once by ``python src/eval.py make-splits`` and committed.
Everything in ``data/splits/HELDOUT_EVAL_DO_NOT_TRAIN.json`` is for evaluation only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
HELDOUT_PATH = SPLITS_DIR / "HELDOUT_EVAL_DO_NOT_TRAIN.json"
TRAIN_PATH = SPLITS_DIR / "train.json"
LCB_SOURCES_PATH = SPLITS_DIR / "lcb_impossible_sources.json"

SEED = 329
IMPOSSIBLE_SPLITS = ("original", "oneoff", "conflicting")

SWE_VERIFIED = "princeton-nlp/SWE-bench_Verified"
SWE_VERIFIED_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"  # same pin as inspect_evals
IMPOSSIBLE_SWE = "fjzzq2002/impossible_swebench"
IMPOSSIBLE_LCB = "fjzzq2002/impossible_livecodebench"

# Held-out sizes (specs/eval.md §3.1).
N_HELDOUT_IMPOSSIBLE_SWE = 100
N_HELDOUT_EXTRA_SWE = 50  # from Verified instances not in ImpossibleBench
N_HELDOUT_IMPOSSIBLE_LCB = 50
N_HELDOUT_EXTRA_LCB = 100

# ImpossibleBench's generator kept hard problems whose decoded private tests are < 40k
# chars; we allow some slack because our length is measured after a JSON round-trip.
MATCH_MAX_TEST_CHARS = 45_000
AMBIGUOUS_TOP_K = 3
# Low-margin matches checked by hand on 2026-10-20 (all correct).
MATCHES_REVIEWED_BY_HAND = [
    "lcbhard_31", "lcbhard_32", "lcbhard_35", "lcbhard_47", "lcbhard_48", "lcbhard_67", "lcbhard_69",
]

Benchmark = Literal[
    "swebench_verified", "impossible_swebench", "livecodebench", "impossible_livecodebench"
]
BENCHMARKS: tuple[Benchmark, ...] = (
    "swebench_verified", "impossible_swebench", "livecodebench", "impossible_livecodebench"
)


class HeldOutLeakError(RuntimeError):
    """A held-out evaluation task was about to be used for training."""


# --------------------------------------------------------------------------- public API


@lru_cache(maxsize=1)
def _heldout() -> dict:
    if not HELDOUT_PATH.exists():
        raise FileNotFoundError(f"{HELDOUT_PATH} missing. Run `python src/eval.py make-splits`.")
    return json.loads(HELDOUT_PATH.read_text())


@lru_cache(maxsize=1)
def _train() -> dict:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_PATH} missing. Run `python src/eval.py make-splits`.")
    return json.loads(TRAIN_PATH.read_text())


@lru_cache(maxsize=1)
def _lcb_sources() -> dict[str, list[str]]:
    """lcbhard_N -> every question_id it may have come from (1 unless ambiguous)."""
    entries = json.loads(LCB_SOURCES_PATH.read_text())["matches"]
    return {
        tid: (m["candidates"] if m["ambiguous"] else [m["question_id"]])
        for tid, m in entries.items()
    }


def heldout_ids(benchmark: Benchmark) -> list[str]:
    return list(_heldout()[benchmark])


def train_ids(benchmark: Benchmark) -> list[str]:
    return list(_train()[benchmark])


@lru_cache(maxsize=1)
def _all_heldout() -> frozenset[str]:
    return frozenset(i for b in BENCHMARKS for i in _heldout()[b])


def assert_not_heldout(ids: Iterable[str]) -> None:
    """Raise HeldOutLeakError if any ID (instance_id, question_id or lcbhard_*) is held out.

    For lcbhard_* IDs this also checks the LiveCodeBench problem(s) it was derived from.
    """
    heldout, sources = _all_heldout(), _lcb_sources()
    leaks = []
    for i in ids:
        if i in heldout:
            leaks.append(i)
        elif any(src in heldout for src in sources.get(i, ())):
            leaks.append(f"{i} (derived from held-out {sources[i]})")
    if leaks:
        raise HeldOutLeakError(f"{len(leaks)} held-out task(s) used for training: {leaks[:20]}")


def split_file_hash() -> str:
    return hashlib.sha256(HELDOUT_PATH.read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- generation


def _stratified_sample(ids: Iterable[str], key: dict[str, str], n: int, rng: random.Random) -> list[str]:
    """Sample n IDs with per-stratum quotas proportional to stratum size (largest remainder)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for i in sorted(ids):
        groups[key[i]].append(i)
    total = sum(len(v) for v in groups.values())
    if n > total:
        raise ValueError(f"cannot sample {n} of {total}")
    exact = {k: n * len(v) / total for k, v in groups.items()}
    quota = {k: math.floor(q) for k, q in exact.items()}
    by_remainder = sorted(groups, key=lambda k: (-(exact[k] - quota[k]), k))
    for k in by_remainder[: n - sum(quota.values())]:
        quota[k] += 1
    picks = []
    for k in sorted(groups):
        picks += rng.sample(groups[k], quota[k])
    return sorted(picks)


def _assign(task_texts: list[str], cand_texts: list[str], **tfidf) -> tuple:
    """TF-IDF cosine + Hungarian. Returns (sim matrix, assigned column per task)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(sublinear_tf=True, **tfidf).fit(task_texts + cand_texts)
    sim = (vec.transform(task_texts) @ vec.transform(cand_texts).T).toarray()
    rows, cols = linear_sum_assignment(-sim)
    return sim, np.asarray(cols)[np.argsort(rows)]


def match_lcb_sources(lcb_tasks: dict[str, dict], index: dict[str, dict]) -> dict:
    """Recover lcbhard_N -> question_id (§3.2).

    lcb_tasks: lcbhard id -> ImpossibleBench row (``prompt``, ``original_test``).
    index: release_v6 rows by question_id. Candidates are the hard problems with small
    private tests, as in ImpossibleBench's generator. Each task's text (prompt + original
    tests, whose literals come from the source's test cases) is matched against each
    candidate's statement + public tests, independently with word and character TF-IDF,
    each followed by a Hungarian assignment. A task is ambiguous if the two disagree.
    """
    import numpy as np

    task_ids = sorted(lcb_tasks, key=lambda t: int(t.split("_")[-1]))
    cand_ids = sorted(
        q for q, r in index.items()
        if r["difficulty"] == "hard" and r["private_tests_chars"] < MATCH_MAX_TEST_CHARS
    )
    task_texts = [lcb_tasks[t]["prompt"] + "\n" + lcb_tasks[t]["original_test"] for t in task_ids]
    cand_texts = [
        index[q]["question_content"] + "\n"
        + "\n".join(c["input"] + "\n" + c["output"] for c in index[q]["public_test_cases"])
        for q in cand_ids
    ]
    sim, word = _assign(task_texts, cand_texts, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    _, char = _assign(task_texts, cand_texts, analyzer="char_wb", ngram_range=(3, 5))

    matches = {}
    for r, t in enumerate(task_ids):
        c = word[r]
        order = np.argsort(-sim[r])
        runner_up = next(j for j in order if j != c)
        ambiguous = bool(word[r] != char[r])
        candidates = {cand_ids[c]}
        if ambiguous:
            candidates |= {cand_ids[char[r]]} | {cand_ids[j] for j in order[:AMBIGUOUS_TOP_K]}
        matches[t] = {
            "question_id": cand_ids[c],
            "title": index[cand_ids[c]]["question_title"],
            "score": round(float(sim[r, c]), 4),
            "runner_up": cand_ids[runner_up],
            "margin": round(float(sim[r, c] - sim[r, runner_up]), 4),
            "ambiguous": ambiguous,
            "candidates": sorted(candidates),
        }
    return {
        "method": "word + char TF-IDF cosine, Hungarian assignment each; ambiguous if they disagree",
        "candidate_filter": f"difficulty == hard and private_tests_chars < {MATCH_MAX_TEST_CHARS}",
        "n_candidates": len(cand_ids),
        "reviewed_by_hand": MATCHES_REVIEWED_BY_HAND,
        "matches": matches,
    }


def _hf_sha(repo: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo).sha


def _load_ids(repo: str, field: str, revision: str | None = None) -> dict[str, dict[str, dict]]:
    """split -> id -> row for every ImpossibleBench split (or Verified's test split)."""
    from datasets import load_dataset

    splits = IMPOSSIBLE_SPLITS if repo != SWE_VERIFIED else ("test",)
    return {s: {r[field]: r for r in load_dataset(repo, split=s, revision=revision)} for s in splits}


def make_splits(force: bool = False) -> dict:
    """Generate and write the split files. Refuses to overwrite unless ``force``."""
    import lcb_real

    if HELDOUT_PATH.exists() and not force:
        raise FileExistsError(
            f"{HELDOUT_PATH} already exists. Splits must not change once experiments start; "
            "pass --force only if you are sure."
        )
    rng = random.Random(SEED)

    # ---- SWE-bench
    verified = _load_ids(SWE_VERIFIED, "instance_id", SWE_VERIFIED_REVISION)["test"]
    ib_swe = _load_ids(IMPOSSIBLE_SWE, "instance_id")
    ib_swe_ids = set(ib_swe["original"])
    assert all(set(v) == ib_swe_ids for v in ib_swe.values()), "impossible_swebench splits differ"
    assert ib_swe_ids <= set(verified), "impossible_swebench has IDs outside SWE-bench Verified"
    repo_of = {i: r["repo"] for i, r in verified.items()}
    held_ib_swe = _stratified_sample(ib_swe_ids, repo_of, N_HELDOUT_IMPOSSIBLE_SWE, rng)
    held_extra_swe = _stratified_sample(set(verified) - ib_swe_ids, repo_of, N_HELDOUT_EXTRA_SWE, rng)
    held_swe = sorted(set(held_ib_swe) | set(held_extra_swe))

    # ---- LiveCodeBench
    ib_lcb = _load_ids(IMPOSSIBLE_LCB, "task_id")
    ib_lcb_ids = set(ib_lcb["original"])
    assert all(set(v) == ib_lcb_ids for v in ib_lcb.values()), "impossible_livecodebench splits differ"
    index = lcb_real.load_index()
    sources = match_lcb_sources(ib_lcb["original"], index)
    srcs = {t: m["candidates"] for t, m in sources["matches"].items()}

    held_lcbhard = sorted(rng.sample(sorted(ib_lcb_ids), N_HELDOUT_IMPOSSIBLE_LCB))
    train_lcbhard = sorted(ib_lcb_ids - set(held_lcbhard))
    held_src = {q for t in held_lcbhard for q in srcs[t]}
    # A train-side task whose possible sources overlap held-out problems would leak; drop it.
    excluded_lcbhard = sorted(t for t in train_lcbhard if set(srcs[t]) & held_src)
    train_lcbhard = [t for t in train_lcbhard if t not in excluded_lcbhard]
    train_src = {q for t in train_lcbhard for q in srcs[t]}
    difficulty = {q: r["difficulty"] for q, r in index.items()}
    pool = set(index) - held_src - train_src
    held_lcb = sorted(held_src | set(_stratified_sample(pool, difficulty, N_HELDOUT_EXTRA_LCB, rng)))

    heldout = {
        "_README": "EVALUATION ONLY. Never train on, score candidates on, or show the proposer "
        "any task listed here, in any variant. See specs/eval.md §3.",
        "seed": SEED,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            SWE_VERIFIED: SWE_VERIFIED_REVISION,
            IMPOSSIBLE_SWE: _hf_sha(IMPOSSIBLE_SWE),
            IMPOSSIBLE_LCB: _hf_sha(IMPOSSIBLE_LCB),
            lcb_real.LCB_REPO: lcb_real.LCB_REVISION + " (release_v6)",
        },
        "swebench_verified": held_swe,
        "impossible_swebench": held_ib_swe,
        "livecodebench": held_lcb,
        "impossible_livecodebench": held_lcbhard,
    }
    train = {
        "seed": SEED,
        "swebench_verified": sorted(set(verified) - set(held_swe)),
        "impossible_swebench": sorted(ib_swe_ids - set(held_ib_swe)),
        "livecodebench": sorted(set(index) - set(held_lcb)),
        "impossible_livecodebench": train_lcbhard,
        "excluded_impossible_livecodebench": excluded_lcbhard,
    }

    for b in BENCHMARKS:
        assert not set(heldout[b]) & set(train[b]), f"{b}: train and held-out overlap"
    assert not {q for t in train_lcbhard for q in srcs[t]} & set(held_lcb), "lcbhard source leak"
    assert set(held_ib_swe) <= set(held_swe)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    LCB_SOURCES_PATH.write_text(json.dumps(sources, indent=2) + "\n")
    HELDOUT_PATH.write_text(json.dumps(heldout, indent=2) + "\n")
    TRAIN_PATH.write_text(json.dumps(train, indent=2) + "\n")
    for f in (_heldout, _train, _lcb_sources, _all_heldout):
        f.cache_clear()
    return {b: (len(heldout[b]), len(train[b])) for b in BENCHMARKS} | {
        "excluded_impossible_livecodebench": len(excluded_lcbhard),
        "ambiguous_lcb_matches": sum(m["ambiguous"] for m in sources["matches"].values()),
    }
