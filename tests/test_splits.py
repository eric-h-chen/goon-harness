import json
import random
from collections import Counter

import pytest

import splits
from splits import BENCHMARKS, HeldOutLeakError, assert_not_heldout, heldout_ids, train_ids


def test_sizes_match_spec():
    assert len(heldout_ids("swebench_verified")) == 150
    assert len(heldout_ids("impossible_swebench")) == 100
    assert len(heldout_ids("livecodebench")) == 150
    assert len(heldout_ids("impossible_livecodebench")) == 50
    assert len(heldout_ids("swebench_verified")) + len(train_ids("swebench_verified")) == 500


@pytest.mark.parametrize("bench", BENCHMARKS)
def test_train_and_heldout_disjoint(bench):
    assert not set(heldout_ids(bench)) & set(train_ids(bench))


def test_impossible_heldout_nested_in_real_heldout():
    assert set(heldout_ids("impossible_swebench")) <= set(heldout_ids("swebench_verified"))
    sources = json.loads(splits.LCB_SOURCES_PATH.read_text())["matches"]
    held_src = {sources[t]["question_id"] for t in heldout_ids("impossible_livecodebench")}
    assert held_src <= set(heldout_ids("livecodebench"))
    train_src = {sources[t]["question_id"] for t in train_ids("impossible_livecodebench")}
    assert not train_src & set(heldout_ids("livecodebench"))


def test_lcb_sources_are_one_to_one():
    matches = json.loads(splits.LCB_SOURCES_PATH.read_text())["matches"]
    assert len(matches) == 103
    assert len({m["question_id"] for m in matches.values()}) == 103


def test_assert_not_heldout():
    assert_not_heldout(train_ids("swebench_verified") + train_ids("livecodebench")
                       + train_ids("impossible_livecodebench"))
    for bench in BENCHMARKS:
        with pytest.raises(HeldOutLeakError):
            assert_not_heldout([heldout_ids(bench)[0]])


def test_swe_heldout_is_stratified_by_repo():
    from datasets import load_dataset

    ib = load_dataset(splits.IMPOSSIBLE_SWE, split="original")
    repo = {r["instance_id"]: r["repo"] for r in ib}
    held = Counter(repo[i] for i in heldout_ids("impossible_swebench"))
    total = Counter(repo.values())
    for r, n in total.items():
        assert abs(held[r] - n * 100 / len(repo)) < 1


def test_stratified_sample_is_deterministic_and_proportional():
    ids = [f"x{i}" for i in range(100)]
    key = {i: ("a" if int(i[1:]) < 70 else "b") for i in ids}
    s1 = splits._stratified_sample(ids, key, 10, random.Random(0))
    s2 = splits._stratified_sample(ids, key, 10, random.Random(0))
    assert s1 == s2
    assert Counter(key[i] for i in s1) == {"a": 7, "b": 3}


def test_make_splits_refuses_to_overwrite():
    with pytest.raises(FileExistsError):
        splits.make_splits()
