"""The LiveCodeBench test runner, executed on the host (no sandbox needed)."""

import json
import subprocess
import sys

from lcb_real import IMPORT_HEADER, RUNNER, find_code


def run(tmp_path, code, cases, func_name=None, per_test=2):
    (tmp_path / "solution.py").write_text(IMPORT_HEADER + code)
    (tmp_path / "runner.py").write_text(RUNNER)
    (tmp_path / "tests.json").write_text(json.dumps({"func_name": func_name, "cases": cases}))
    out = subprocess.run([sys.executable, str(tmp_path / "runner.py"), str(tmp_path), str(per_test)],
                         capture_output=True, text=True, timeout=60)
    return json.loads(out.stdout.strip().splitlines()[-1])


STDIN_CASES = [
    {"input": "27\n", "output": "3\n", "testtype": "stdin"},
    {"input": "100\n", "output": "-1\n", "testtype": "stdin"},
]


def test_stdin_correct(tmp_path):
    code = "B = int(input())\nprint(next((a for a in range(1, 16) if a**a == B), -1))\n"
    assert run(tmp_path, code, STDIN_CASES)["passed"]


def test_stdin_wrong_and_whitespace(tmp_path):
    assert not run(tmp_path, "print(-1)\n", STDIN_CASES)["passed"]
    trailing = "B = int(input())\nprint(str(next((a for a in range(1, 16) if a**a == B), -1)) + '   ')\n"
    assert run(tmp_path, trailing, STDIN_CASES)["passed"]


def test_numeric_tokens_compare_by_value(tmp_path):
    cases = [{"input": "", "output": "0.50\n", "testtype": "stdin"}]
    assert run(tmp_path, "print(0.5)\n", cases)["passed"]


def test_timeout_and_runtime_error(tmp_path):
    assert run(tmp_path, "while True: pass\n", STDIN_CASES, per_test=1)["reason"] == "timeout"
    assert "runtime error" in run(tmp_path, "raise ValueError('x')\n", STDIN_CASES)["reason"]


FUNC_CASES = [
    {"input": '"leetcode"', "output": "true", "testtype": "functional"},
    {"input": '"abcd"', "output": "false", "testtype": "functional"},
]


def test_functional(tmp_path):
    good = (
        "class Solution:\n"
        "    def isSubstringPresent(self, s: str) -> bool:\n"
        "        r = s[::-1]\n"
        "        return any(s[i:i+2] in r for i in range(len(s) - 1))\n"
    )
    assert run(tmp_path, good, FUNC_CASES, "isSubstringPresent")["passed"]
    bad = "class Solution:\n    def isSubstringPresent(self, s):\n        return True\n"
    assert not run(tmp_path, bad, FUNC_CASES, "isSubstringPresent")["passed"]


def test_functional_multiple_args_and_tuples(tmp_path):
    cases = [{"input": "[1, 2]\n3", "output": "[1, 2, 3]", "testtype": "functional"}]
    code = "class Solution:\n    def f(self, xs, y):\n        return tuple(xs) + (y,)\n"
    assert run(tmp_path, code, cases, "f")["passed"]


def test_find_code():
    assert find_code("text\n```python\na = 1\n```\nmore\n```\nb = 2\n```") == "b = 2\n"
    assert find_code("plain") == "plain"
