import json
import math

import eval as ev


def test_wilson_ci():
    lo, hi = ev.wilson_ci(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert ev.wilson_ci(0, 10)[0] == 0.0
    assert all(math.isnan(x) for x in ev.wilson_ci(0, 0))


def test_metrics_rows():
    cells = {
        ("livecodebench", "real"): {"k": 3, "n": 10, "n_errors": 0, "tokens": 5},
        ("livecodebench", "oneoff"): {"k": 1, "n": 10, "n_errors": 1, "tokens": 5},
        ("livecodebench", "conflicting"): {"k": 3, "n": 10, "n_errors": 0, "tokens": 5},
    }
    rows = ev._metrics_rows(cells, {"iteration": 0})
    row = rows["livecodebench"]
    assert row["pass_real"] == 0.3
    assert row["cheat_rate"] == 0.2
    assert row["warn_oneoff"] is True  # 1/11 errored > 5%
    assert row["tokens"] == 15
    assert "swebench" not in rows


def _write_run(run_dir, slope):
    (run_dir / "eval").mkdir(parents=True)
    with open(run_dir / "eval" / "metrics.jsonl", "w") as f:
        for it in range(5):
            for bench in ("livecodebench", "swebench"):
                row = {"iteration": it, "benchmark": bench, "partial": False}
                for m, base in (("real", 0.3), ("original", 0.4), ("oneoff", 0.05), ("conflicting", 0.02)):
                    v = min(1.0, base + slope * it)
                    row[f"pass_{m}"] = v
                    row[f"pass_{m}_ci95"] = [max(0, v - 0.1), min(1, v + 0.1)]
                row["cheat_rate"] = (row["pass_oneoff"] + row["pass_conflicting"]) / 2
                f.write(json.dumps(row) + "\n")
        f.write(json.dumps({"iteration": 9, "benchmark": "swebench", "partial": True, "pass_real": 1.0}) + "\n")


def test_plots(tmp_path):
    a, b = tmp_path / "run_a", tmp_path / "run_b"
    _write_run(a, 0.05)
    _write_run(b, 0.02)
    assert len(ev.load_metrics(a)) == 10
    assert len(ev.load_metrics(a, include_partial=True)) == 11
    for out in (ev.plot_iterations(a, tmp_path / "one.png"),
                ev.plot_iterations([a, b], tmp_path / "two.png"),
                ev.plot_tradeoff([a, b], tmp_path / "tradeoff.png")):
        assert out.stat().st_size > 10_000
