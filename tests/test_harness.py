"""The harness must isolate, measure and verify a run."""
from tabrecon.data import generate
from tabrecon.harness.runner import run_once
from tabrecon.methods.registry import BY_KEY, METHODS


def test_run_once_measures_and_succeeds(tmp_path):
    truth = generate(tmp_path, 300_000, change_fraction=0.01, row_group_size=50_000)
    r = run_once("pyarrow", tmp_path / "a.parquet", tmp_path / "b.parquet")
    assert r["status"] == "ok"
    assert abs(r["match_rate"] - truth["expected_match_rate"]) < 0.5 / truth["n_rows"]
    # Large enough that Windows' ~15 ms CPU-timer tick cannot round the CPU time to zero.
    assert r["wall_s"] > 0 and r["cpu_s"] > 0 and r["peak_mem_mb"] > 0


def test_parallel_method_reports_child_processes(tmp_path):
    generate(tmp_path, 20_000, change_fraction=0.01, row_group_size=5_000)
    r = run_once("parallel_row_groups", tmp_path / "a.parquet", tmp_path / "b.parquet", {"workers": 2})
    assert r["status"] == "ok"
    assert r["n_child_processes"] >= 2


def test_memory_guard_stops_a_run(tmp_path):
    generate(tmp_path, 20_000, change_fraction=0.01, row_group_size=5_000)
    r = run_once("pandas", tmp_path / "a.parquet", tmp_path / "b.parquet", mem_cap_bytes=10 * 2**20)
    assert r["status"] == "memory_guard"


def test_failure_is_reported_not_raised(tmp_path):
    r = run_once("pyarrow", tmp_path / "missing_a.parquet", tmp_path / "missing_b.parquet")
    assert r["status"] == "error"


def test_registry_is_complete_and_unique():
    keys = [m.key for m in METHODS]
    assert len(keys) == len(set(keys)) == len(BY_KEY)
    assert sum(m.family == "new" for m in METHODS) == 3
    assert all(m.lines_of_code > 0 for m in METHODS)
