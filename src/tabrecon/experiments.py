"""The studies that make up the benchmark. Each writes JSON to ``results/raw/``.

benchmark      every method on one dataset, repeated and interleaved
scaling        every method across dataset sizes (records memory-guard failures)
batch_sweep    PyArrow runtime against batch size; saves the winner for "tuned"
worker_sweep   parallel row groups against worker count
dirty_sweep    row-group fingerprinting against how many row groups differ
compile_cost   one-off cost of the Numba kernel's first call
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil

from tabrecon import data
from tabrecon.harness.runner import run_once
from tabrecon.methods.registry import METHODS, get
from tabrecon.paths import RAW_DIR, RESULTS_DIR, SRC, dataset_dir

DEFAULT_ROWS = 7_000_000


# --------------------------------------------------------------------------- helpers
def environment() -> dict:
    import duckdb
    import numba
    import numpy
    import pandas
    import polars
    import pyarrow

    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(vm.total / 2**30, 1),
        "available_ram_gb_at_start": round(vm.available / 2**30, 1),
        "libraries": {
            "pandas": pandas.__version__, "polars": polars.__version__, "duckdb": duckdb.__version__,
            "pyarrow": pyarrow.__version__, "numba": numba.__version__, "numpy": numpy.__version__,
        },
    }


def memory_cap_bytes(cap_gb: float | None = None) -> int:
    """Per-run memory budget for a whole process tree: an explicit cap, else 75% of physical RAM.

    "Available" memory is deliberately not used: the OS reclaims it from idle applications on
    demand, so it understates what a run can get. A system-wide low-memory kill switch in the
    runner is the second safety net.
    """
    return int(cap_gb * 2**30 if cap_gb else 0.75 * psutil.virtual_memory().total)


def ensure_dataset(n_rows: int, tag: str = "", **gen_kwargs) -> Path:
    out = dataset_dir(n_rows, tag)
    if not (out / "truth.json").exists():
        print(f"  generating {n_rows:,} rows -> {out}", flush=True)
        data.generate(out, n_rows, **gen_kwargs)
    return out


def warm_file_cache(*paths: Path) -> None:
    """Read the files once so every method sees the same warm OS cache."""
    for p in paths:
        with open(p, "rb") as f:
            while f.read(64 * 2**20):
                pass


def save(name: str, meta: dict, runs: list[dict]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"meta": meta, "runs": runs}, indent=1))
    os.replace(tmp, path)
    return path


def run_matrix(
    variants: list[dict],
    ds: Path,
    repeats: int,
    mem_cap_gb: float | None = None,
    timeout_s: float = 900,
    on_row=None,
    skip: set[str] | None = None,
) -> list[dict]:
    """Run each variant ``repeats`` times, interleaved (round-robin) to spread any drift.

    A variant is ``{"id": str, "key": method_key, "params": {...}}``. A variant that fails
    is not retried in later rounds. Every successful answer is checked against ground truth.
    """
    truth = data.load_truth(ds)
    warm_file_cache(ds / "a.parquet", ds / "b.parquet")
    failed = set(skip or ())
    rows: list[dict] = []
    for rep in range(repeats):
        for v in variants:
            if v["id"] in failed:
                continue
            r = run_once(v["key"], ds / "a.parquet", ds / "b.parquet", v.get("params"),
                         mem_cap_bytes=memory_cap_bytes(mem_cap_gb), timeout_s=timeout_s)
            r.update(variant=v["id"], repeat=rep, n_rows=truth["n_rows"])
            if r["status"] == "ok":
                r["correct"] = abs(r["match_rate"] - truth["expected_match_rate"]) <= 0.5 / truth["n_rows"]
            else:
                failed.add(v["id"])
            rows.append(r)
            print(f"  [{rep + 1}/{repeats}] {v['id']:34s} {r['status']:12s} "
                  + (f"{r['wall_s']:8.3f}s  cpu {r['cpu_s']:7.2f}s  mem {r['peak_mem_mb']:8.0f}MB"
                     f"{'' if r.get('correct') else '  *** WRONG ANSWER ***'}" if r["status"] == "ok"
                     else f"{r.get('error', '')[:80]}"), flush=True)
            if on_row:
                on_row(rows)
    return rows


def _method_variants(keys: list[str] | None = None) -> list[dict]:
    return [{"id": m.key, "key": m.key, "params": {}} for m in METHODS if not keys or m.key in keys]


# --------------------------------------------------------------------------- studies
def benchmark(n_rows: int = DEFAULT_ROWS, repeats: int = 7, mem_cap_gb: float | None = None) -> Path:
    print(f"== benchmark: {n_rows:,} rows, {repeats} repeats")
    ds = ensure_dataset(n_rows)
    meta = {"experiment": "benchmark", "n_rows": n_rows, "repeats": repeats, "env": environment(),
            "truth": {k: v for k, v in data.load_truth(ds).items() if k != "changed_row_ids"}}
    rows = run_matrix(_method_variants(), ds, repeats, mem_cap_gb, on_row=lambda r: save("benchmark", meta, r))
    return save("benchmark", meta, rows)


def scaling(sizes: list[int], repeats: int = 3, mem_cap_gb: float | None = None) -> Path:
    print(f"== scaling: sizes {sizes}, {repeats} repeats")
    meta = {"experiment": "scaling", "sizes": sizes, "repeats": repeats, "env": environment()}
    all_rows: list[dict] = []
    failed: set[str] = set()
    for n in sizes:
        print(f"-- {n:,} rows")
        ds = ensure_dataset(n)
        rows = run_matrix(_method_variants(), ds, repeats, mem_cap_gb, skip=failed,
                          on_row=lambda r: save("scaling", meta, all_rows + r))
        failed |= {r["variant"] for r in rows if r["status"] != "ok"}
        all_rows += rows
        save("scaling", meta, all_rows)
    return save("scaling", meta, all_rows)


def batch_sweep(n_rows: int = DEFAULT_ROWS, repeats: int = 5, mem_cap_gb: float | None = None) -> Path:
    print(f"== batch size sweep: {n_rows:,} rows, {repeats} repeats")
    sizes = [10_000, 20_000, 40_000, 80_000, 131_072, 250_000, 500_000, 1_000_000, 2_000_000, 3_500_000, 7_000_000]
    sizes = [s for s in sizes if s <= n_rows]
    ds = ensure_dataset(n_rows)
    variants = [{"id": f"batch={s}", "key": "pyarrow", "params": {"batch_size": s}} for s in sizes]
    meta = {"experiment": "batch_sweep", "n_rows": n_rows, "repeats": repeats, "batch_sizes": sizes}
    rows = run_matrix(variants, ds, repeats, mem_cap_gb, on_row=lambda r: save("batch_sweep", meta, r))
    path = save("batch_sweep", meta, rows)

    import statistics
    med = {}
    for r in rows:
        if r["status"] == "ok":
            med.setdefault(r["params"]["batch_size"], []).append(r["wall_s"])
    best = min(med, key=lambda s: statistics.median(med[s]))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "batch_size_sweep.json").write_text(json.dumps(
        {"best_batch_size": best, "n_rows": n_rows,
         "median_wall_s": {str(s): statistics.median(v) for s, v in sorted(med.items())}}, indent=1))
    print(f"   best batch size: {best:,}")
    return path


def worker_sweep(n_rows: int = DEFAULT_ROWS, repeats: int = 5, mem_cap_gb: float | None = None) -> Path:
    print(f"== worker sweep: {n_rows:,} rows, {repeats} repeats")
    cpus = psutil.cpu_count() or 1
    counts = sorted({1, 2, 4, 6, 8, cpus})
    ds = ensure_dataset(n_rows)
    variants = [{"id": f"workers={w}", "key": "parallel_row_groups", "params": {"workers": w}} for w in counts]
    meta = {"experiment": "worker_sweep", "n_rows": n_rows, "repeats": repeats, "workers": counts,
            "env": environment()}
    rows = run_matrix(variants, ds, repeats, mem_cap_gb, on_row=lambda r: save("worker_sweep", meta, r))
    return save("worker_sweep", meta, rows)


def dirty_sweep(n_rows: int = DEFAULT_ROWS, repeats: int = 5, mem_cap_gb: float | None = None) -> Path:
    print(f"== dirty row-group sweep: {n_rows:,} rows, {repeats} repeats")
    n_groups = -(-n_rows // data.DEFAULT_ROW_GROUP_SIZE)
    levels = sorted({0, 1, 2, max(1, n_groups // 7), max(1, n_groups // 4), max(1, n_groups // 2), n_groups})
    meta = {"experiment": "dirty_sweep", "n_rows": n_rows, "repeats": repeats, "dirty_groups": levels,
            "n_row_groups": n_groups, "env": environment()}
    keys = ["rowgroup_fingerprint", "pyarrow_tuned", "polars_vectorized"]
    rows: list[dict] = []
    for g in levels:
        print(f"-- {g} of {n_groups} row groups differ")
        ds = ensure_dataset(n_rows, tag=f"_dirty{g}", dirty_groups=g)
        part = run_matrix(_method_variants(keys), ds, repeats, mem_cap_gb)
        for r in part:
            r["dirty_groups"] = g
        rows += part
        save("dirty_sweep", meta, rows)
    return save("dirty_sweep", meta, rows)


def compile_cost() -> Path:
    """Time ``prepare()`` with an empty and then a warm Numba cache."""
    import tempfile

    code = ("import time,sys; t=time.perf_counter(); "
            "from tabrecon.methods import numba_kernel as m; t_import=time.perf_counter()-t; "
            "t=time.perf_counter(); m.prepare(); print(t_import, time.perf_counter()-t)")
    runs = {}
    with tempfile.TemporaryDirectory() as cache:
        env = {**os.environ, "NUMBA_CACHE_DIR": cache,
               "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        for label in ("cold_cache", "warm_cache"):
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
            t_import, t_prepare = map(float, out.stdout.split())
            runs[label] = {"import_s": t_import, "first_call_s": t_prepare}
    print("== numba compile cost:", runs)
    return save("numba_compile", {"experiment": "numba_compile"}, [runs])
