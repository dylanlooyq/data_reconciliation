"""Compare Parquet row groups concurrently in a pool of worker processes."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

from tabrecon.methods.pyarrow_batches import count_true, row_equal

# Per-process cache of open files, so a worker opens each file once.
_OPEN: dict[str, pq.ParquetFile] = {}


def _file(path: str) -> pq.ParquetFile:
    pf = _OPEN.get(path)
    if pf is None:
        pf = _OPEN[path] = pq.ParquetFile(path)
    return pf


def _compare_group(args: tuple[str, str, int]) -> tuple[int, int]:
    path_a, path_b, g = args
    # use_threads=False: parallelism comes from the pool, not from inside Arrow.
    t1 = _file(path_a).read_row_group(g, use_threads=False)
    t2 = _file(path_b).read_row_group(g, use_threads=False)
    return count_true(row_equal(t1, t2)), t1.num_rows


def run(path_a: str, path_b: str, workers: int | None = None) -> float:
    meta_a, meta_b = pq.ParquetFile(path_a).metadata, pq.ParquetFile(path_b).metadata
    if meta_a.num_rows != meta_b.num_rows or meta_a.num_row_groups != meta_b.num_row_groups:
        raise ValueError("Files must have the same number of rows and row groups")

    n_groups = meta_a.num_row_groups
    workers = max(1, min(workers or os.cpu_count() or 1, n_groups))
    tasks = [(path_a, path_b, g) for g in range(n_groups)]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_compare_group, tasks))

    matched = sum(m for m, _ in results)
    total = sum(n for _, n in results)
    return matched / total if total else 1.0
