"""Lock-step batch comparison with Arrow compute kernels.

``batch_size=None`` reads each file as a single batch (the untuned baseline).
``batch_size="tuned"`` uses the winner of the batch-size sweep, if one exists.
"""
from __future__ import annotations

import json
from functools import reduce

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tabrecon.paths import RESULTS_DIR

DEFAULT_TUNED_BATCH = 131_072


def tuned_batch_size() -> int:
    path = RESULTS_DIR / "batch_size_sweep.json"
    try:
        return int(json.loads(path.read_text())["best_batch_size"])
    except (OSError, KeyError, ValueError):
        return DEFAULT_TUNED_BATCH


def row_equal(b1: pa.RecordBatch | pa.Table, b2: pa.RecordBatch | pa.Table) -> pa.Array:
    """AND of the per-column equality masks. Null == null yields null, i.e. no match."""
    return reduce(
        pc.and_kleene,
        (pc.equal(b1.column(i), b2.column(i)) for i in range(b1.num_columns)),
    )


def count_true(mask) -> int:
    return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)


def run(path_a: str, path_b: str, batch_size: int | str | None = None) -> float:
    pf1, pf2 = pq.ParquetFile(path_a), pq.ParquetFile(path_b)
    if pf1.metadata.num_rows != pf2.metadata.num_rows:
        raise ValueError("Total row count mismatch")

    if batch_size == "tuned":
        batch_size = tuned_batch_size()
    elif batch_size is None:
        batch_size = max(pf1.metadata.num_rows, 1)

    total = matched = 0
    for b1, b2 in zip(pf1.iter_batches(batch_size=batch_size), pf2.iter_batches(batch_size=batch_size)):
        if b1.num_rows != b2.num_rows or b1.num_columns != b2.num_columns:
            raise ValueError("Batch shape mismatch")
        matched += count_true(row_equal(b1, b2))
        total += b1.num_rows
    return matched / total if total else 1.0
