"""A hand-written compiled kernel over raw Arrow buffers.

Both files are read into Arrow tables, and a Numba-compiled loop (parallel over
rows) compares the integer arrays and the variable-length string bytes directly.
No intermediate boolean arrays are created.

Deliberate limits, because the author of a kernel owns its correctness:
* only int64 and string columns are supported,
* nulls are not supported (a column with nulls raises rather than guessing).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange


@njit(parallel=True, cache=True)
def _count_matches(ints_a, ints_b, offs_a, data_a, offs_b, data_b):
    n = ints_a[0].shape[0]
    total = 0
    for i in prange(n):
        ok = True
        for c in range(len(ints_a)):
            if ints_a[c][i] != ints_b[c][i]:
                ok = False
                break
        if ok:
            for c in range(len(offs_a)):
                sa = offs_a[c][i]
                sb = offs_b[c][i]
                la = offs_a[c][i + 1] - sa
                if la != offs_b[c][i + 1] - sb:
                    ok = False
                    break
                for k in range(la):
                    if data_a[c][sa + k] != data_b[c][sb + k]:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            total += 1
    return total


def _view(buf: pa.Buffer, dtype, offset: int, length: int) -> np.ndarray:
    return np.frombuffer(buf, dtype=dtype)[offset : offset + length]


def _arrays(table: pa.Table):
    """Split a table into tuples of zero-copy NumPy views the kernel understands."""
    ints, offs, data = [], [], []
    for name in table.column_names:
        arr = table.column(name).combine_chunks()
        if arr.null_count:
            raise NotImplementedError(f"column {name!r} contains nulls")
        if pa.types.is_int64(arr.type):
            ints.append(_view(arr.buffers()[1], np.int64, arr.offset, len(arr)))
        elif pa.types.is_string(arr.type):
            offs.append(_view(arr.buffers()[1], np.int32, arr.offset, len(arr) + 1))
            data.append(np.frombuffer(arr.buffers()[2], dtype=np.uint8))
        else:
            raise NotImplementedError(f"unsupported column type {arr.type} ({name!r})")
    if not ints or not offs:
        raise NotImplementedError("kernel needs at least one int64 and one string column")
    return tuple(ints), tuple(offs), tuple(data)


def prepare() -> None:
    """Load (or compile) the kernel before the clock starts, using the benchmark schema."""
    from tabrecon.data import N_LETTER_COLS, _table

    tiny = _table(
        np.arange(1, 5, dtype=np.int64),
        np.full((4, N_LETTER_COLS), 65, dtype=np.uint8),
        np.arange(4, dtype=np.int64),
    )
    ints, offs, data = _arrays(tiny)
    _count_matches(ints, ints, offs, data, offs, data)


def run(path_a: str, path_b: str) -> float:
    t1, t2 = pq.read_table(path_a), pq.read_table(path_b)
    if t1.num_rows != t2.num_rows or t1.schema != t2.schema:
        raise ValueError("Files must have identical schema and row count")
    ints_a, offs_a, data_a = _arrays(t1)
    ints_b, offs_b, data_b = _arrays(t2)
    matched = _count_matches(ints_a, ints_b, offs_a, data_a, offs_b, data_b)
    return matched / t1.num_rows if t1.num_rows else 1.0
