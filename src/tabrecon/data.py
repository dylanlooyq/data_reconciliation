"""Synthetic reconciliation workload with recorded ground truth.

Two parquet files with identical schema: ``a.parquet`` is random data and
``b.parquet`` is a copy in which a known set of rows has been altered. The
generator writes ``truth.json`` next to them so every method's answer can be
checked, not just timed.

Files are written one row group at a time, so generating 70M rows needs only
the memory of a single row group.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_LETTER_COLS = 10
LETTER_COLS = [f"L{i}" for i in range(1, N_LETTER_COLS + 1)]
VALUE_COL = "value"
ROW_COL = "row"

# The original experiment changed 200 of 7,000,000 rows. Keep that proportion so
# workloads of different sizes have the same share of breaks.
DEFAULT_CHANGE_FRACTION = 200 / 7_000_000
DEFAULT_ROW_GROUP_SIZE = 250_000


def _string_column(codes: np.ndarray) -> pa.Array:
    """Build an Arrow string array of one-letter strings from uint8 letter codes."""
    n = len(codes)
    offsets = np.arange(n + 1, dtype=np.int32)
    return pa.Array.from_buffers(
        pa.string(), n, [None, pa.py_buffer(offsets), pa.py_buffer(codes)]
    )


def _table(row_ids: np.ndarray, letters: np.ndarray, values: np.ndarray) -> pa.Table:
    cols = {ROW_COL: pa.array(row_ids)}
    for i, name in enumerate(LETTER_COLS):
        cols[name] = _string_column(np.ascontiguousarray(letters[:, i]))
    cols[VALUE_COL] = pa.array(values)
    return pa.table(cols)


def _empty_table() -> pa.Table:
    return _table(
        np.zeros(0, np.int64), np.zeros((0, N_LETTER_COLS), np.uint8), np.zeros(0, np.int64)
    )


def _choose_changed_rows(
    rng: np.random.Generator,
    n_rows: int,
    n_changes: int,
    row_group_size: int,
    dirty_groups: int | None,
) -> np.ndarray:
    """Pick the 0-based row positions to alter, sorted."""
    if n_changes == 0:
        return np.zeros(0, dtype=np.int64)
    if dirty_groups is None:
        return np.sort(rng.choice(n_rows, size=n_changes, replace=False))

    n_groups = -(-n_rows // row_group_size)
    dirty_groups = min(dirty_groups, n_groups, n_changes)
    groups = rng.choice(n_groups, size=dirty_groups, replace=False)
    bounds = [(g * row_group_size, min((g + 1) * row_group_size, n_rows)) for g in groups]
    # One change per selected group guarantees each of them is dirty ...
    picked = {int(rng.integers(lo, hi)) for lo, hi in bounds}
    # ... and the remainder land randomly inside the selected groups only.
    while len(picked) < n_changes:
        lo, hi = bounds[int(rng.integers(len(bounds)))]
        picked.add(int(rng.integers(lo, hi)))
    return np.array(sorted(picked), dtype=np.int64)


def generate(
    out_dir: Path,
    n_rows: int,
    change_fraction: float = DEFAULT_CHANGE_FRACTION,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    dirty_groups: int | None = None,
    seed: int = 42,
) -> dict:
    """Write ``a.parquet``, ``b.parquet`` and ``truth.json`` into ``out_dir``.

    ``dirty_groups`` confines every change to that many row groups (chosen at
    random). By default changes are scattered uniformly across the whole file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_changes = max(1, round(n_rows * change_fraction))
    if dirty_groups == 0:  # identical files
        n_changes = 0
    changed = _choose_changed_rows(rng, n_rows, n_changes, row_group_size, dirty_groups)
    n_changes = len(changed)

    schema = _empty_table().schema
    changed_ids: list[int] = []

    with pq.ParquetWriter(out_dir / "a.parquet", schema, compression="snappy") as wa, \
         pq.ParquetWriter(out_dir / "b.parquet", schema, compression="snappy") as wb:
        for start in range(0, n_rows, row_group_size):
            stop = min(start + row_group_size, n_rows)
            m = stop - start
            row_ids = np.arange(start + 1, stop + 1, dtype=np.int64)
            letters = rng.integers(65, 91, size=(m, N_LETTER_COLS), dtype=np.uint8)  # 'A'..'Z'
            values = rng.integers(1, 101, size=m, dtype=np.int64)
            wa.write_table(_table(row_ids, letters, values), row_group_size=m)

            letters_b, values_b = letters, values
            local = changed[(changed >= start) & (changed < stop)] - start
            if len(local):
                letters_b, values_b = letters.copy(), values.copy()
                letters_b[local] = rng.integers(65, 91, size=(len(local), N_LETTER_COLS), dtype=np.uint8)
                new_vals = rng.integers(1, 101, size=len(local), dtype=np.int64)
                same = new_vals == values[local]
                while same.any():  # force the value to differ so the row is certainly changed
                    new_vals[same] = rng.integers(1, 101, size=int(same.sum()))
                    same = new_vals == values[local]
                values_b[local] = new_vals
                changed_ids.extend((local + start + 1).tolist())
            wb.write_table(_table(row_ids, letters_b, values_b), row_group_size=m)

    truth = {
        "n_rows": n_rows,
        "n_changed": n_changes,
        "expected_match_rate": (n_rows - n_changes) / n_rows,
        "row_group_size": row_group_size,
        "n_row_groups": -(-n_rows // row_group_size),
        "dirty_groups": dirty_groups,
        "seed": seed,
        "changed_row_ids": changed_ids,
    }
    (out_dir / "truth.json").write_text(json.dumps(truth))
    return truth


def load_truth(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / "truth.json").read_text())
