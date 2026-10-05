"""Skip decoding row groups whose raw Parquet bytes are identical.

Two row groups with byte-identical column chunks necessarily hold identical
data, so the byte comparison can only cause *extra* work (a false "differs"),
never a wrong answer. It is a shortcut that depends on both files having been
written with the same layout and settings; if they were not, every group is
decoded and the method degrades to a plain PyArrow comparison.
"""
from __future__ import annotations

import pyarrow.parquet as pq

from tabrecon.methods.pyarrow_batches import count_true, row_equal


def _byte_range(rg) -> tuple[int, int]:
    """File byte range [start, end) covering all column chunks of a row group."""
    start, end = None, 0
    for c in range(rg.num_columns):
        col = rg.column(c)
        first = min(o for o in (col.dictionary_page_offset, col.data_page_offset) if o)
        start = first if start is None else min(start, first)
        end = max(end, first + col.total_compressed_size)
    return start, end


def _same_layout(pf1: pq.ParquetFile, pf2: pq.ParquetFile) -> bool:
    m1, m2 = pf1.metadata, pf2.metadata
    return (
        pf1.schema_arrow.equals(pf2.schema_arrow)
        and m1.num_row_groups == m2.num_row_groups
        and all(m1.row_group(g).num_rows == m2.row_group(g).num_rows for g in range(m1.num_row_groups))
    )


def run(path_a: str, path_b: str) -> float:
    pf1, pf2 = pq.ParquetFile(path_a), pq.ParquetFile(path_b)
    if pf1.metadata.num_rows != pf2.metadata.num_rows:
        raise ValueError("Total row count mismatch")

    comparable = _same_layout(pf1, pf2)
    total = matched = 0
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        for g in range(pf1.metadata.num_row_groups):
            n = pf1.metadata.row_group(g).num_rows
            total += n
            if comparable:
                sa, ea = _byte_range(pf1.metadata.row_group(g))
                sb, eb = _byte_range(pf2.metadata.row_group(g))
                fa.seek(sa)
                fb.seek(sb)
                if fa.read(ea - sa) == fb.read(eb - sb):
                    matched += n  # identical bytes: no decoding needed
                    continue
            matched += count_true(row_equal(pf1.read_row_group(g), pf2.read_row_group(g)))
    return matched / total if total else 1.0
