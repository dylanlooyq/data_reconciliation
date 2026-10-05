"""Method metadata: names, descriptions and tradeoffs, in narrative order.

Kept free of heavy imports so the report can read it without loading pandas,
Polars, DuckDB or Numba. ``module`` is imported lazily by the harness worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

ORIGINAL, NEW = "original", "new"


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    family: str  # ORIGINAL (from the first study) or NEW (added in this study)
    module: str
    description: str
    tradeoff: str
    exact: str  # how trustworthy the answer is
    nulls: str  # how nulls are treated
    params: dict = field(default_factory=dict)

    @property
    def source_path(self) -> Path:
        return Path(import_module("tabrecon.paths").SRC).joinpath(*self.module.split(".")).with_suffix(".py")

    @property
    def lines_of_code(self) -> int:
        """Non-blank, non-comment, non-docstring-free lines: a rough size-of-solution proxy."""
        import ast

        tree = ast.parse(self.source_path.read_text())
        doc_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
        count = 0
        for i, line in enumerate(self.source_path.read_text().splitlines(), start=1):
            s = line.strip()
            if s and not s.startswith("#") and i not in doc_lines:
                count += 1
        return count


METHODS: list[Method] = [
    Method(
        key="pandas",
        label="Pandas",
        family=ORIGINAL,
        module="tabrecon.methods.pandas_eq",
        description=(
            "Loads both files into DataFrames and tests every cell with "
            "`(df1 == df2).all(axis=1)`. The approach most analysts would write first."
        ),
        tradeoff=(
            "Shortest and most familiar code. String columns are held as Python objects, which "
            "costs memory, and comparing them cell by cell is single-threaded."
        ),
        exact="Exact",
        nulls="NaN != NaN (null rows count as mismatches)",
    ),
    Method(
        key="polars",
        label="Polars",
        family=ORIGINAL,
        module="tabrecon.methods.polars_eager",
        description=(
            "Reads both files eagerly into Polars frames, compares them cell by cell, then counts the "
            "all-true rows by iterating Python tuples."
        ),
        tradeoff=(
            "Fast reader and comparison, but turning every row into a Python tuple hands back much of "
            "the gain."
        ),
        exact="Exact",
        nulls="null propagates (null rows are not counted as matches)",
    ),
    Method(
        key="duckdb",
        label="DuckDB",
        family=ORIGINAL,
        module="tabrecon.methods.duckdb_sql",
        description=(
            "Expresses the comparison as SQL: number the rows of each file, join on the row number and "
            "average a null-safe equality across every column."
        ),
        tradeoff=(
            "Declarative, multi-threaded and null-safe. The row-number join costs memory, and matching "
            "by position relies on the engine returning rows in file order."
        ),
        exact="Exact",
        nulls="null == null counts as a match (IS NOT DISTINCT FROM)",
    ),
    Method(
        key="polars_streaming",
        label="Polars (Streaming)",
        family=ORIGINAL,
        module="tabrecon.methods.polars_streaming",
        description=(
            "Lazily scans both files, reduces each row to a 64-bit hash and compares the two hash "
            "streams with a join on row position, using Polars' streaming engine."
        ),
        tradeoff=(
            "Built for bounded memory and larger-than-RAM inputs. Equality of hashes is probabilistic "
            "(a collision could hide a break) and the join adds overhead."
        ),
        exact="Probabilistic (64-bit row hash)",
        nulls="nulls hash to a fixed value, so null == null matches",
    ),
    Method(
        key="pyarrow",
        label="PyArrow",
        family=ORIGINAL,
        module="tabrecon.methods.pyarrow_batches",
        description=(
            "Walks the two files in lock-step record batches and combines per-column Arrow equality "
            "kernels. The batch size is the whole table, so each file is read in a single batch."
        ),
        tradeoff="Few moving parts and no dataframe layer, but a single huge batch means peak memory scales with the file.",
        exact="Exact",
        nulls="null == null is not a match",
        params={"batch_size": None},
    ),
    Method(
        key="pyarrow_tuned",
        label="PyArrow (Tuned batch size)",
        family=ORIGINAL,
        module="tabrecon.methods.pyarrow_batches",
        description=(
            "The same comparison with the batch size chosen by a sweep, trading per-batch overhead "
            "against cache locality and memory."
        ),
        tradeoff=(
            "Smaller batches keep memory flat and data cache-resident. The best size depends on the "
            "machine and the schema, so it has to be re-measured."
        ),
        exact="Exact",
        nulls="null == null is not a match",
        params={"batch_size": "tuned"},
    ),
    Method(
        key="polars_vectorized",
        label="Polars (Vectorized)",
        family=ORIGINAL,
        module="tabrecon.methods.polars_vectorized",
        description=(
            "Reads eagerly and compares the frames purely with Polars expressions (`==`, "
            "`all_horizontal`, `mean`), so no Python-level loop runs at all."
        ),
        tradeoff=(
            "Very short code with multi-threaded, SIMD-friendly execution. It materialises both tables "
            "plus a boolean frame of the same shape."
        ),
        exact="Exact",
        nulls="null propagates (null rows are not counted as matches)",
    ),
    Method(
        key="parallel_row_groups",
        label="PyArrow (Parallel row groups)",
        family=NEW,
        module="tabrecon.methods.parallel_row_groups",
        description=(
            "Splits the work by Parquet row group and compares groups concurrently in a pool of worker "
            "processes, each using single-threaded PyArrow kernels."
        ),
        tradeoff=(
            "Uses every core and bounds memory by row-group size. Each worker pays process start-up and "
            "its own library imports, and speed-up flattens once workers exceed physical cores."
        ),
        exact="Exact",
        nulls="null == null is not a match",
    ),
    Method(
        key="rowgroup_fingerprint",
        label="Row-group fingerprint",
        family=NEW,
        module="tabrecon.methods.rowgroup_fingerprint",
        description=(
            "Compares the raw compressed bytes of each Parquet row group across the two files and only "
            "decodes the groups whose bytes differ; identical groups are counted without being read as data."
        ),
        tradeoff=(
            "Potentially orders of magnitude faster when few row groups differ, but it needs both files "
            "written with the same layout and settings, and when differences touch every group it only "
            "adds a byte-comparison pass."
        ),
        exact="Exact (identical bytes imply identical data)",
        nulls="identical groups match including nulls; differing groups follow PyArrow rules",
    ),
    Method(
        key="numba_kernel",
        label="Numba kernel",
        family=NEW,
        module="tabrecon.methods.numba_kernel",
        description=(
            "Reads both files into Arrow buffers and runs a hand-written, JIT-compiled loop over the "
            "integer arrays and raw string offsets, parallelised across rows."
        ),
        tradeoff=(
            "Closest to the hardware, with no intermediate arrays. In exchange the author owns "
            "correctness: only int64 and string columns, no null handling, a one-off JIT compile and "
            "the most code to maintain."
        ),
        exact="Exact (for supported types)",
        nulls="not supported (raises)",
    ),
]

BY_KEY = {m.key: m for m in METHODS}


def get(key: str) -> Method:
    return BY_KEY[key]
