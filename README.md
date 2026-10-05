# Accelerating Tabular Reconciliation

*A benchmark of execution strategies for row-level comparison of large datasets.*

## The problem

A data analyst owns a month-end reconciliation: two systems each produce a file of the
same shape, and the question is how many rows agree. The files have grown from thousands
of rows to millions, the spreadsheet no longer opens, and the deadline has not moved.

This project treats that situation as a research question:

> For a row-by-row comparison of two large tables, which execution strategy gives the
> shortest time to an answer, and what does each one cost in memory, CPU and complexity?

Reconciliation is the worked example; the findings apply to any job that has to scan
large tables quickly.

## Methods compared

Seven methods from the first study, and three added in the second.

| # | Method | Idea | |
|---|---|---|---|
| 1 | Pandas | `(df1 == df2).all(axis=1)` | original |
| 2 | Polars | eager read, count all-true rows from Python tuples | original |
| 3 | DuckDB | SQL join on row number with null-safe equality | original |
| 4 | Polars (Streaming) | lazy scan, 64-bit row hash, streaming join | original |
| 5 | PyArrow | lock-step record batches with Arrow compute kernels | original |
| 6 | PyArrow (Tuned batch size) | same, batch size chosen by a sweep | original |
| 7 | Polars (Vectorized) | pure expression comparison, no Python loop | original |
| 8 | **PyArrow (Parallel row groups)** | one worker process per Parquet row group | new |
| 9 | **Row-group fingerprint** | compare raw bytes of each row group; decode only the groups that differ | new |
| 10 | **Numba kernel** | hand-written JIT-compiled loop over raw Arrow buffers | new |

Each method is one small module in [src/tabrecon/methods/](src/tabrecon/methods/), exposing
`run(path_a, path_b) -> match_rate`. Descriptions and trade-offs are in
[registry.py](src/tabrecon/methods/registry.py) and in the report.

## What is measured

Per method, per run, in a fresh subprocess:

- **Time**: wall-clock for reading both files and computing the answer.
- **Peak memory**: summed private memory of the process and any workers it starts.
- **CPU time**: CPU-seconds across all threads and processes (divided by wall time, the number of cores kept busy).
- **Code size**: lines of code, as a proxy for complexity.
- **Correctness**: every answer is checked against ground truth recorded by the data generator.

Studies: the main benchmark; scaling across dataset sizes (with a memory guard that records
methods that cannot finish); and sensitivity studies for PyArrow batch size, parallel worker
count, and how widely the differences are spread across row groups.

## Quick start

```bash
pip install -e .[dev]
python -m pytest                      # every method must agree with ground truth

python -m tabrecon all                # data + every study + report (tens of minutes)
```

Step by step:

```bash
python -m tabrecon batch-sweep        # tune PyArrow batch size first
python -m tabrecon benchmark --rows 7M --repeats 7
python -m tabrecon scaling --sizes 1M 7M 14M 28M 56M
python -m tabrecon worker-sweep
python -m tabrecon dirty-sweep
python -m tabrecon compile-cost
python -m tabrecon report
```

Before a full run, close memory-hungry applications (browsers in particular). The harness
stops any run that would exhaust the machine, but a machine that is already short of memory
produces noisy timings and spurious failures.

Generated data goes in `data/` (git-ignored, recreated on demand).

## Layout

```
README.md
pyproject.toml
src/tabrecon/
  data.py             synthetic workload generator that records ground truth
  methods/            the ten methods + registry (descriptions, trade-offs)
  harness/            isolated subprocess runs, resource sampling, memory guard
  experiments.py      benchmark, scaling, sensitivity studies
  plots.py            figures (light and dark)
  report.py           builds the report from raw results
  cli.py              python -m tabrecon ...
tests/                correctness against ground truth; harness behaviour
results/raw/          raw measurements (JSON, small enough to commit)
report/               report.html (self-contained), REPORT.md, figures/
archive/              the original numbered scripts, notebook and results, for reference
```

## Reading the results

Open [report/report.html](report/report.html) (self-contained, follows light/dark mode) or
[report/REPORT.md](report/REPORT.md). Every number in the report's prose is computed from
`results/raw/`, so re-running the studies and then `python -m tabrecon report` keeps the
document consistent with the data.

## Limitations

Synthetic data on one machine; positional (row *i* vs row *i*) comparison only; sparse
differences; Parquet inputs. See section 8 of the report for the full list.
