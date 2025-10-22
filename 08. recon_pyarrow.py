import time
from functools import reduce
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from utils_results import load_results, save_results

# --- Load existing results (robust to empty/corrupt pickle) ---
results = load_results()

# --- Config ---
path1 = "data.parquet"
path2 = "data_modified.parquet"
batch_size = 7_000_000  # untuned, batch size = table size
num_runs = 5

times = []
match_rate = None  # final match rate (same across runs)

for _ in range(num_runs):
    start_time = time.time()

    pf1, pf2 = pq.ParquetFile(path1), pq.ParquetFile(path2)

    # Fast fail if total row counts differ
    if pf1.metadata.num_rows != pf2.metadata.num_rows:
        raise ValueError(
            f"Total row count mismatch: {pf1.metadata.num_rows} vs {pf2.metadata.num_rows}"
        )

    total_rows = 0
    matched_rows = 0

    # Iterate files in lock-step by batches
    for b1, b2 in zip(
        pf1.iter_batches(batch_size=batch_size),
        pf2.iter_batches(batch_size=batch_size),
    ):
        if b1.num_rows != b2.num_rows or b1.num_columns != b2.num_columns:
            raise ValueError("Batch shape mismatch")

        # Row-wise full equality: AND across all per-column equalities
        # Note: pc.equal yields null for null==null; and_kleene preserves nulls.
        # Casting to int64 makes True->1, False->0, null->null; pc.sum ignores nulls.
        row_equal = reduce(
            pc.and_kleene,
            (pc.equal(b1.column(i), b2.column(i)) for i in range(b1.num_columns)),
        )

        matches_in_batch = int(pc.sum(pc.cast(row_equal, pa.int64())).as_py())
        matched_rows += matches_in_batch
        total_rows += b1.num_rows

    match_rate = matched_rows / total_rows if total_rows else 1.0
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)

avg_time = sum(times) / num_runs

# --- Update results dict & persist ---
results["PyArrow"] = {
    "match_rate": round(match_rate, 10),
    "time_taken_sec": round(avg_time, 10),
}
save_results(results)

print(f"Row-level match rate: {match_rate:.10f}")
print(f"Average time over {num_runs} runs: {avg_time:.6f} sec")
print(results)
