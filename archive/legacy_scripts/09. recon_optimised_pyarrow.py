import time
import pickle
from pathlib import Path
from functools import reduce
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from utils_results import load_results, save_results

# --- Load/validate tuned batch size from pyarrow.pkl ---
def load_tuned_batch_size(path: Path = Path("pyarrow.pkl"), default: int = 131_072) -> int:
    """
    Load an integer batch size from pyarrow.pkl with robust fallbacks.
    Accepts either:
      - an int directly, or
      - a dict containing one of the keys:
        ['optimal_batch_size', 'batch_size', 'best_batch', 'tuned_batch_size'].
    """
    if not path.exists() or path.stat().st_size == 0:
        print(f"[warn] {path.name} not found/empty. Using default batch size = {default}.")
        return default

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"[warn] Failed to read {path.name} ({type(e).__name__}). Using default batch size = {default}.")
        return default

    if isinstance(obj, int) and obj > 0:
        return obj

    if isinstance(obj, dict):
        for k in ["optimal_batch_size", "batch_size", "best_batch", "tuned_batch_size"]:
            if k in obj and isinstance(obj[k], int) and obj[k] > 0:
                return obj[k]

    print(f"[warn] {path.name} did not contain a valid positive int. Using default batch size = {default}.")
    return default


# --- Load persisted results dict (robust) ---
results = load_results()

# --- Config ---
path1 = "data.parquet"
path2 = "data_modified.parquet"
batch_size = load_tuned_batch_size()   # <-- pulled from pyarrow.pkl (with fallback)
num_runs = 5

times = []
match_rate = None  # to store final match rate (same across runs)

for _ in range(num_runs):
    start_time = time.time()

    pf1, pf2 = pq.ParquetFile(path1), pq.ParquetFile(path2)

    # Fast fail if total row counts differ
    if pf1.metadata.num_rows != pf2.metadata.num_rows:
        raise ValueError(f"Total row count mismatch: {pf1.metadata.num_rows} vs {pf2.metadata.num_rows}")

    total_rows = 0
    matched_rows = 0

    for b1, b2 in zip(
        pf1.iter_batches(batch_size=batch_size),
        pf2.iter_batches(batch_size=batch_size)
    ):
        if b1.num_rows != b2.num_rows or b1.num_columns != b2.num_columns:
            raise ValueError("Batch shape mismatch")

        # Row-wise full equality: AND across all per-column equalities
        row_equal = reduce(
            pc.and_kleene,
            (pc.equal(b1.column(i), b2.column(i)) for i in range(b1.num_columns))
        )

        # Count True values in this batch
        matches_in_batch = int(pc.sum(pc.cast(row_equal, pa.int64())).as_py())
        matched_rows += matches_in_batch
        total_rows += b1.num_rows

    match_rate = matched_rows / total_rows if total_rows else 1.0
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)

avg_time = sum(times) / num_runs

# --- Update results dict & persist ---
results["PyArrow\n(Tuned batch size)"] = {
    "match_rate": round(match_rate, 10),
    "time_taken_sec": round(avg_time, 10),
    "batch_size": batch_size,  # record what was used (handy for plots)
}
save_results(results)

print(f"Row-level match rate: {match_rate:.10f}")
print(f"Average time over {num_runs} runs (batch_size={batch_size}): {avg_time:.6f} sec")
print(results)
