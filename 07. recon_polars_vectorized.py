import time
import polars as pl
from utils_results import load_results, save_results

# --- Load existing results dict (robust to empty/corrupt) ---
results = load_results()

# Start timer
start_time = time.time()

# Read Parquet files using Polars
df1 = pl.read_parquet("data.parquet")
df2 = pl.read_parquet("data_modified.parquet")

# assume both frames have identical schemas and column order
match_rate = (
    (df1 == df2)
    .select(pl.all_horizontal(pl.all()).alias("row_match"))   # row-wise AND across all columns
    .select(pl.col("row_match").mean().alias("match_rate"))   # average of booleans
    .to_series(0)
    .item()
)

# Print result
print(f"Row-level match rate: {match_rate:.10f}")

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

# --- Update results dictionary ---
results["Polars\n (Vectorized)"] = {
    "match_rate": round(match_rate, 10),
    "time_taken_sec": round(elapsed_time, 10)
}

# --- Save updated results ---
save_results(results)

# --- Print summary ---
print(results)

# 🔬 Why “vectorized” = fast
# No Python loop / GIL → runs natively in Rust/C++.
# Columnar memory layout → better CPU cache utilization.
# Batch processing (SIMD) → one CPU instruction handles many values.
# Threaded execution → Polars automatically parallelizes across cores.

# Every heavy operation (==, all_horizontal, mean) runs inside the Polars engine, which:
# avoids Python’s GIL,
# uses columnar Arrow buffers (contiguous memory),
# leverages SIMD CPU instructions,
# and parallelizes across cores automatically.