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

# --- Logic from snippet 1: guards + null/NaN-safe, engine-only comparison ---

# Basic guard: row counts must match
if df1.height != df2.height:
    raise ValueError(f"Row count mismatch: {df1.height} vs {df2.height}")

# Align columns by name & order (preserve df1 order)
common_cols = [c for c in df1.columns if c in df2.columns]
if not common_cols:
    raise ValueError("No overlapping columns between the two Parquet files.")

lhs = df1.select(common_cols)
rhs = df2.select(common_cols).rename({c: f"{c}__r" for c in common_cols})

# Build NULL/NaN-safe equality expressions per column (stay in Polars engine)
eq_exprs = []
for c in common_cols:
    a = pl.col(c)
    b = pl.col(f"{c}__r")
    e = (a == b) | (a.is_null() & b.is_null())
    # treat NaN == NaN as equal for float columns
    if df1.schema[c] in (pl.Float32, pl.Float64):
        e = e | (a.is_nan() & b.is_nan())
    eq_exprs.append(e)

# Row matches if all columns match; take mean (engine-side)
match_rate = (
    pl.concat([lhs, rhs], how="horizontal")
      .select(pl.all_horizontal(eq_exprs).alias("row_match"))
      .select(pl.col("row_match").mean())
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