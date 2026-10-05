import time
import polars as pl
from utils_results import load_results, save_results

# --- Load persisted results dict (robust to empty/corrupt pickle) ---
results = load_results()

# --- Start timer ---
start_time = time.time()

# Lazily stream from Parquet (no pandas conversion)
l1 = pl.scan_parquet("data.parquet")
l2 = pl.scan_parquet("data_modified.parquet")

# Get column names without forcing full resolution
cols1 = l1.collect_schema().names()
cols2 = l2.collect_schema().names()
# Use overlapping columns (preserves l1 order)
cols = [c for c in cols1 if c in cols2]
if not cols:
    raise ValueError("No overlapping columns between the two Parquet files.")

# Row-level hash via struct hashing (version-proof-ish; seed pins current version)
h1 = l1.select(pl.struct(cols).hash(seed=0).alias("h"))
h2 = l2.select(pl.struct(cols).hash(seed=0).alias("h"))

# Attach a row index to preserve order, then join on index
h1i = h1.with_row_index("rn")
h2i = h2.with_row_index("rn")

# Join and compute match rate; use the non-deprecated engine kw
match_rate = (
    h1i.join(h2i, on="rn", how="inner", suffix="_right")
       .with_columns((pl.col("h") == pl.col("h_right")).alias("row_match"))
       .select(pl.col("row_match").mean())
       .collect(engine="streaming")
       .item()
)

elapsed_time = time.time() - start_time

print(f"Row-level match rate: {match_rate:.10f}")
print(f"Elapsed: {elapsed_time:.3f}s")

# --- Persist results ---
results["Polars\n(Streaming)"] = {
    "match_rate": round(match_rate, 10),
    "time_taken_sec": round(elapsed_time, 10),
}
save_results(results)

print(results)
