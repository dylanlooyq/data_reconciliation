import time
import polars as pl
from utils_results import load_results, save_results

# --- Load existing results dict (robust to empty/corrupt) ---
results = load_results()

# --- Start timer ---
start_time = time.time()

# Read Parquet files using Polars
df1 = pl.read_parquet("data.parquet")
df2 = pl.read_parquet("data_modified.parquet")

# Compute row-level match rate
# (df1 == df2) gives a Boolean DataFrame
# .row(eq=True) checks if all values in the row are True
match_rate = (df1 == df2).rows().count((True,) * df1.width) / df1.height

# Print result
print(f"Row-level match rate: {match_rate:.10f}")

# --- End timer ---
elapsed_time = time.time() - start_time

# --- Update results dictionary ---
results["Polars"] = {
    "match_rate": round(match_rate, 10),
    "time_taken_sec": round(elapsed_time, 10)
}

# --- Save updated results ---
save_results(results)

# --- Print summary ---
print(results)
