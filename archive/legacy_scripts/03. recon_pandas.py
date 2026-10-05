import time
import pandas as pd
from utils_results import load_results, save_results

results = load_results()

# --- Start timer ---
start_time = time.time()

# Load data
df1 = pd.read_parquet("data.parquet")
df2 = pd.read_parquet("data_modified.parquet")

# Compute row-level match rate
match_rate = (df1 == df2).all(axis=1).mean()
print(f"Row-level match rate: {match_rate:.10f}")

# --- End timer ---
elapsed_time = time.time() - start_time

# --- Update results dictionary ---
results["Pandas"] = {
    "match_rate": round(match_rate, 10),
    "time_taken_sec": round(elapsed_time, 10)
}

# --- Save updated results ---
save_results(results)
print(results)
