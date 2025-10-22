import pickle
import time
import numpy as np
from functools import reduce

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import matplotlib
import matplotlib.pyplot as plt

path1 = "data.parquet"
path2 = "data_modified.parquet"

#### OPTIMISER
# Testing different batch sizes
batch_sizes = list(range(10_000, 400_001, 10_000))  # 10k → 1M in 10k steps
batch_times = []

for bs in batch_sizes:
    start_time = time.time()

    pf1, pf2 = pq.ParquetFile(path1), pq.ParquetFile(path2)

    total_rows = matched_rows = 0
    for b1, b2 in zip(pf1.iter_batches(batch_size=bs), pf2.iter_batches(batch_size=bs)):
        row_equal = reduce(
            pc.and_kleene,
            (pc.equal(b1.column(i), b2.column(i)) for i in range(b1.num_columns))
        )
        matched_rows += int(pc.sum(pc.cast(row_equal, pa.int64())).as_py())
        total_rows += b1.num_rows

    elapsed_time = time.time() - start_time
    batch_times.append(elapsed_time)
    print(f"Batch size {bs:,} → {elapsed_time:.4f} sec")

# Find optimal batch size (min time)
optimal_batch_size = batch_sizes[batch_times.index(min(batch_times))]
print(f"\n✅ Optimal batch size: {optimal_batch_size:,} rows\n")

# Save to file
with open("pyarrow.pkl", "wb") as f:
    pickle.dump(optimal_batch_size, f)

#### VISUALISATION
# --- Smoothing (rolling mean) with edge padding ---
window = max(5, len(batch_times) // 10)
if window % 2 == 0:
    window += 1
kernel = np.ones(window, dtype=float) / window

# pad by half-window on each side, then 'valid' to recenter
half = window // 2
padded = np.pad(batch_times, (half, half), mode='reflect')  # or mode='edge'
smoothed_times = np.convolve(padded, kernel, mode='valid')

# --- Plot results (raw + smoothed) ---
plt.figure(figsize=(9, 5))
plt.plot(batch_sizes, batch_times, marker='o', linewidth=1, label='Raw')
plt.plot(
    batch_sizes,
    smoothed_times,
    linewidth=2,
    linestyle='--',   # dotted or dashed line
    alpha=0.5,        # 50% opacity
    color = 'black',
    label=f'Trendline'
)
plt.title("Parquet Reconciliation Runtime vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Time Taken (seconds)")
plt.grid(True)
plt.legend()

# Remove box spines
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.savefig("09. pyarrow_batch_optimizer.png")


