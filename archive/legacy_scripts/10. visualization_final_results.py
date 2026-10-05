import pickle
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# --- Load results from pickle ---
RESULTS_PATH = Path("results.pkl")

if RESULTS_PATH.exists():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
else:
    raise FileNotFoundError("⚠️ 'results.pkl' not found. Run your analysis scripts first.")

# --- Extract methods and times ---
methods = list(results.keys())
times = [v['time_taken_sec'] for v in results.values()]

# --- Sort results by descending time ---
sorted_data = sorted(zip(methods, times), key=lambda x: x[1], reverse=True)
methods, times = zip(*sorted_data)

# --- Normalize for color scaling ---
max_time = max(times)
min_time = min(times)
norm_times = [(t - min_time) / (max_time - min_time) if max_time > min_time else 0.5 for t in times]
colors = [plt.cm.Blues(1 - n * 0.6) for n in norm_times]

# --- Plot ---
plt.figure(figsize=(9, 4))

# Add spacing between bars by reducing width and shifting x positions slightly
x = range(len(methods))
bar_width = 0.5  # smaller = more spacing
bars = plt.bar(x, times, width=bar_width, color=colors)

# Add padding at edges of chart
plt.xlim(-0.5, len(methods) - 0)  # adds space left/right
plt.margins(x=3)  # additional horizontal breathing room

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}s",
             ha='center', va='bottom', fontsize=10)

# Titles and labels
plt.title("Time to Reconcile Datasets")
# plt.xlabel("Reconciliation Method")
plt.xticks(x, methods)  # reapply method labels

# Remove gridlines and box spines
plt.grid(False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Remove y-axis completely
plt.gca().yaxis.set_visible(False)

plt.tight_layout()
plt.savefig("10. results.png")

print("Final results saved in '10. results.png'")
