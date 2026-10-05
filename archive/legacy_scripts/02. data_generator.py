# pip install pandas pyarrow numpy
import numpy as np
import pandas as pd
import string
from pathlib import Path
import shutil

np.random.seed(42)

N = 7_000_000
LETTER_COLS = [f"L{i}" for i in range(1, 11)]
VALUE_COL = "value"
ROW_COL = "row"

# 1) Generate base dataset
letters = np.array(list(string.ascii_uppercase))  # 26 uppercase letters

# random letters for the 10 columns, shape (N, 10)
letter_matrix = letters[np.random.randint(0, len(letters), size=(N, len(LETTER_COLS)))]

# random integer values [1, 100]
values = np.random.randint(1, 101, size=N)

df = pd.DataFrame(
    {ROW_COL: np.arange(1, N + 1, dtype=np.int64),
     **{col: letter_matrix[:, i] for i, col in enumerate(LETTER_COLS)},
     VALUE_COL: values}
)

# (Optional) make string columns Arrow-backed for memory/perf when writing
for col in LETTER_COLS:
    df[col] = df[col].astype("string[pyarrow]")

base_path = Path("data.parquet")
df.to_parquet(base_path, engine="pyarrow", compression="snappy", index=False)

# 2) Copy file and modify exactly 5 random rows (letters + value)
copy_path = Path("data_modified.parquet")
shutil.copy(base_path, copy_path)

# Load, mutate 5 rows, save back
df2 = pd.read_parquet(copy_path, engine="pyarrow")

# Pick X unique target row indices (0-based)
size = 200
change_idx = np.random.choice(df2.index.values, size=size, replace=False)

# Change letters: re-draw fresh random letters
new_letter_matrix = letters[np.random.randint(0, len(letters), size=(size, len(LETTER_COLS)))]
for i, col in enumerate(LETTER_COLS):
    df2.loc[change_idx, col] = new_letter_matrix[:, i]

# Change value: re-draw fresh ints in [1, 100]
# (Ensures a change by re-drawing any equal-to-original entries)
old_vals = df2.loc[change_idx, VALUE_COL].to_numpy()
new_vals = np.random.randint(1, 101, size=size)
mask_same = new_vals == old_vals
while mask_same.any():
    new_vals[mask_same] = np.random.randint(1, 101, size=mask_same.sum())
    mask_same = new_vals == old_vals

df2.loc[change_idx, VALUE_COL] = new_vals

# Save the modified copy
df2.to_parquet(copy_path, engine="pyarrow", compression="snappy", index=False)

print("Done.")
print("Changed row numbers:", df2.loc[change_idx, ROW_COL].to_list())
