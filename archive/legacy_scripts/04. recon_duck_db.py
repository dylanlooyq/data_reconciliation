import time
import duckdb
from pathlib import Path
from utils_results import load_results, save_results

DATA1 = Path("data.parquet")
DATA2 = Path("data_modified.parquet")

# --- Helper: quote identifiers for DuckDB ---
def quote_ident(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'

# --- Load prior results (handles empty/corrupt via utils) ---
results = load_results()

# --- Connect once ---
con = duckdb.connect()

# --- Infer column names from both files and align ---
# Use LIMIT 0 to get schema without scanning
cur1 = con.execute(f"SELECT * FROM read_parquet('{DATA1.as_posix()}') LIMIT 0")
cols1 = [c[0] for c in cur1.description]  # DB-API description tuples

cur2 = con.execute(f"SELECT * FROM read_parquet('{DATA2.as_posix()}') LIMIT 0")
cols2 = [c[0] for c in cur2.description]

# Take intersection, preserving order from the first file
cols = [c for c in cols1 if c in cols2]

if not cols:
    raise ValueError("No overlapping columns between the two Parquet files.")

# --- Build NULL-safe equality across all overlapping columns ---
eq_conditions = " AND ".join(
    f"(t1.{quote_ident(c)} IS NOT DISTINCT FROM t2.{quote_ident(c)})" for c in cols
)

# --- Query: compare row i vs row i using row_number() ---
query = f"""
WITH
t1 AS (
  SELECT ROW_NUMBER() OVER () AS rn, * FROM read_parquet('{DATA1.as_posix()}')
),
t2 AS (
  SELECT ROW_NUMBER() OVER () AS rn, * FROM read_parquet('{DATA2.as_posix()}')
),
joined AS (
  SELECT {eq_conditions} AS row_match
  FROM t1
  JOIN t2 USING (rn)
)
SELECT AVG(CASE WHEN row_match THEN 1 ELSE 0 END)::DOUBLE AS match_rate
FROM joined;
"""

# --- Run & time ---
t0 = time.time()
match_rate = con.execute(query).fetchone()[0]
elapsed = time.time() - t0

print(f"Row-level match rate: {match_rate:.10f}")

# --- Update & persist results ---
results["DuckDB"] = {
    "match_rate": round(match_rate, 10),
    "time_taken_sec": round(elapsed, 10),
}

save_results(results)
print(results)
