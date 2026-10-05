from pathlib import Path

import duckdb


def _quote_ident(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def run(path_a: str, path_b: str) -> float:
    a = Path(path_a).as_posix()
    b = Path(path_b).as_posix()
    con = duckdb.connect()

    # LIMIT 0 reads the schema without scanning.
    cols1 = [c[0] for c in con.execute(f"SELECT * FROM read_parquet('{a}') LIMIT 0").description]
    cols2 = [c[0] for c in con.execute(f"SELECT * FROM read_parquet('{b}') LIMIT 0").description]
    cols = [c for c in cols1 if c in cols2]
    if not cols:
        raise ValueError("No overlapping columns between the two Parquet files.")

    # NULL-safe equality across every shared column.
    eq = " AND ".join(
        f"(t1.{_quote_ident(c)} IS NOT DISTINCT FROM t2.{_quote_ident(c)})" for c in cols
    )
    query = f"""
    WITH
    t1 AS (SELECT ROW_NUMBER() OVER () AS rn, * FROM read_parquet('{a}')),
    t2 AS (SELECT ROW_NUMBER() OVER () AS rn, * FROM read_parquet('{b}')),
    joined AS (SELECT {eq} AS row_match FROM t1 JOIN t2 USING (rn))
    SELECT AVG(CASE WHEN row_match THEN 1 ELSE 0 END)::DOUBLE FROM joined
    """
    try:
        return con.execute(query).fetchone()[0]
    finally:
        con.close()
