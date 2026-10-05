import polars as pl


def run(path_a: str, path_b: str) -> float:
    l1 = pl.scan_parquet(path_a)
    l2 = pl.scan_parquet(path_b)

    cols1 = l1.collect_schema().names()
    cols2 = l2.collect_schema().names()
    cols = [c for c in cols1 if c in cols2]  # overlapping columns, in file-A order
    if not cols:
        raise ValueError("No overlapping columns between the two Parquet files.")

    # One 64-bit hash per row, then join the two hash streams on row position.
    h1 = l1.select(pl.struct(cols).hash(seed=0).alias("h")).with_row_index("rn")
    h2 = l2.select(pl.struct(cols).hash(seed=0).alias("h")).with_row_index("rn")

    return (
        h1.join(h2, on="rn", how="inner", suffix="_right")
        .with_columns((pl.col("h") == pl.col("h_right")).alias("row_match"))
        .select(pl.col("row_match").mean())
        .collect(engine="streaming")
        .item()
    )
