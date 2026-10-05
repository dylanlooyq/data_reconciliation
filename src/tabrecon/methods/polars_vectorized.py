import polars as pl


def run(path_a: str, path_b: str) -> float:
    df1 = pl.read_parquet(path_a)
    df2 = pl.read_parquet(path_b)
    # Assumes identical schemas and column order.
    return (
        (df1 == df2)
        .select(pl.all_horizontal(pl.all()).alias("row_match"))
        .select(pl.col("row_match").mean())
        .item()
    )
