import polars as pl


def run(path_a: str, path_b: str) -> float:
    df1 = pl.read_parquet(path_a)
    df2 = pl.read_parquet(path_b)
    # (df1 == df2) is a boolean frame; count the rows that are all True by
    # materialising them as Python tuples.
    return (df1 == df2).rows().count((True,) * df1.width) / df1.height
