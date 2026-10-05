import pandas as pd


def run(path_a: str, path_b: str) -> float:
    df1 = pd.read_parquet(path_a)
    df2 = pd.read_parquet(path_b)
    return float((df1 == df2).all(axis=1).mean())
