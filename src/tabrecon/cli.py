"""Command line interface:  python -m tabrecon <command>"""
from __future__ import annotations

import argparse

from tabrecon import experiments as ex
from tabrecon.paths import dataset_dir


def _size(text: str) -> int:
    """Accept 7000000, 7_000_000, 7M, 1.5M, 500k."""
    t = text.replace("_", "").lower()
    mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(t[-1], 1)
    return int(float(t[:-1] if t[-1] in "kmb" else t) * mult)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tabrecon", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, repeats):
        sp.add_argument("--rows", type=_size, default=ex.DEFAULT_ROWS)
        sp.add_argument("--repeats", type=int, default=repeats)
        sp.add_argument("--mem-cap-gb", type=float, default=None,
                        help="kill a run whose process tree exceeds this (default: 75%% of RAM)")

    g = sub.add_parser("generate", help="create a dataset with ground truth")
    g.add_argument("--rows", type=_size, default=ex.DEFAULT_ROWS)

    common(sub.add_parser("benchmark", help="all methods on one dataset"), 7)
    sc = sub.add_parser("scaling", help="all methods across dataset sizes")
    sc.add_argument("--sizes", type=_size, nargs="+", default=[1_000_000, 7_000_000, 14_000_000, 28_000_000, 56_000_000])
    sc.add_argument("--repeats", type=int, default=3)
    sc.add_argument("--mem-cap-gb", type=float, default=None)
    common(sub.add_parser("batch-sweep", help="PyArrow batch size sweep"), 5)
    common(sub.add_parser("worker-sweep", help="parallel row-group worker sweep"), 5)
    common(sub.add_parser("dirty-sweep", help="fingerprint vs number of differing row groups"), 5)
    sub.add_parser("compile-cost", help="Numba first-call cost")
    sub.add_parser("report", help="build the report from results/raw")
    common(sub.add_parser("all", help="every study, then the report"), 7)

    a = p.parse_args(argv)
    if a.cmd == "generate":
        ex.ensure_dataset(a.rows)
        print("dataset:", dataset_dir(a.rows))
    elif a.cmd == "benchmark":
        ex.benchmark(a.rows, a.repeats, a.mem_cap_gb)
    elif a.cmd == "scaling":
        ex.scaling(a.sizes, a.repeats, a.mem_cap_gb)
    elif a.cmd == "batch-sweep":
        ex.batch_sweep(a.rows, a.repeats, a.mem_cap_gb)
    elif a.cmd == "worker-sweep":
        ex.worker_sweep(a.rows, a.repeats, a.mem_cap_gb)
    elif a.cmd == "dirty-sweep":
        ex.dirty_sweep(a.rows, a.repeats, a.mem_cap_gb)
    elif a.cmd == "compile-cost":
        ex.compile_cost()
    elif a.cmd == "report":
        from tabrecon import report
        report.build()
    elif a.cmd == "all":
        # batch sweep first so the "tuned" method uses its result
        ex.batch_sweep(a.rows, 5, a.mem_cap_gb)
        ex.compile_cost()
        ex.benchmark(a.rows, a.repeats, a.mem_cap_gb)
        ex.worker_sweep(a.rows, 5, a.mem_cap_gb)
        ex.dirty_sweep(a.rows, 5, a.mem_cap_gb)
        ex.scaling([1_000_000, 7_000_000, 14_000_000, 28_000_000, 56_000_000], 3, a.mem_cap_gb)
        from tabrecon import report
        report.build()


if __name__ == "__main__":
    main()
