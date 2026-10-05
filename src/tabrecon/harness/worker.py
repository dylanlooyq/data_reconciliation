"""Subprocess entry point: run one method once and report what it measured.

    python -m tabrecon.harness.worker <method_key> <path_a> <path_b> <params_json>

Library imports and ``prepare()`` happen before the clock starts, so the timed
region is only ``run()``: reading the files and computing the answer.
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import sys
import time


def main() -> None:
    key, path_a, path_b, params_json = sys.argv[1:5]

    from tabrecon.methods.registry import get

    spec = get(key)
    params = {**spec.params, **json.loads(params_json)}
    mod = importlib.import_module(spec.module)
    prepare = getattr(mod, "prepare", None)
    if prepare:
        prepare()
    gc.collect()

    cpu0 = time.process_time()  # this process, all threads
    t0 = time.perf_counter()
    match_rate = mod.run(path_a, path_b, **params)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0

    print("RESULT " + json.dumps({
        "match_rate": float(match_rate),
        "wall_s": wall,
        "cpu_self_s": cpu,
        "pid": os.getpid(),
    }), flush=True)


if __name__ == "__main__":
    main()
