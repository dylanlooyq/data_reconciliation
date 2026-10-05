"""Run one method in an isolated subprocess and sample its resource use.

Measured per run
    wall_s        wall-clock time of ``run()`` only (perf_counter, inside the worker)
    cpu_s         CPU-seconds consumed by the run: the worker's own CPU time plus the
                  CPU time of any child processes it spawned (sampled; see below)
    peak_mem_mb   peak of the summed private memory (USS) of the whole process tree

Memory and child CPU are sampled every ``SAMPLE_INTERVAL`` seconds, so peaks that
last less than that can be missed and a child's final few milliseconds of CPU are
not counted. A memory guard kills the run if it threatens to exhaust the machine.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

from tabrecon.paths import SRC

SAMPLE_INTERVAL = 0.02
MIN_FREE_BYTES = 400 * 2**20  # abort a run if the whole machine drops below this


def _mem(p: psutil.Process) -> int:
    try:
        return p.memory_full_info().uss
    except (psutil.AccessDenied, AttributeError):
        return p.memory_info().rss


def run_once(
    key: str,
    path_a: Path,
    path_b: Path,
    params: dict | None = None,
    mem_cap_bytes: int | None = None,
    timeout_s: float = 900,
) -> dict:
    """Run a method once. Always returns a dict; ``status`` says how it ended."""
    import os

    env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "tabrecon.harness.worker", key, str(path_a), str(path_b), json.dumps(params or {})]

    peak_mem = 0
    cpu_seen: dict[int, float] = {}  # pid -> latest CPU seconds, for child processes
    status = "ok"

    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, env=env, text=True)
        root = psutil.Process(proc.pid)
        start = time.perf_counter()

        while proc.poll() is None:
            try:
                tree = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            total = 0
            for p in tree:
                try:
                    total += _mem(p)
                    if p.pid != root.pid:
                        t = p.cpu_times()
                        cpu_seen[p.pid] = t.user + t.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            peak_mem = max(peak_mem, total)

            if mem_cap_bytes and total > mem_cap_bytes:
                status = "memory_guard"
            elif psutil.virtual_memory().available < MIN_FREE_BYTES:
                status = "memory_guard"
            elif time.perf_counter() - start > timeout_s:
                status = "timeout"
            if status != "ok":
                for p in reversed(tree):
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                break
            time.sleep(SAMPLE_INTERVAL)

        proc.wait()
        out = proc.stdout.read()
        err.seek(0)
        stderr_tail = err.read().decode(errors="replace").strip().splitlines()[-3:]

    row = {"method": key, "params": params or {}, "status": status,
           "peak_mem_mb": round(peak_mem / 2**20, 1)}
    result_line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
    if status == "ok" and result_line:
        r = json.loads(result_line[len("RESULT "):])
        row.update(
            match_rate=r["match_rate"],
            wall_s=r["wall_s"],
            cpu_s=r["cpu_self_s"] + sum(cpu_seen.values()),
            n_child_processes=len(cpu_seen),
        )
    elif status == "ok":
        row["status"] = "error"
        row["error"] = " | ".join(stderr_tail)[-400:]
    return row
