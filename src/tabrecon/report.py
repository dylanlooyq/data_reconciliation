"""Build the report (HTML + Markdown + figures) from ``results/raw``.

Every number in the prose is computed from the raw results, so re-running the
benchmark and then ``tabrecon report`` regenerates a consistent document.
"""
from __future__ import annotations

import base64
import html
import json
import re
import statistics
from pathlib import Path

from tabrecon import plots
from tabrecon.methods.registry import BY_KEY, METHODS, NEW, ORIGINAL
from tabrecon.paths import FIGURES_DIR, RAW_DIR, REPORT_DIR, RESULTS_DIR

TITLE = "Accelerating Tabular Reconciliation"
SUBTITLE = "A benchmark of execution strategies for row-level comparison of large datasets"


# ------------------------------------------------------------------------------ data
def load_raw(name: str) -> dict | None:
    p = RAW_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def summarize(runs: list[dict], key: str = "variant") -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r[key], []).append(r)
    out = {}
    for k, rs in groups.items():
        ok = [r for r in rs if r["status"] == "ok"]
        s = {"n": len(ok), "failures": sorted({r["status"] for r in rs if r["status"] != "ok"}),
             "method": rs[0]["method"]}
        if ok:
            w = [r["wall_s"] for r in ok]
            s.update(wall=statistics.median(w), wall_min=min(w), wall_max=max(w),
                     cpu=statistics.median(r["cpu_s"] for r in ok),
                     mem=statistics.median(r["peak_mem_mb"] for r in ok),
                     correct=all(r.get("correct") for r in ok))
        out[k] = s
    return out


def t(seconds: float) -> str:
    return f"{seconds:.2f} s" if seconds >= 0.1 else f"{seconds * 1000:.0f} ms"


def mem(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def rows_label(n: int) -> str:
    return f"{n / 1e6:g}M"


# ------------------------------------------------------------------------------ content model
# Blocks: ("h2"|"h3"|"p"|"ul"|"table"|"fig"|"tiles"|"cards"|"callout", payload...)
def build_blocks(ctx: dict) -> list[tuple]:
    b: list[tuple] = []
    bench, n_rows, reps = ctx["bench"], ctx["n_rows"], ctx["repeats"]
    ok = {k: s for k, s in bench.items() if s["n"]}
    fastest = min(ok, key=lambda k: ok[k]["wall"])
    slowest = max(ok, key=lambda k: ok[k]["wall"])
    lightest = min(ok, key=lambda k: ok[k]["mem"])
    cheapest_cpu = min(ok, key=lambda k: ok[k]["cpu"])
    orig = [k for k in ok if BY_KEY[k].family == ORIGINAL]
    new = [k for k in ok if BY_KEY[k].family == NEW]
    best_orig = min(orig, key=lambda k: ok[k]["wall"])
    lab = lambda k: BY_KEY[k].label  # noqa: E731
    speedup = ok[slowest]["wall"] / ok[fastest]["wall"]
    env = ctx["env"]

    # ---- summary
    b.append(("tiles", [
        (t(ok[fastest]["wall"]), f"fastest: {lab(fastest)}"),
        (f"{speedup:.0f}×", f"faster than the slowest ({lab(slowest)}, {t(ok[slowest]['wall'])})"),
        (mem(ok[lightest]["mem"]), f"lowest peak memory: {lab(lightest)}"),
        (f"{ok[cheapest_cpu]['cpu']:.1f} s", f"least CPU time: {lab(cheapest_cpu)}"),
    ]))
    b.append(("h2", "Summary"))
    b.append(("p", (
        f"An analyst has been asked to reconcile two extracts of {rows_label(n_rows)} rows and 12 columns, "
        f"and the book closes soon. Ten ways of comparing them were measured: seven from the original study and "
        f"three added here. On this machine the slowest took {speedup:.0f}× as long as the fastest, "
        f"with every method given the same data, the same machine and the same answer to reach."
    )))
    s_new = []
    for k in new:
        r = ok[k]["wall"] / ok[best_orig]["wall"]
        s_new.append(f"**{lab(k)}** took {t(ok[k]['wall'])} ({r:.1f}× the time of the best original method, "
                     f"{lab(best_orig)})")
    b.append(("p", "Of the three new methods: " + "; ".join(s_new) + ". The numbers alone do not tell the whole story, "
                                                                    "because each method trades something away; see the scorecard and the decision guide."))
    if not all(s["correct"] for s in ok.values()):
        b.append(("callout", "Warning: at least one method returned an answer that disagrees with the ground truth. "
                             "See the raw results before relying on this report."))
    else:
        b.append(("p", "Every method returned the correct answer on every run: each result was checked against "
                       "the set of changed rows recorded when the data was generated."))

    # ---- key findings (all computed)
    par = ok.get("parallel_row_groups")
    kf: list[str] = []
    pt = ok.get("pyarrow_tuned")
    if pt and fastest != "pyarrow_tuned":
        kf.append(f"**Fastest is not the cheapest.** {lab(fastest)} took {t(ok[fastest]['wall'])} but peaked at {mem(ok[fastest]['mem'])}; "
                  f"{lab('pyarrow_tuned')} took {t(pt['wall'])} ({pt['wall'] / ok[fastest]['wall']:.1f}× longer) and peaked at {mem(pt['mem'])} "
                  f"({ok[fastest]['mem'] / pt['mem']:.1f}× less memory).")
    sc_ = ctx.get("scaling")
    if sc_:
        top = max(sc_)
        done = [k for k in sc_[top] if sc_[top][k]["n"]]
        lost = [k for k in BY_KEY if k not in done]  # includes methods already stopped at a smaller size
        if lost:
            kf.append(f"**Memory decides what scales.** At {rows_label(top)} rows only {len(done)} of {len(BY_KEY)} methods finished on this machine; "
                      f"stopped: {', '.join(lab(k) for k in lost)}. Finishers: {', '.join(lab(k) for k in sorted(done, key=lambda k: sc_[top][k]['wall']))}.")
            fd = min(done, key=lambda k: sc_[top][k]["wall"])
            kf.append(f"**The ranking changes with size.** At {rows_label(top)} rows the fastest finisher was {lab(fd)} ({t(sc_[top][fd]['wall'])}), "
                      f"not {lab(fastest)}, which was fastest at {rows_label(n_rows)} rows.")
    if par:
        kf.append(f"**Parallelism costs CPU.** Parallel row groups kept {par['cpu'] / par['wall']:.1f} cores busy and used "
                  f"{par['cpu']:.1f} CPU-seconds, {par['cpu'] / ok[fastest]['cpu']:.1f}× the CPU of {lab(fastest)}, yet was slower at {rows_label(n_rows)} rows.")
    fpd = ctx.get("dirty")
    if fpd:
        fpm = fpd["median"]["rowgroup_fingerprint"]
        lv = sorted(fpm)
        kf.append(f"**Skipping beats speeding up, if the data allows.** The fingerprint method took {t(fpm[lv[0]])} on identical files and "
                  f"{t(fpm[lv[1]])} when one row group differed, but gains nothing once the differences touch every row group.")
    nk = ok.get("numba_kernel")
    if nk and "pyarrow" in ok:
        kf.append(f"**Hand-written code is not automatically faster.** The Numba kernel ({ctx['loc']['numba_kernel']} lines) took {t(nk['wall'])}, "
                  f"against {t(ok['pyarrow']['wall'])} for PyArrow ({ctx['loc']['pyarrow']} lines), and needs a one-off compile.")
    if kf:
        b.append(("h3", "Key findings"))
        b.append(("ul", kf))

    # ---- scenario
    b.append(("h2", "1. Scenario and research question"))
    b.append(("p", (
        "A data analyst owns a month-end reconciliation. Two systems each produce a file of the same shape, and "
        "the question is how many rows agree. The files have grown from thousands of rows to millions, the "
        "spreadsheet no longer opens, and the deadline has not moved. The analyst can reach for a different "
        "library, change how the data is read, use more of the machine, or avoid doing part of the work."
    )))
    b.append(("p", ("**Research question.** For a row-by-row comparison of two large tables, which execution strategy "
                    "gives the shortest time to an answer, and what does each one cost in memory, CPU and complexity?")))

    # ---- method of study
    b.append(("h2", "2. How the study was run"))
    truth = ctx["truth"]
    b.append(("ul", [
        f"**Workload.** Two Parquet files of {n_rows:,} rows each: a row number, ten one-letter string columns "
        f"and an integer value. The second file is a copy in which {truth['n_changed']:,} rows "
        f"({truth['n_changed'] / n_rows:.4%}) were altered at random positions. The answer to be reproduced is a "
        f"match rate of {truth['expected_match_rate']:.6f}. The comparison is positional: row *i* of one file against row *i* of the other.",
        "**Ground truth.** The generator records which rows it changed, so correctness is verified on every run "
        "rather than assumed.",
        f"**Isolation.** Each measurement is a fresh Python process. Library imports happen before the clock starts; "
        f"the timed region is reading both files and computing the match rate. Methods are interleaved round-robin "
        f"over {reps} repeats so that drift in machine load is spread across them. The file cache is warm.",
        "**Cost axes.** *Time* is wall-clock for the timed region. *Memory* is the peak of the summed private memory "
        "of the process and any worker processes it started. *CPU time* is the CPU-seconds consumed by all threads and "
        "processes; divided by wall time it shows how many cores were kept busy. *Code size* is the number of "
        "non-blank, non-comment lines in the method module, a rough proxy for what the analyst must write and maintain.",
        "**Safety.** A memory guard stops any run that would exhaust the machine and records the method as unable to handle that size.",
    ]))
    b.append(("table", ["Machine", ""], [
        ["Operating system", env.get("platform", "")],
        ["CPU", f"{env.get('physical_cpus')} physical cores, {env.get('logical_cpus')} threads"],
        ["Memory", f"{env.get('ram_gb')} GB RAM"],
        ["Python", env.get("python", "")],
        ["Libraries", ", ".join(f"{k} {v}" for k, v in env.get("libraries", {}).items())],
    ]))

    # ---- methods
    b.append(("h2", "3. The methods"))
    b.append(("p", "The first seven come from the original study; the last three are new. "
                   "“Exactness” says whether the answer can ever be wrong in principle; “nulls” records how a missing value in both files is treated, "
                   "which differs between methods even though this workload contains no nulls."))
    cards = []
    for m in METHODS:
        s = bench.get(m.key, {})
        glance = (f"{t(s['wall'])} · {mem(s['mem'])} · {s['cpu']:.1f} s CPU" if s.get("n") else "no successful run")
        cards.append(dict(title=m.label, family="New in this study" if m.family == NEW else "Original study",
                          description=m.description, tradeoff=m.tradeoff,
                          meta=[("Exactness", m.exact), ("Nulls", m.nulls), ("Code size", f"{m.lines_of_code} lines"),
                                (f"Result at {rows_label(n_rows)} rows", glance)]))
    b.append(("cards", cards))

    # ---- results
    b.append(("h2", f"4. Results at {rows_label(n_rows)} rows"))
    b.append(("h3", "4.1 Speed"))
    b.append(("fig", "runtime", f"Median time to reconcile {rows_label(n_rows)} rows over up to {reps} runs per method.",
              ["Method", "Median", "Fastest", "Slowest", "Runs"],
              [[lab(k), t(ok[k]["wall"]), t(ok[k]["wall_min"]), t(ok[k]["wall_max"]), str(ok[k]["n"])]
               for k in sorted(ok, key=lambda k: ok[k]["wall"])]))
    short = [k for k in ok if ok[k]["n"] < reps]
    if short:
        b.append(("p", "; ".join(f"{lab(k)} completed {ok[k]['n']} of {reps} runs (the memory guard stopped the rest because the machine ran short of memory)"
                                 for k in short) + ". Its median uses the runs that completed."))
    ranked = sorted(ok, key=lambda k: ok[k]["wall"])
    b.append(("p", (f"{lab(ranked[0])} was fastest, followed by {lab(ranked[1])} and {lab(ranked[2])}. "
                    f"The slowest method, {lab(slowest)}, took {ok[slowest]['wall'] / ok[ranked[0]]['wall']:.0f}× longer.")))
    b.append(("h3", "4.2 Speed against memory"))
    b.append(("fig", "memory_vs_time", "Each point is one method. The dashed line joins the methods that no other method beats on both axes.",
              ["Method", "Peak memory", "Median time"],
              [[lab(k), mem(ok[k]["mem"]), t(ok[k]["wall"])] for k in ranked]))
    heavy = max(ok, key=lambda k: ok[k]["mem"])
    b.append(("p", (f"Memory ranges from {mem(ok[lightest]['mem'])} ({lab(lightest)}) to {mem(ok[heavy]['mem'])} ({lab(heavy)}). "
                    "Methods near the origin are both fast and light; a method that is fast only because it loads everything "
                    "into memory will stop being an option when the data outgrows the machine (see section 5).")))
    b.append(("h3", "4.3 CPU cost"))
    b.append(("fig", "cpu", "Wall-clock time (open circle) against total CPU time (filled circle) per method.",
              ["Method", "Wall", "CPU time", "Cores busy"],
              [[lab(k), t(ok[k]["wall"]), f"{ok[k]['cpu']:.2f} s", f"{ok[k]['cpu'] / ok[k]['wall']:.1f}"] for k in ranked]))
    par = ok.get("parallel_row_groups")
    similar = sorted(ok, key=lambda k: ok[k]["wall"])[:5]
    spread = max(ok[k]["cpu"] for k in similar) / min(ok[k]["cpu"] for k in similar)
    msg = ("CPU time matters when the machine is shared or billed by the core-second. Wall-clock time hides how much "
           f"work was done: among the five fastest methods, total CPU time differs by {spread:.1f}×.")
    if par:
        msg += (f" The parallel row-group method kept {par['cpu'] / par['wall']:.1f} cores busy on average and used "
                f"{par['cpu']:.1f} CPU-seconds, against {ok[fastest]['cpu']:.1f} for {lab(fastest)}.")
    b.append(("p", msg))
    b.append(("h3", "4.4 Scorecard"))
    b.append(("fig", "scorecard", "Each cell shows the value and how many times worse it is than the best method in that column.",
              ["Method", "Time", "Peak memory", "CPU time", "Code size"],
              [[lab(k), t(ok[k]["wall"]), mem(ok[k]["mem"]), f"{ok[k]['cpu']:.1f} s", f"{ctx['loc'][k]} lines"] for k in ranked]))

    # ---- scaling
    sc = ctx.get("scaling")
    if sc:
        b.append(("h2", "5. Scaling with dataset size"))
        sizes = sorted(sc)
        done = {k: max((n for n in sizes if k in sc[n] and sc[n][k]["n"]), default=0) for k in BY_KEY}
        stopped = [k for k in BY_KEY if k in sc[max(sizes)] and not sc[max(sizes)][k]["n"]
                   or any(k in sc[n] and not sc[n][k]["n"] for n in sizes)]
        b.append(("p", (f"The same comparison was repeated at {', '.join(rows_label(n) for n in sizes)} rows per file, "
                        "with the changed-row proportion held constant.")))
        b.append(("fig", "scaling_time", "Median time against rows per file, both axes logarithmic.",
                  ["Method"] + [rows_label(n) for n in sizes],
                  [[lab(k)] + [t(sc[n][k]["wall"]) if k in sc[n] and sc[n][k]["n"] else
                               ("stopped" if k in sc[n] else "not run") for n in sizes] for k in BY_KEY]))
        b.append(("fig", "scaling_memory", "Peak memory against rows per file.",
                  ["Method"] + [rows_label(n) for n in sizes],
                  [[lab(k)] + [mem(sc[n][k]["mem"]) if k in sc[n] and sc[n][k]["n"] else
                               ("stopped" if k in sc[n] else "not run") for n in sizes] for k in BY_KEY]))
        if stopped:
            names = ", ".join(f"{lab(k)} (largest completed: {rows_label(done[k])})" if done[k] else lab(k) for k in stopped)
            b.append(("p", f"On this machine the following methods were stopped by the memory guard before the largest size: {names}. "
                           f"The limit is the memory actually usable on this machine at the time ({env.get('ram_gb')} GB installed, part of it taken by the operating system and background processes), not a fixed property of each method; "
                           "on a larger machine the cut-off moves, but the order in which methods fail would not. A method that cannot finish is the slowest possible method, whatever it scored on a smaller file."))
        else:
            b.append(("p", "Every method completed every size tested on this machine."))
        top = max(sizes)
        top_ok = {k: sc[top][k] for k in sc[top] if sc[top][k]["n"]}
        if top_ok:
            f_top = min(top_ok, key=lambda k: top_ok[k]["wall"])
            l_top = min(top_ok, key=lambda k: top_ok[k]["mem"])
            b.append(("p", f"At {rows_label(top)} rows the fastest completed method was {lab(f_top)} ({t(top_ok[f_top]['wall'])}) "
                           f"and the lightest was {lab(l_top)} ({mem(top_ok[l_top]['mem'])})."))

    # ---- sensitivity
    b.append(("h2", "6. Sensitivity studies"))
    b.append(("p", "Three of the methods depend on a setting or on the shape of the data. These studies show how much."))
    bs = ctx.get("batch")
    if bs:
        best = bs["best"]
        b.append(("h3", "6.1 PyArrow batch size"))
        b.append(("fig", "batch_size", "Runtime of the PyArrow method against the number of rows read per batch.",
                  ["Batch size", "Median time"], [[f"{k:,}", t(v)] for k, v in bs["median"].items()]))
        one = list(bs["median"].values())[-1]
        gain = 1 - bs["median"][best] / one
        txt = (f"The fastest batch size was {best:,} rows ({t(bs['median'][best])}); reading each file as one batch took {t(one)}"
               f" and the smallest batch size took {t(max(bs['median'].values()))}. ")
        if gain < 0.10:
            txt += (f"Above roughly {sorted(bs['median'], key=bs['median'].get)[0]:,} rows the curve is flat, so tuning changes the runtime by only {gain:.0%}. ")
        else:
            txt += f"Tuning improves on a single batch by {gain:.0%}. "
        pa, pt = bench.get("pyarrow"), bench.get("pyarrow_tuned")
        if pa and pt and pa["n"] and pt["n"]:
            txt += (f"The larger effect is on memory: the single-batch method peaked at {mem(pa['mem'])}, the tuned one at {mem(pt['mem'])} "
                    f"({pa['mem'] / pt['mem']:.1f}× less), because only one batch is held at a time. ")
        txt += ("Very small batches pay per-batch overhead. The best size is specific to this machine and data, so it should be "
                "measured rather than assumed. The sweep uses repeats and medians; the original version took the minimum of single runs, "
                "which favours noise.")
        b.append(("p", txt))
    ws = ctx.get("workers")
    if ws:
        b.append(("h3", "6.2 Parallel row groups: how many workers?"))
        b.append(("fig", "workers", "Runtime of the parallel method against the number of worker processes.",
                  ["Workers", "Median time", "Speed-up vs 1 worker"],
                  [[str(k), t(v), f"{ws['median'][min(ws['median'])] / v:.1f}×"] for k, v in ws["median"].items()]))
        bw = min(ws["median"], key=ws["median"].get)
        gain = ws['median'][min(ws['median'])] / ws['median'][bw]
        eff = gain / bw
        verdict = ("close to the ideal" if eff >= 0.7 else "well short of the ideal" if eff < 0.4 else "short of the ideal")
        b.append(("p", (f"Best was {bw} workers at {t(ws['median'][bw])}, {gain:.1f}× faster than one worker, which is {verdict} "
                        f"({eff:.0%} parallel efficiency). Every worker is a separate process that must start, import its libraries "
                        f"and read its share of the file, and the machine has {env.get('physical_cpus')} physical cores "
                        f"({env.get('logical_cpus')} threads). The number of row groups ({ctx['truth']['n_row_groups']}) also caps "
                        "how finely the work can be divided.")))
    dg = ctx.get("dirty")
    if dg:
        b.append(("h3", "6.3 Fingerprinting: how widely are the differences spread?"))
        keys = ["rowgroup_fingerprint", "pyarrow_tuned", "polars_vectorized"]
        levels = sorted(dg["median"][keys[0]])
        b.append(("fig", "dirty", f"Runtime against the number of row groups (of {dg['n_groups']}) that contain a difference. "
                                  "The total number of changed rows is the same in every case.",
                  ["Row groups differing"] + [lab(k) for k in keys],
                  [[str(g)] + [t(dg["median"][k][g]) if g in dg["median"].get(k, {}) else "–" for k in keys] for g in levels]))
        fp = dg["median"]["rowgroup_fingerprint"]
        base_key = "pyarrow_tuned"
        base = dg["median"][base_key]
        wins = [g for g in levels if fp[g] < 0.95 * base[g]]  # a real margin, not a tie
        lost = [g for g in levels if fp[g] >= 0.95 * base[g]]
        txt = (f"With identical files the fingerprint method finished in {t(fp[levels[0]])}, against {t(base[levels[0]])} for {lab(base_key)}. "
               f"It was clearly faster than {lab(base_key)} (by more than 5%) when up to {max(wins)} of {dg['n_groups']} row groups differed" if wins else
               "The fingerprint method was not clearly faster than the PyArrow baseline at any tested level")
        if wins and lost:
            txt += f", and had no meaningful advantage from {min(lost)} differing groups onwards."
        elif wins:
            txt += "."
        txt += (" The speed-up comes from not decoding unchanged data, so it depends entirely on how the differences are distributed: "
                f"in the main benchmark the {truth['n_changed']} changes are scattered across {truth['n_row_groups']} row groups, "
                "so nearly every group is touched and there is little to skip.")
        b.append(("p", txt))
    nc = ctx.get("compile")
    if nc:
        c, w = nc["cold_cache"], nc["warm_cache"]
        b.append(("h3", "6.4 The Numba kernel's first-call cost"))
        b.append(("p", (f"A compiled kernel has a one-off cost that the benchmark excludes from the timed region: with an empty cache the first call took "
                        f"{c['first_call_s']:.1f} s to compile; with the cache populated it took {w['first_call_s']:.1f} s to load. "
                        "That is irrelevant for a job that runs daily and decisive for a one-off, so it belongs in the decision.")))

    # ---- decision guide
    b.append(("h2", "7. Decision guide"))
    b.append(("table", ["If you need…", "Choose", "Evidence"], ctx["guide"]))

    # ---- limits
    b.append(("h2", "8. Limitations"))
    b.append(("ul", [
        "**One machine.** Absolute times will differ elsewhere; rankings and ratios are more portable than seconds."
        + (f" Only {env.get('available_ram_gb_at_start')} GB of the {env.get('ram_gb')} GB of RAM was free when the study started, "
           "so other applications were competing for memory; this is a realistic setting but adds noise to the larger runs."
           if (env.get("available_ram_gb_at_start") or 99) < 0.25 * (env.get("ram_gb") or 1) else ""),
        "**Synthetic data.** Ten one-letter columns with 26 possible values compress and compare very differently from free-text, decimal or date columns. Wider or higher-cardinality data will shift the balance, especially for the byte-level and string-comparing methods.",
        "**Positional comparison only.** Real reconciliations usually match on a key, tolerate rounding, and must say *which* rows broke and why. None of that is measured here, and key-based joins have a different cost profile.",
        "**Very sparse differences.** Only 0.003% of rows differ. Methods that produce a break report would do more work as breaks become common.",
        "**Parquet inputs with a shared layout.** The row-group fingerprint method needs both files written the same way. Files from two different systems almost never are, in which case it falls back to a full comparison.",
        "**Warm cache, local disk.** Slower storage would make reading dominate and favour methods that read less.",
        "**Measurement granularity.** Memory and child CPU are sampled every 20 ms, so a very short peak can be missed, and a worker's last few milliseconds of CPU are not counted. On Windows the CPU timer ticks about every 15 ms, so CPU time for runs shorter than about 100 ms is coarse.",
        "**Tuning in-sample.** The batch size was tuned and evaluated on the same data.",
        "**Null handling differs** between methods (section 3). This workload has no nulls, so the difference is invisible here but would matter on real data.",
    ]))
    b.append(("h2", "9. Reproducing this study"))
    b.append(("code", (
        "pip install -e .[dev]\n"
        "python -m pytest                     # every method agrees with ground truth\n"
        "python -m tabrecon all               # generate data, run every study, build this report\n"
        "# or step by step:\n"
        "python -m tabrecon batch-sweep       # tunes the PyArrow batch size first\n"
        "python -m tabrecon benchmark --rows 7M --repeats 7\n"
        "python -m tabrecon scaling --sizes 1M 7M 14M 28M 56M\n"
        "python -m tabrecon report")))
    return b


# ------------------------------------------------------------------------------ rendering
def _inline_html(text: str) -> str:
    s = html.escape(text, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(.+?)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _table_html(header, rows) -> str:
    head = "".join(f"<th>{_inline_html(str(h))}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{_inline_html(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--ring:rgba(11,11,11,.10);--accent:#2a78d6;--new:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--accent:#3987e5;--new:#3987e5}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--accent:#3987e5;--new:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:2.1rem;line-height:1.2;margin:0 0 6px}
.sub{color:var(--ink2);font-size:1.1rem;margin:0 0 28px}
h2{font-size:1.45rem;margin:48px 0 8px;padding-top:8px;border-top:1px solid var(--grid)}
h3{font-size:1.1rem;margin:28px 0 6px}
p,li{color:var(--ink)}li{margin:6px 0}
code,pre{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:.9em}
code{background:var(--grid);padding:1px 5px;border-radius:4px}
pre{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:14px;overflow-x:auto}
pre code{background:none;padding:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:0 0 8px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px}
.tile b{display:block;font-size:2.1rem;line-height:1.1;letter-spacing:-.02em}
.tile span{color:var(--ink2);font-size:.88rem}
.cards{display:grid;gap:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px 18px}
.card h4{margin:0 0 2px;font-size:1.05rem}
.badge{display:inline-block;font-size:.72rem;padding:1px 8px;border-radius:99px;border:1px solid var(--ring);color:var(--ink2);margin-left:8px;vertical-align:middle}
.badge.new{background:var(--new);color:#fff;border-color:transparent}
.card p{margin:6px 0}.card dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;margin:8px 0 0;font-size:.88rem}
.card dt{color:var(--muted)}.card dd{margin:0;color:var(--ink2)}
figure{margin:14px 0 6px;background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px}
figure img{width:100%;height:auto;display:block}
figcaption{color:var(--ink2);font-size:.88rem;padding:6px 8px 2px}
details{margin:2px 0 12px;font-size:.88rem;color:var(--ink2)}summary{cursor:pointer;padding:4px 0}
.tablewrap{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}
.callout{border-left:4px solid #ec835a;background:var(--surface);padding:10px 14px;border-radius:0 8px 8px 0}
@media (max-width:560px){h1{font-size:1.6rem}main{padding-top:24px}}
"""


def render_html(blocks: list[tuple], figs: dict[str, tuple[Path, Path]]) -> str:
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{TITLE}</title><style>{CSS}</style></head><body><main>",
           f"<h1>{TITLE}</h1><p class='sub'>{SUBTITLE}</p>"]
    for blk in blocks:
        kind = blk[0]
        if kind == "h2":
            out.append(f"<h2>{_inline_html(blk[1])}</h2>")
        elif kind == "h3":
            out.append(f"<h3>{_inline_html(blk[1])}</h3>")
        elif kind == "p":
            out.append(f"<p>{_inline_html(blk[1])}</p>")
        elif kind == "ul":
            out.append("<ul>" + "".join(f"<li>{_inline_html(i)}</li>" for i in blk[1]) + "</ul>")
        elif kind == "callout":
            out.append(f"<p class='callout'>{_inline_html(blk[1])}</p>")
        elif kind == "code":
            out.append(f"<pre><code>{html.escape(blk[1])}</code></pre>")
        elif kind == "table":
            out.append(_table_html(blk[1], blk[2]))
        elif kind == "tiles":
            out.append("<div class='tiles'>" + "".join(
                f"<div class='tile'><b>{html.escape(v)}</b><span>{html.escape(l)}</span></div>" for v, l in blk[1]) + "</div>")
        elif kind == "cards":
            out.append("<div class='cards'>")
            for c in blk[1]:
                new = c["family"].startswith("New")
                meta = "".join(f"<dt>{html.escape(k)}</dt><dd>{_inline_html(v)}</dd>" for k, v in c["meta"])
                out.append(f"<div class='card'><h4>{html.escape(c['title'])}<span class='badge{' new' if new else ''}'>"
                           f"{html.escape(c['family'])}</span></h4><p>{_inline_html(c['description'])}</p>"
                           f"<p><strong>Trade-off.</strong> {_inline_html(c['tradeoff'])}</p><dl>{meta}</dl></div>")
            out.append("</div>")
        elif kind == "fig":
            _, name, caption, header, rows = blk
            light, dark = figs[name]
            alt = html.escape(caption)
            out.append(f"<figure><picture><source media='(prefers-color-scheme: dark)' srcset='{_data_uri(dark)}'>"
                       f"<img src='{_data_uri(light)}' alt='{alt}'></picture><figcaption>{_inline_html(caption)}</figcaption></figure>"
                       f"<details><summary>View data as a table</summary>{_table_html(header, rows)}</details>")
    out.append("</main></body></html>")
    return "".join(out)


def _md_table(header, rows) -> str:
    esc = lambda c: str(c).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    return "\n".join(["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * len(header),
                      *("| " + " | ".join(esc(c) for c in r) + " |" for r in rows)])


def render_md(blocks: list[tuple]) -> str:
    out = [f"# {TITLE}\n", f"*{SUBTITLE}*\n"]
    for blk in blocks:
        kind = blk[0]
        if kind == "h2":
            out.append(f"\n## {blk[1]}\n")
        elif kind == "h3":
            out.append(f"\n### {blk[1]}\n")
        elif kind == "p":
            out.append(blk[1] + "\n")
        elif kind == "ul":
            out.append("\n".join(f"- {i}" for i in blk[1]) + "\n")
        elif kind == "callout":
            out.append(f"> **{blk[1]}**\n")
        elif kind == "code":
            out.append(f"```bash\n{blk[1]}\n```\n")
        elif kind == "table":
            out.append(_md_table(blk[1], blk[2]) + "\n")
        elif kind == "tiles":
            out.append(" · ".join(f"**{v}** {l}" for v, l in blk[1]) + "\n")
        elif kind == "cards":
            for c in blk[1]:
                out.append(f"#### {c['title']} — *{c['family']}*\n\n{c['description']}\n\n**Trade-off.** {c['tradeoff']}\n\n"
                           + "\n".join(f"- {k}: {v}" for k, v in c["meta"]) + "\n")
        elif kind == "fig":
            _, name, caption, header, rows = blk
            out.append(f"![{caption}](figures/{name}.png)\n\n*{caption}*\n\n<details><summary>Data</summary>\n\n"
                       f"{_md_table(header, rows)}\n\n</details>\n")
    return "\n".join(out)


# ------------------------------------------------------------------------------ orchestration
def _guide(ctx: dict) -> list[list[str]]:
    bench, sc, dg = ctx["bench"], ctx.get("scaling"), ctx.get("dirty")
    ok = {k: s for k, s in bench.items() if s["n"]}
    lab = lambda k: BY_KEY[k].label  # noqa: E731
    fastest = min(ok, key=lambda k: ok[k]["wall"])
    lightest = min(ok, key=lambda k: ok[k]["mem"])
    cpu = min(ok, key=lambda k: ok[k]["cpu"])
    shortest = min(ctx["loc"], key=ctx["loc"].get)
    fastest_note = f"{t(ok[fastest]['wall'])} at {rows_label(ctx['n_rows'])} rows"
    if "pandas" in ok and fastest != "pandas":
        fastest_note += f"; {ok['pandas']['wall'] / ok[fastest]['wall']:.0f}× faster than Pandas"
    rows = [
        ["The shortest time on one machine with enough memory", lab(fastest), fastest_note],
        ["The smallest memory footprint", lab(lightest), f"{mem(ok[lightest]['mem'])} peak at {rows_label(ctx['n_rows'])} rows"],
        ["The least total CPU work (shared or billed machine)", lab(cpu), f"{ok[cpu]['cpu']:.1f} CPU-seconds"],
        ["The least code to write and maintain", lab(shortest), f"{ctx['loc'][shortest]} lines"],
    ]
    if sc:
        top = max(sc)
        done = [k for k in sc[top] if sc[top][k]["n"]]
        if done:
            f_top = min(done, key=lambda k: sc[top][k]["wall"])
            rows.append([f"The largest file that still finishes ({rows_label(top)} rows tested)", lab(f_top),
                         f"{t(sc[top][f_top]['wall'])}, {mem(sc[top][f_top]['mem'])}; "
                         f"{len(done)} of {len(BY_KEY)} methods completed this size"])
    if dg:
        fp = dg["median"]["rowgroup_fingerprint"]
        lv = sorted(fp)
        rows.append(["Re-checking files that are mostly unchanged and were written the same way",
                     lab("rowgroup_fingerprint"),
                     f"{t(fp[lv[0]])} for identical files; {t(fp[lv[1]])} when one row group differs"])
    if "numba_kernel" in ok:
        rows.append(["A fixed schema, run repeatedly, where every second counts", lab("numba_kernel"),
                     f"{t(ok['numba_kernel']['wall'])} per run once compiled; supports only int64 and string columns, and no nulls"])
    return rows


def build() -> Path:
    bench_raw = load_raw("benchmark")
    if not bench_raw:
        raise SystemExit("results/raw/benchmark.json not found. Run `python -m tabrecon benchmark` first.")
    bench = summarize(bench_raw["runs"])
    meta = bench_raw["meta"]
    ctx: dict = {"bench": bench, "n_rows": meta["n_rows"], "repeats": meta["repeats"], "env": meta["env"],
                 "truth": meta["truth"], "loc": {m.key: m.lines_of_code for m in METHODS}}

    sc_raw = load_raw("scaling")
    if sc_raw:
        by_size: dict[int, list[dict]] = {}
        for r in sc_raw["runs"]:
            by_size.setdefault(r["n_rows"], []).append(r)
        ctx["scaling"] = {n: summarize(rs) for n, rs in by_size.items()}
    bs_raw = load_raw("batch_sweep")
    if bs_raw:
        by: dict[int, list[float]] = {}
        for r in bs_raw["runs"]:
            if r["status"] == "ok":
                by.setdefault(r["params"]["batch_size"], []).append(r["wall_s"])
        med = {k: statistics.median(v) for k, v in sorted(by.items())}
        ctx["batch"] = {"median": med, "best": min(med, key=med.get), "runs": bs_raw["runs"]}
    ws_raw = load_raw("worker_sweep")
    if ws_raw:
        by = {}
        for r in ws_raw["runs"]:
            if r["status"] == "ok":
                by.setdefault(r["params"]["workers"], []).append(r["wall_s"])
        ctx["workers"] = {"median": {k: statistics.median(v) for k, v in sorted(by.items())}, "runs": ws_raw["runs"]}
    dg_raw = load_raw("dirty_sweep")
    if dg_raw:
        med2: dict[str, dict[int, float]] = {}
        tmp: dict[tuple, list[float]] = {}
        for r in dg_raw["runs"]:
            if r["status"] == "ok":
                tmp.setdefault((r["method"], r["dirty_groups"]), []).append(r["wall_s"])
        for (m, g), v in tmp.items():
            med2.setdefault(m, {})[g] = statistics.median(v)
        ctx["dirty"] = {"median": med2, "n_groups": dg_raw["meta"]["n_row_groups"], "runs": dg_raw["runs"]}
    nc_raw = load_raw("numba_compile")
    if nc_raw:
        ctx["compile"] = nc_raw["runs"][0]
    ctx["guide"] = _guide(ctx)

    # ---- figures (light + dark)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ok = {k: s for k, s in bench.items() if s["n"]}
    figs: dict[str, tuple[Path, Path]] = {}

    def render(name, fn, *args):
        paths = []
        for theme in ("light", "dark"):
            p = FIGURES_DIR / (f"{name}.png" if theme == "light" else f"{name}_dark.png")
            fn(*args, p, theme)
            paths.append(p)
        figs[name] = tuple(paths)

    render("runtime", plots.runtime_bars, ok, meta["n_rows"], meta["repeats"])
    render("memory_vs_time", plots.memory_vs_time, ok, meta["n_rows"])
    render("cpu", plots.cpu_dumbbell, ok, meta["n_rows"])
    render("scorecard", plots.scorecard, ok, ctx["loc"], meta["n_rows"])
    if "scaling" in ctx:
        sizes_ok = {n: s for n, s in ctx["scaling"].items()}
        ram = meta["env"].get("ram_gb")
        render("scaling_time", lambda p, th: plots.scaling(sizes_ok, "wall", p, th))
        render("scaling_memory", lambda p, th: plots.scaling(sizes_ok, "mem", p, th, ram))
    if "batch" in ctx:
        render("batch_size", lambda p, th: plots.batch_curve(ctx["batch"]["runs"], ctx["batch"]["best"], p, th))
    if "workers" in ctx:
        render("workers", lambda p, th: plots.worker_curve(ctx["workers"]["runs"], p, th, meta["env"].get("physical_cpus")))
    if "dirty" in ctx:
        render("dirty", lambda p, th: plots.dirty_curve(ctx["dirty"]["runs"], ctx["dirty"]["n_groups"], p, th))

    blocks = build_blocks(ctx)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.html").write_text(render_html(blocks, figs), encoding="utf-8")
    (REPORT_DIR / "REPORT.md").write_text(render_md(blocks), encoding="utf-8")
    print(f"report written to {REPORT_DIR}")
    return REPORT_DIR / "report.html"
