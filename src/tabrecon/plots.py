"""Figures for the report, rendered in a light and a dark variant.

Colour rules (see the dataviz method): the three methods added in this study carry
fixed categorical hues (blue, orange, aqua), the seven original methods share one
neutral grey, and every mark is directly labelled so colour is never the only cue.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from tabrecon.methods.registry import BY_KEY, NEW  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", orig="#898781", orig_dark="#52514e",
                  new={"parallel_row_groups": "#2a78d6", "rowgroup_fingerprint": "#eb6834", "numba_kernel": "#1baf7a"},
                  ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", orig="#898781", orig_dark="#c3c2b7",
                 new={"parallel_row_groups": "#3987e5", "rowgroup_fingerprint": "#d95926", "numba_kernel": "#199e70"},
                 ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]),
}
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def color_for(key: str, t: dict) -> str:
    return t["new"].get(key, t["orig"])


def label_for(key: str) -> str:
    return BY_KEY[key].label


def _setup(t: dict, figsize=(9, 5.2)):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10})
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=t["ink2"], length=0)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    return fig, ax


def _title(ax, t, title, subtitle=None):
    ax.set_title(title, loc="left", color=t["ink"], fontsize=13, fontweight="semibold", pad=26 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=9.5, va="bottom")


def _family_legend(ax, t, keys, loc="upper right", **kw):
    handles = [Patch(facecolor=t["orig"], label="Original methods")]
    handles += [Patch(facecolor=t["new"][k], label=f"New: {label_for(k)}") for k in keys if k in t["new"]]
    leg = ax.legend(handles=handles, loc=loc, frameon=False, fontsize=8.5, labelcolor=t["ink2"], **kw)
    return leg


def _save(fig, path: Path, t):
    fig.savefig(path, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Nudge sorted label positions apart so labels do not collide (positions in any linear unit)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for _ in range(50):
        moved = False
        for j in range(1, len(pos)):
            if pos[j] - pos[j - 1] < min_gap:
                push = (min_gap - (pos[j] - pos[j - 1])) / 2
                pos[j - 1] -= push
                pos[j] += push
                moved = True
        if not moved:
            break
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = pos[rank]
    return out


def _log_ticks(ax, axis: str, values: list[float], fmt) -> None:
    """Label a log axis at 1-2-5 steps that bracket the data."""
    lo, hi = min(values), max(values)
    cand = [m * 10.0 ** e for e in range(-4, 6) for m in (1, 2, 5)]
    ticks = [c for c in cand if lo / 2.5 <= c <= hi * 2.5]
    if axis == "y":
        ax.set_yticks(ticks, [fmt(c) for c in ticks])
        ax.minorticks_off()
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.set_xticks(ticks, [fmt(c) for c in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _fmt_time(s: float) -> str:
    return f"{s:.2f} s" if s >= 0.1 else f"{s * 1000:.0f} ms"


def _fmt_mem(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


# ------------------------------------------------------------------------------- figures
def runtime_bars(stats: dict, n_rows: int, repeats: int, path: Path, theme: str) -> None:
    t = THEMES[theme]
    keys = sorted(stats, key=lambda k: stats[k]["wall"])
    fig, ax = _setup(t, (9, 0.5 * len(keys) + 1.8))
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        s = stats[k]
        ax.barh(yi, s["wall"], height=0.56, color=color_for(k, t))
        ax.plot([s["wall_min"], s["wall_max"]], [yi, yi], color=t["ink"], lw=1, alpha=0.55, solid_capstyle="butt")
        ax.text(max(s["wall_max"], s["wall"]) + 0.01 * max(v["wall_max"] for v in stats.values()), yi,
                _fmt_time(s["wall"]), va="center", color=t["ink"], fontsize=9)
    ax.set_yticks(y, [label_for(k) for k in keys], color=t["ink"])
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, max(v["wall_max"] for v in stats.values()) * 1.15)
    _title(ax, t, f"Time to reconcile {n_rows / 1e6:g}M rows",
           f"Median of up to {repeats} runs, end to end including reading both files · whisker = fastest to slowest run")
    _family_legend(ax, t, list(t["new"]), loc="upper right")
    _save(fig, path, t)


def memory_vs_time(stats: dict, n_rows: int, path: Path, theme: str) -> None:
    t = THEMES[theme]
    fig, ax = _setup(t)
    pts = {k: (s["mem"], s["wall"]) for k, s in stats.items()}
    # Pareto frontier: no other method is both faster and lighter.
    front = sorted((m, w, k) for k, (m, w) in pts.items()
                   if not any(m2 <= m and w2 <= w and (m2, w2) != (m, w) for m2, w2 in pts.values()))
    if len(front) > 1:
        ax.step([p[0] for p in front], [p[1] for p in front], where="post", color=t["muted"], lw=1.2, ls="--", alpha=0.8)
    xs = [m for m, _ in pts.values()]
    ys = [w for _, w in pts.values()]
    ax.set_yscale("log")
    for k, (m, w) in pts.items():
        ax.scatter(m, w, s=70, color=color_for(k, t), edgecolor=t["surface"], linewidth=1.5, zorder=3)
    # label with small vertical nudges in log space
    ly = _spread([math.log10(w) for w in ys], 0.055)
    for (k, (m, w)), yy in zip(pts.items(), ly):
        ax.annotate(label_for(k), (m, w), xytext=(m + (max(xs) - min(xs)) * 0.025, 10 ** yy), color=t["ink"],
                    fontsize=8.5, va="center",
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.5, shrinkA=0, shrinkB=4))
    ax.set_xlim(min(xs) - (max(xs) - min(xs)) * 0.08, max(xs) + (max(xs) - min(xs)) * 0.38)
    ax.set_xlabel("Peak memory, whole process tree (MB)")
    ax.set_ylabel("Time (s, log scale)")
    _log_ticks(ax, "y", ys, lambda v: f"{v:g}")
    ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.grid(axis="x", color=t["grid"], lw=0.6)
    _title(ax, t, "Speed against memory", f"{n_rows / 1e6:g}M rows · bottom-left is better · dashed line = best trade-offs")
    ax.legend(handles=[Patch(facecolor=t["orig"], label="Original methods")]
              + [Patch(facecolor=t["new"][k], label=f"New: {label_for(k)}") for k in t["new"]],
              loc="upper right", frameon=False, fontsize=8.5, labelcolor=t["ink2"])
    _save(fig, path, t)


def cpu_dumbbell(stats: dict, n_rows: int, path: Path, theme: str) -> None:
    t = THEMES[theme]
    keys = sorted(stats, key=lambda k: stats[k]["wall"])
    fig, ax = _setup(t, (9, 0.5 * len(keys) + 1.9))
    y = np.arange(len(keys))[::-1]
    ax.set_xscale("log")
    for yi, k in zip(y, keys):
        s = stats[k]
        c = color_for(k, t)
        ax.plot([s["wall"], s["cpu"]], [yi, yi], color=c, lw=2.2, solid_capstyle="round", alpha=0.9)
        ax.scatter(s["wall"], yi, s=60, facecolor=t["surface"], edgecolor=c, linewidth=2, zorder=3)
        ax.scatter(s["cpu"], yi, s=60, color=c, zorder=3)
        ax.text(max(s["cpu"], s["wall"]) * 1.12, yi, f"{s['cpu'] / s['wall']:.1f}× cores busy",
                va="center", color=t["ink2"], fontsize=8.5)
    ax.set_yticks(y, [label_for(k) for k in keys], color=t["ink"])
    lo = min(min(s["wall"], s["cpu"]) for s in stats.values())
    hi = max(max(s["wall"], s["cpu"]) for s in stats.values())
    ax.set_xlim(lo / 1.5, hi * 3.2)
    ax.set_xlabel("Seconds (log scale)")
    ax.grid(axis="x", color=t["grid"], lw=0.6)
    _title(ax, t, "Clock time against CPU time",
           f"{n_rows / 1e6:g}M rows · the gap shows how many cores the method kept busy")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=t["surface"], markeredgecolor=t["ink2"],
                              markeredgewidth=2, label="Wall-clock time"),
                       Line2D([], [], marker="o", ls="", color=t["ink2"], label="CPU-seconds (all cores, summed)")],
              loc="upper right", frameon=False, fontsize=8.5, labelcolor=t["ink2"])
    _log_ticks(ax, "x", [lo, hi], lambda v: f"{v:g}")
    _save(fig, path, t)


def scorecard(stats: dict, loc: dict, n_rows: int, path: Path, theme: str) -> None:
    t = THEMES[theme]
    keys = sorted(stats, key=lambda k: stats[k]["wall"])
    cols = [("Time", "wall", lambda v: _fmt_time(v)), ("Peak memory", "mem", _fmt_mem),
            ("CPU time", "cpu", lambda v: f"{v:.1f} s"), ("Code size", "loc", lambda v: f"{v:.0f} lines")]
    vals = np.array([[stats[k]["wall"], stats[k]["mem"], stats[k]["cpu"], loc[k]] for k in keys], dtype=float)
    ratio = vals / vals.min(axis=0)
    score = np.log(ratio) / max(np.log(ratio).max(), 1e-9)
    fig, ax = _setup(t, (9, 0.5 * len(keys) + 1.8))
    ramp = t["ramp"]
    import matplotlib.colors as mc
    cmap = mc.LinearSegmentedColormap.from_list("seq", ramp[:6])
    for i, k in enumerate(keys):
        for j, (_, _, fmt) in enumerate(cols):
            ax.add_patch(plt.Rectangle((j + 0.04, len(keys) - 1 - i + 0.06), 0.92, 0.88,
                                       color=cmap(score[i, j]), lw=0))
            r, g, b, _ = cmap(score[i, j])
            fg = "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"
            ax.text(j + 0.5, len(keys) - 1 - i + 0.5, f"{fmt(vals[i, j])}\n{ratio[i, j]:.1f}× best" if ratio[i, j] > 1.005
                    else f"{fmt(vals[i, j])}\nbest", ha="center", va="center", color=fg, fontsize=8)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(keys))
    ax.set_xticks([j + 0.5 for j in range(len(cols))], [c[0] for c in cols], color=t["ink"])
    ax.xaxis.tick_top()
    ax.set_yticks([len(keys) - 1 - i + 0.5 for i in range(len(keys))], [label_for(k) for k in keys], color=t["ink"])
    ax.text(0, 1.2, "Scorecard: cost of each method relative to the best in its column", transform=ax.transAxes,
            color=t["ink"], fontsize=13, fontweight="semibold", va="bottom")
    ax.text(0, 1.145, f"{n_rows / 1e6:g}M rows · lighter is cheaper · every column is lower-is-better",
            transform=ax.transAxes, color=t["ink2"], fontsize=9.5, va="bottom")
    _save(fig, path, t)


def scaling(stats_by_size: dict[int, dict], metric: str, path: Path, theme: str, ram_gb: float | None = None) -> None:
    """Lines of metric vs dataset size per method; a cross marks where the memory guard stopped a method."""
    t = THEMES[theme]
    sizes = sorted(stats_by_size)
    keys = [k for k in BY_KEY if any(k in stats_by_size[n] for n in sizes)]
    fig, ax = _setup(t, (9.6, 5.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ends = []
    for k in keys:
        pts = [(n, stats_by_size[n][k][metric]) for n in sizes
               if k in stats_by_size[n] and stats_by_size[n][k]["n"] > 0]
        if not pts:
            continue
        new = k in t["new"]
        c = color_for(k, t)
        ax.plot(*zip(*pts), color=c, lw=2.2 if new else 1.4, alpha=1 if new else 0.85,
                marker="o", ms=4.5 if new else 3.5, zorder=3 if new else 2)
        fail_n = [n for n in sizes if k in stats_by_size[n] and stats_by_size[n][k]["n"] == 0]
        if fail_n:
            ax.scatter(fail_n[0], pts[-1][1], marker="X", s=70, color=c, edgecolor=t["surface"], zorder=4)
        ends.append((k, pts[-1][0], pts[-1][1], bool(fail_n)))
    ylog = [math.log10(e[2]) for e in ends]
    ly = _spread(ylog, 0.09)
    xmax = max(sizes)
    for (k, n, v, failed), yy in zip(ends, ly):
        ax.annotate(label_for(k) + ("  (stopped)" if failed else ""), (n, v),
                    xytext=(xmax * 1.18, 10 ** yy), color=t["ink"], fontsize=8.3, va="center",
                    arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.5, shrinkA=0, shrinkB=3))
    ax.set_xlim(min(sizes) * 0.8, xmax * 9)
    ax.set_xlabel("Rows per file (log scale)")
    ax.set_ylabel("Time (s, log scale)" if metric == "wall" else "Peak memory (MB, log scale)")
    ax.set_xticks(sizes, [f"{n / 1e6:g}M" for n in sizes])
    all_y = [stats_by_size[n][k][metric] for n in sizes for k in stats_by_size[n] if stats_by_size[n][k]["n"]]
    _log_ticks(ax, "y", all_y, (lambda v: f"{v:g}") if metric == "wall" else (lambda v: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(color=t["grid"], lw=0.6)
    title = "How runtime grows with dataset size" if metric == "wall" else "How peak memory grows with dataset size"
    _title(ax, t, title, "Median per size · a cross marks the size where a method was stopped by the memory guard")
    ax.legend(handles=[Patch(facecolor=t["orig"], label="Original methods")]
              + [Patch(facecolor=t["new"][k], label=f"New: {label_for(k)}") for k in t["new"]],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.5, labelcolor=t["ink2"])
    _save(fig, path, t)


def batch_curve(rows: list[dict], best: int, path: Path, theme: str) -> None:
    t = THEMES[theme]
    by: dict[int, list[float]] = {}
    for r in rows:
        if r["status"] == "ok":
            by.setdefault(r["params"]["batch_size"], []).append(r["wall_s"])
    xs = sorted(by)
    med = [float(np.median(by[x])) for x in xs]
    lo = [min(by[x]) for x in xs]
    hi = [max(by[x]) for x in xs]
    fig, ax = _setup(t, (9, 4.8))
    ax.set_xscale("log")
    c = t["orig"]
    ax.fill_between(xs, lo, hi, color=c, alpha=0.2, lw=0)
    ax.plot(xs, med, color=t["orig_dark"], lw=2, marker="o", ms=4.5)
    bi = xs.index(best)
    ax.scatter(best, med[bi], s=110, color=t["ink"], edgecolor=t["surface"], linewidth=1.5, zorder=4)
    ax.annotate(f"fastest: {best:,} rows\n{_fmt_time(med[bi])}", (best, med[bi]), xytext=(0, -42), textcoords="offset points",
                ha="center", color=t["ink"], fontsize=9, arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.6))
    ax.annotate(f"one batch per file\n{_fmt_time(med[-1])}", (xs[-1], med[-1]), xytext=(-10, 26), textcoords="offset points",
                ha="right", color=t["ink2"], fontsize=8.5)
    ax.set_xlabel("Batch size (rows, log scale)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.set_ylim(0, max(hi) * 1.1)
    _title(ax, t, "PyArrow runtime against batch size", "Line = median of repeats · band = fastest to slowest run")
    _save(fig, path, t)


def worker_curve(rows: list[dict], path: Path, theme: str, physical: int | None) -> None:
    t = THEMES[theme]
    by: dict[int, list[float]] = {}
    for r in rows:
        if r["status"] == "ok":
            by.setdefault(r["params"]["workers"], []).append(r["wall_s"])
    xs = sorted(by)
    med = [float(np.median(by[x])) for x in xs]
    c = t["new"]["parallel_row_groups"]
    fig, ax = _setup(t, (9, 4.8))
    ax.plot(xs, med, color=c, lw=2.2, marker="o", ms=6)
    ideal = [med[0] * xs[0] / x for x in xs]
    ax.plot(xs, ideal, color=t["muted"], lw=1.2, ls="--")
    ax.text(xs[-1], ideal[-1] * 0.85, "perfect scaling", color=t["ink2"], fontsize=8.5, ha="right", va="top")
    for x, m in zip(xs, med):
        ax.annotate(f"{_fmt_time(m)}\n{med[0] / m:.1f}×", (x, m), xytext=(0, 11), textcoords="offset points",
                    ha="center", color=t["ink"], fontsize=8.5)
    if physical:
        ax.axvline(physical, color=t["muted"], lw=1, ls=":")
        ax.text(physical, ax.get_ylim()[1] * 0.97, f" {physical} physical cores", color=t["ink2"], fontsize=8.5, va="top")
    ax.set_xticks(xs)
    ax.set_xlabel("Worker processes")
    ax.set_ylabel("Time (s)")
    ax.set_ylim(0, max(med) * 1.25)
    ax.grid(axis="y", color=t["grid"], lw=0.6)
    _title(ax, t, "Parallel row groups: runtime against worker count",
           "Labels show median time and speed-up over one worker")
    _save(fig, path, t)


def dirty_curve(rows: list[dict], n_groups: int, path: Path, theme: str) -> None:
    t = THEMES[theme]
    by: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        if r["status"] == "ok":
            by.setdefault(r["method"], {}).setdefault(r["dirty_groups"], []).append(r["wall_s"])
    fig, ax = _setup(t, (9, 4.9))
    ends = []
    for k, d in by.items():
        xs = sorted(d)
        med = [float(np.median(d[x])) for x in xs]
        new = k in t["new"]
        ax.plot(xs, med, color=color_for(k, t) if new else (t["orig_dark"] if k == "pyarrow_tuned" else t["orig"]),
                lw=2.4 if new else 1.6, ls="-" if k != "polars_vectorized" else (0, (4, 2)), marker="o", ms=5)
        ends.append((k, xs[-1], med[-1]))
    ly = _spread([e[2] for e in ends], max(e[2] for e in ends) * 0.07)
    for (k, x, v), yy in zip(ends, ly):
        ax.annotate(label_for(k), (x, v), xytext=(n_groups * 1.04, yy), color=t["ink"], fontsize=8.5, va="center")
    ax.set_xlim(-0.5, n_groups * 1.5)
    ax.set_xlabel(f"Row groups containing a difference (out of {n_groups})")
    ax.set_ylabel("Time (s)")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=t["grid"], lw=0.6)
    _title(ax, t, "Fingerprinting pays off only when few row groups differ",
           "Same number of changed rows each time; only how widely they are spread changes")
    _save(fig, path, t)
