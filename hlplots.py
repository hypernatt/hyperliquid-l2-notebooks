"""Figures of the notebook. One function per figure; each returns the matplotlib Figure.

Colors: one fixed color per order-book view everywhere, one hue for magnitudes, two opposite hues around a
neutral midpoint for "more or less than usual", text always in ink colors. Every figure has its table printed
next to it in the notebook.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from hlbook import PX_SCALE, SZ_SCALE, VIEWS

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
VIEW_COLOR = {"REF": "#2a78d6", "M2": "#eb6834", "M5": "#1baf7a", "AGG4": "#eda100"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f0a8a8", "#e34948", "#a32626"])
HALO = [pe.withStroke(linewidth=3, foreground=SURFACE)]


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2, "axes.labelsize": 10.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "ytick.major.size": 0, "xtick.major.size": 3.5,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    })


def titles(ax, title: str, subtitle: str = "") -> None:
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", color=INK, pad=10 + 15 * lines)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, va="bottom", ha="left", fontsize=9.5, color=INK2,
                linespacing=1.35)


def save(fig, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def end_dot(ax, x, y, color, size: float = 7.5) -> None:
    ax.plot([x], [y], marker="o", ms=size, mfc=color, mec=SURFACE, mew=1.8, ls="none", zorder=5)


def label(ax, text: str, xy, offset, ha="left", va="center") -> None:
    ax.annotate(text, xy, xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=9.5, color=INK2,
                path_effects=HALO, zorder=6)


def plain(ax, axis: str, ticks) -> None:
    """Explicit ticks on a log axis, written as plain numbers."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(mpl.ticker.FixedLocator(list(ticks)))
    target.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
    target.set_minor_locator(mpl.ticker.NullLocator())


def utc_clock(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%H:%M:%S")


# --------------------------------------------------------------------------------------------------------------

def day_mid(bbo: pd.DataFrame, day: str):
    """Mid price through the day (one point per second: the last BBO of the second)."""
    t = pd.to_datetime(bbo["time"], unit="ms", utc=True)
    mid = ((bbo["bid"] + bbo["ask"]) / 2).set_axis(t)
    per_second = mid.resample("1s").last().dropna()
    values = mid.to_numpy()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(per_second.index, per_second.values, color=BLUE, lw=1.2)
    for i, name, va, dy in ((int(values.argmax()), "high", "bottom", 9), (int(values.argmin()), "low", "top", -9)):
        when, value = t.iloc[i], values[i]
        end_dot(ax, when, value, BLUE)
        label(ax, f"{name} {value:,.1f} at {when:%H:%M}", (when, value), (0, dy), ha="center", va=va)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M", tz="UTC"))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.margins(x=0.01, y=0.18)
    ax.set_xlabel("UTC")
    ax.set_ylabel("USD")
    titles(ax, "BTC mid price on the sample day", f"{day}, mid of the best bid and best ask, last value of each second.")
    return fig


def _stairs(dist: np.ndarray, cum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices of the cumulative-depth staircase: nothing before the first level, then one riser per level."""
    x = np.repeat(dist, 2)
    y = np.concatenate([[0.0], np.repeat(cum, 2)[:-1]])
    return x, y


def one_snapshot(books: dict, t: int, steps_usd: dict):
    """The same book at one exchange time, as each of the four views shows it. Two panels: full reach, zoom."""
    rows = {}
    for v in VIEWS:
        i = int(np.searchsorted(books[v].time, t))
        if books[v].time[i] != t:
            raise ValueError(f"{v} has no snapshot at {t}")
        rows[v] = i
    ref = books["REF"]
    mid = (ref.bid_px[rows["REF"], 0] + ref.ask_px[rows["REF"], 0]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 9.4), gridspec_kw={"hspace": 0.5})
    for ax, shown in zip(axes, (VIEWS, VIEWS[:2])):
        handles = {}
        for v in reversed(shown):                       # coarsest first, finest on top
            b, i = books[v], rows[v]
            for side, sign in (("bid", -1), ("ask", 1)):
                px, sz = b.side(side)
                ok = px[i] >= 0
                dist = sign * (px[i][ok] - mid) / mid * 1e4
                cum = np.cumsum(sz[i][ok]) / SZ_SCALE
                x, y = _stairs(dist, cum)
                handles[v], = ax.plot(sign * x, y, color=VIEW_COLOR[v], lw=2)
                end_dot(ax, sign * dist[-1], cum[-1], VIEW_COLOR[v])
                if side == "ask":
                    label(ax, v, (dist[-1], cum[-1]), (8, 0))
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_ylabel("cumulative BTC")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.08, y=0.08)
        ax.legend([handles[v] for v in shown], [f"{v} · {steps_usd[v]:g} USD per level" for v in shown],
                  loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1), handlelength=1.4, columnspacing=1.6)
    axes[1].set_xlabel("distance from the mid, bps  (bids to the left, asks to the right)", labelpad=30)
    titles(axes[0], "One snapshot of the BTC book, as each view shows it",
           f"Exchange time {utc_clock(t)} UTC, mid {mid / PX_SCALE:,.1f} USD. Each line ends where the view's 20 levels end.")
    titles(axes[1], "Same snapshot, close to the mid", "The two finest views only.")
    return fig


def depth_day(grid_bps: np.ndarray, bands: dict, reach_bps: dict, steps_usd: dict, n_snapshots: int,
              min_cover: float):
    """Stitched depth through the day: median and 10th-90th percentile band per side.

    The median line takes the color of the view that supplies each part of the curve (median reach of the view).
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    handles = {}
    for side, sign in (("bid", -1), ("ask", 1)):
        q10, q50, q90, cover = bands[side]
        keep = cover >= min_cover
        d, y = grid_bps[keep], q50[keep]
        ax.fill_between(sign * d, q10[keep], q90[keep], color=INK2, alpha=0.09, lw=0)
        start = 0.0
        for v in VIEWS:
            end = reach_bps[v][side]
            part = (d >= start - 1e-9) & (d <= end + 1e-9)
            if v == VIEWS[-1]:
                part = d >= start - 1e-9
            if part.any():
                handles[v], = ax.plot(sign * d[part], y[part], color=VIEW_COLOR[v], lw=2.4)
                end_dot(ax, sign * d[part][-1], y[part][-1], VIEW_COLOR[v])
                if side == "ask" and v == VIEWS[0]:
                    label(ax, v, (d[part][-1], y[part][-1]), (9, -5), ha="left", va="top")
                elif side == "ask":
                    label(ax, v, (d[part][-1], y[part][-1]), (-7, 8), ha="right", va="bottom")
            start = d[part][-1] if part.any() else start
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.03, y=0.06)
    ax.set_xlabel("distance from the mid, bps  (bids to the left, asks to the right)")
    ax.set_ylabel("cumulative BTC")
    ax.legend([handles[v] for v in VIEWS if v in handles],
              [f"{v} · {steps_usd[v]:g} USD per level" for v in VIEWS if v in handles],
              loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.15), handlelength=1.4, columnspacing=1.6)
    titles(ax, "Depth of the BTC book through the day, four views stitched",
           f"Line: median over {n_snapshots:,} snapshots, colored by the view that supplies that part of the curve.\n"
           "Band: 10th-90th percentile. Lower bound of the stitched curve.")
    return fig


def cost_by_size(sizes: np.ndarray, medians: dict, band: tuple, native_limit_btc: float):
    """Cost of a market order against its size: median per side, 10th-90th percentile band of both sides pooled."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.fill_between(sizes, band[0], band[1], color=INK2, alpha=0.09, lw=0)
    ends = []
    for side, color, name in (("ask", BLUE, "market buy"), ("bid", ORANGE, "market sell")):
        ax.plot(sizes, medians[side], color=color, lw=2, marker="o", ms=6.5, mfc=color, mec=SURFACE, mew=1.6,
                label=name)
        ends.append((float(medians[side][-1]), name))
    (low_y, low_name), (high_y, high_name) = sorted(ends)
    label(ax, high_name, (sizes[-1], high_y), (10, 7))
    label(ax, low_name, (sizes[-1], low_y), (10, -7))
    ax.axvline(native_limit_btc, color=AXIS, lw=1)
    ax.text(native_limit_btc, 0.04, f" beyond {native_limit_btc:.0f} BTC the native\n 20 levels run out\n (median snapshot)",
            transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=9, color=INK2, linespacing=1.3,
            path_effects=HALO)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain(ax, "x", sizes)
    plain(ax, "y", [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25])
    ax.margins(x=0.06, y=0.06)
    ax.set_xlabel("order size, BTC (log scale)")
    ax.set_ylabel("cost, bps from the mid (log scale)")
    ax.legend(loc="upper left", handlelength=1.4)
    titles(ax, "Cost of a market order by size, from the stitched book",
           "Lines: median over the day, per side. Band: 10th-90th percentile, both sides pooled.\n"
           "Upper bound: aggregated levels are priced at their far edge.")
    return fig


def cost_by_hour(ratio: pd.DataFrame):
    """Cost of each hour relative to the day's median for the same size; diverging colors around 1."""
    values = ratio.to_numpy(dtype=float)
    span = float(np.nanmax(np.abs(np.log(values))))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=DIVERGING, edgecolors=SURFACE, linewidth=2,
                         norm=LogNorm(vmin=float(np.exp(-span)), vmax=float(np.exp(span))))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(values.shape[1]) + 0.5, [f"{h:02d}" for h in ratio.columns])
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, [f"{s:g}" for s in ratio.index])
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("hour of the day, UTC")
    ax.set_ylabel("order size, BTC")
    bar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=18)
    bar.outline.set_visible(False)
    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8) if np.exp(-span) <= t <= np.exp(span)]
    bar.set_ticks(ticks, labels=[f"{t:g}x" for t in ticks])
    bar.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    bar.ax.tick_params(length=0, labelsize=9, labelcolor=INK2)
    bar.set_label("cost / median cost of the day for that size", color=INK2, fontsize=9.5)
    titles(ax, "When a market order costs more, hour by hour",
           "Median cost of each hour divided by the day's median for the same size (upper bound, mean of buy and sell).\n"
           "Empty cell: the size was larger than what the book showed in more than half of the hour's snapshots.")
    return fig


def order_sizes(orders: pd.DataFrame):
    """Share of taker orders, and share of the BTC they traded, that come from orders of at least a given size."""
    size = np.sort(orders["size"].to_numpy())
    n = len(size)
    by_count = 1.0 - np.arange(n) / n
    by_volume = 1.0 - np.concatenate([[0.0], np.cumsum(size)[:-1]]) / size.sum()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(size, by_count, color=BLUE, lw=2, label="share of orders")
    ax.plot(size, by_volume, color=ORANGE, lw=2, label="share of BTC traded")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-4)
    ax.set_ylim(1.0 / n * 0.7, 2.2)
    plain(ax, "x", [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000])
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{v * 100:g} %"))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    one = int(np.searchsorted(size, 1.0))
    for curve, color, what, dy in ((by_count, BLUE, "orders", 9), (by_volume, ORANGE, "BTC traded", 10)):
        end_dot(ax, size[one], curve[one], color)
        label(ax, f"{curve[one] * 100:.1f} % of {what}", (size[one], curve[one]), (9, dy))
    ax.set_xlabel("order size, BTC (log scale)")
    ax.set_ylabel("share from orders of at least this size (log scale)")
    ax.legend(loc="lower left", handlelength=1.4)
    titles(ax, "Taker orders of the day by size",
           f"{n:,} taker orders (fills sharing a transaction hash and a side). Marked: orders of at least 1 BTC.")
    return fig


def realized_vs_book(table: pd.DataFrame):
    """Per size class: realized cost of the orders and cost of the same sizes on the last book before them."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(table))
    for dx, color, mid, lo, hi, name in (
            (-0.13, BLUE, "book_median", "book_q25", "book_q75", "stitched book just before the order"),
            (0.13, ORANGE, "realized_median", "realized_q25", "realized_q75", "realized by the order")):
        ax.vlines(x + dx, table[lo], table[hi], color=color, lw=1.6, alpha=0.55)
        ax.plot(x + dx, table[mid], ls="none", marker="o", ms=8, mfc=color, mec=SURFACE, mew=1.8, label=name)
    ax.set_yscale("log")
    plain(ax, "y", [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25])
    ax.set_xticks(x, [f"{c}\n{n:,} orders" for c, n in zip(table.index, table["orders"])])
    ax.tick_params(axis="x", length=0)
    ax.margins(x=0.06, y=0.08)
    ax.set_xlabel("order size, BTC")
    ax.set_ylabel("cost, bps from the mid (log scale)")
    ax.legend(loc="upper left", handlelength=1.0)
    titles(ax, "What taker orders paid, against the stitched book",
           "Dots: median per size class; bars: 25th-75th percentile. Book cost: upper bound, on the last snapshot\n"
           "before the order. Realized cost: average fill price against the last BBO mid before the order.")
    return fig
