"""Write README.md from results/summary.json (the file the notebook writes). No figure is typed by hand.

    python tools/render_readme.py            write README.md
    python tools/render_readme.py --check    fail if README.md is not the current output
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWS = ("REF", "M2", "M5", "AGG4")


def render(summary: dict) -> str:
    rel, views, check = summary["release"], summary["views"], summary["cross_check"]
    cost, stitched, orders = summary["cost"], summary["stitched"], summary["orders"]
    day = f"{rel['day'][:4]}-{rel['day'][4:6]}-{rel['day'][6:]}"

    view_rows = "\n".join(
        f"| `{v}` | {views[v]['price step, USD']:g} | {views[v]['reach ask, bps (median)']:.1f} "
        f"| {views[v]['BTC bid, median']:.0f} | {views[v]['BTC ask, median']:.0f} |" for v in VIEWS)
    cost_rows = "\n".join(
        f"| {size} | {row['buy: low, bps']:.3f} - {row['buy: high, bps']:.3f} "
        f"| {row['sell: low, bps']:.3f} - {row['sell: high, bps']:.3f} "
        f"| {row['priceable, four views, %']:.1f} % | {row['priceable, native view alone, %']:.1f} % |"
        for size, row in sorted(cost.items(), key=lambda kv: float(kv[0].split()[0])))

    return f"""# Hyperliquid L2 notebooks

Reproducible analyses of the free 24-hour sample of the [HyperNatt Hyperliquid L2 archive](https://hypernatt.com/datasets).

## The BTC order book at four resolutions

Notebook: [`notebooks/four_resolutions.ipynb`](notebooks/four_resolutions.ipynb)

Hyperliquid's public `l2Book` stream returns 20 price levels per side, on the native price step or on a coarser
price grid. The archive records four of these views side by side. The notebook measures what each view shows,
checks that the four views describe the same book, stitches them into one depth curve, prices market orders on
that curve and compares the result with the taker orders executed that day.

It describes one day of data ({day}, BTC). It is not trading advice.

![One snapshot of the book, as each view shows it](results/01_one_snapshot.png)

### Results on the sample day

Every figure below is copied from `results/summary.json`, which the notebook writes.

**What each view shows** (medians over the day)

| View | Price step, USD | Reach from the mid, bps | BTC on the bid side | BTC on the ask side |
|---|---|---|---|---|
{view_rows}

**Do the views agree?** On snapshots that carry the same exchange time, {check['levels_equal']:,} of
{check['levels_tested']:,} aggregated levels hold exactly the sum of the finer levels that fall into them
(largest gap: {check['largest_gap_btc']:g} BTC). {check['snapshots_same_time_4_views']:,} snapshots carry the same
exchange time in the four views ({100 * check['share_same_time_4_views']:.1f} % of the snapshots of a view).

**One depth curve.** Stitched, the four views show {stitched['btc_ratio_stitched_over_native_median']:.1f} times
the BTC of the native view and reach {stitched['reach_ratio_stitched_over_native_median']:.1f} times farther from
the mid (median snapshot).

![Depth of the book through the day](results/02_depth_day.png)

**Cost of a market order** (bps from the mid, median over the day, low - high bound) and how often each size can
be priced at all

| Order size | Buy | Sell | Priceable, four views | Priceable, native view alone |
|---|---|---|---|---|
{cost_rows}

![Cost of a market order by size](results/03_cost_by_size.png)

![When a market order costs more, hour by hour](results/04_cost_by_hour.png)

**The orders that were executed.** {orders['taker orders']:,} taker orders (fills that share a transaction hash
and a side) traded on the sample day; {orders['orders of at least 1 BTC']:,} of them were of at least 1 BTC and
carried {orders['share of BTC traded by orders of at least 1 BTC, %']:.1f} % of the BTC traded. The notebook
compares what these orders paid with the cost of the same sizes on the stitched book just before them.

![What taker orders paid, against the stitched book](results/06_realized_vs_book.png)

Definitions, tables and limits are in the notebook.

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt        # Windows: .venv\\Scripts\\pip
.venv/bin/python tools/run_notebook.py --check       # executes the notebook, compares results/summary.json
.venv/bin/python -m pytest -q                        # unit tests on hand-made books
```

The notebook downloads the sample ({rel['bytes']:,} bytes) from hypernatt.com, with Hugging Face and Zenodo as
fallbacks, and stops if its SHA-256 differs from the one published in
[`release.json`](https://hypernatt.com/datasets/release.json). Dependencies: numpy, pandas and matplotlib, with
exact versions in `requirements.txt`.

## Data

| | |
|---|---|
| File | `{rel['filename']}` |
| SHA-256 | `{rel['sha256']}` |
| DOI | [{rel['doi']}](https://doi.org/{rel['doi']}) |
| Licence of the sample | {rel['licence']} (attribution terms in `LICENSE.txt` inside the archive) |
| Release | {rel['release_id']} |

The full archive has the same format for BTC and ETH: <https://hypernatt.com/datasets>.

## Repository

| Path | Content |
|---|---|
| `notebooks/four_resolutions.ipynb` | the notebook, executed, with its tables and figures |
| `hlbook.py` | download and SHA-256 check, readers, cross-check, stitched depth curve, cost of an order |
| `hlplots.py` | the figures |
| `tests/` | unit tests |
| `results/` | `summary.json` and the figures written by the notebook |
| `tools/` | `run_notebook.py`, `render_readme.py` |

## Licence

Code: MIT (see `LICENSE`). Sample data: {rel['licence']}.
"""


def main() -> int:
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    text = render(summary)
    target = ROOT / "README.md"
    if "--check" in sys.argv:
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            print("README.md is not the output of tools/render_readme.py")
            return 1
        print("README.md is up to date")
        return 0
    target.write_bytes(text.encode("utf-8"))
    print("README.md written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
