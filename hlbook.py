"""Plain functions for the HyperNatt Hyperliquid L2 sample.

Download and integrity check, readers for the recorded streams, the bucket-by-bucket comparison of the four
order-book views, the stitched cumulative depth curve, the cost of a market order with its two bounds, and the
grouping of fills into taker orders. Standard library, numpy and pandas only.

Units. Prices and sizes are parsed once into integers (tenths of a dollar, 1e-5 BTC) so that sums and
comparisons are exact; they are converted back to dollars and BTC at the edge of each function that reports.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE_URL = "https://hypernatt.com/datasets/release.json"
SAMPLE_SOURCES = (
    "https://hypernatt.com/datasets/{filename}",
    "https://huggingface.co/datasets/hypernatt/hyperliquid-l2-sample/resolve/main/{filename}",
    "https://zenodo.org/records/24151403/files/{filename}?download=1",
)
SAMPLE_DOI = "10.5281/zenodo.24151403"
USER_AGENT = "hyperliquid-l2-notebooks"

VIEWS = ("REF", "M2", "M5", "AGG4")  # finest to coarsest
PX_SCALE = 10        # integer price unit = 0.1 USD
SZ_SCALE = 100_000   # integer size unit = 0.00001 BTC
ZERO_HASH = "0x" + "0" * 64


class IntegrityError(RuntimeError):
    """The file on disk is not the file described by release.json."""


# --------------------------------------------------------------------------------------------------------------
# Download and integrity
# --------------------------------------------------------------------------------------------------------------

def fetch_release(url: str = RELEASE_URL) -> dict:
    """Return the published release.json (file names, sizes and SHA-256 of every archive and of the sample)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_sample(path: Path, release: dict) -> str:
    """Return the SHA-256 of `path`; raise IntegrityError unless size and SHA-256 match release.json."""
    want = release["sample"]
    size = Path(path).stat().st_size
    if size != want["bytes"]:
        raise IntegrityError(f"{path}: {size} bytes, release.json says {want['bytes']}")
    got = sha256_file(path)
    if got != want["sha256"]:
        raise IntegrityError(f"{path}: SHA-256 {got}, release.json says {want['sha256']}")
    return got


def download_sample(release: dict, dest_dir: Path, sources: tuple[str, ...] = SAMPLE_SOURCES) -> Path:
    """Download the free sample named in release.json into dest_dir (or reuse a verified copy) and verify it."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = release["sample"]["filename"]
    path = dest_dir / filename
    if path.exists():
        verify_sample(path, release)
        return path
    errors = []
    for template in sources:
        url = template.format(filename=filename)
        part = path.with_suffix(path.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            verify_sample(part, release)
            part.replace(path)
            return path
        except Exception as exc:  # try the next copy; every copy must pass the same SHA-256 check
            errors.append(f"{url}: {exc}")
            part.unlink(missing_ok=True)
    raise IntegrityError("no source delivered the published file:\n" + "\n".join(errors))


# --------------------------------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------------------------------

class Sample:
    """Read-only access to the sample zip (hourly .jsonl.gz files, one directory per stream)."""

    def __init__(self, zip_path: Path, coin: str = "BTC"):
        self.zip = zipfile.ZipFile(zip_path)
        self.names = self.zip.namelist()
        self.root = self.names[0].split("/")[0]
        self.coin = coin

    def streams(self) -> list[str]:
        prefix = f"{self.root}/data/{self.coin}/"
        return sorted({n[len(prefix):].split("/")[0] for n in self.names if n.startswith(prefix)})

    def hour_files(self, stream: str) -> list[str]:
        prefix = f"{self.root}/data/{self.coin}/{stream}/"
        return sorted(n for n in self.names if n.startswith(prefix) and not n.endswith("/"))

    def read_json(self, name: str) -> dict:
        return json.loads(self.zip.read(f"{self.root}/{name}").decode("utf-8"))

    def records(self, stream: str):
        """Yield the exchange payload of every recorded DATA event of a stream, in file order."""
        for name in self.hour_files(stream):
            with gzip.open(io.BytesIO(self.zip.read(name)), "rt", encoding="utf-8") as f:
                for line in f:
                    event = json.loads(line)
                    if event.get("event_type") == "DATA":
                        yield event["raw"]


def _to_units(values: np.ndarray, scale: int, what: str) -> np.ndarray:
    scaled = values * scale
    units = np.rint(scaled)
    if np.nanmax(np.abs(scaled - units), initial=0.0) > 1e-6:
        raise ValueError(f"{what} has more decimals than 1/{scale}")
    return np.where(np.isnan(units), -1, units).astype(np.int64)


@dataclass
class Books:
    """All snapshots of one order-book view. Levels are best first; absent levels are -1."""

    view: str
    time: np.ndarray     # (N,) exchange time, ms
    bid_px: np.ndarray   # (N, L) price units
    bid_sz: np.ndarray   # (N, L) size units
    ask_px: np.ndarray
    ask_sz: np.ndarray
    duplicates: int = 0  # snapshots dropped because their exchange time was already seen

    def __len__(self) -> int:
        return len(self.time)

    def side(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return (self.bid_px, self.bid_sz) if name == "bid" else (self.ask_px, self.ask_sz)


def load_books(sample: Sample, view: str) -> Books:
    """Read every snapshot of one view (prices and sizes only)."""
    times, rows, seen, dup = [], {"bp": [], "bs": [], "ap": [], "as": []}, set(), 0
    width = 0
    for raw in sample.records(f"l2Book_{view}"):
        t = raw["time"]
        if t in seen:
            dup += 1
            continue
        seen.add(t)
        bids, asks = raw["levels"]
        times.append(t)
        rows["bp"].append([float(x["px"]) for x in bids])
        rows["bs"].append([float(x["sz"]) for x in bids])
        rows["ap"].append([float(x["px"]) for x in asks])
        rows["as"].append([float(x["sz"]) for x in asks])
        width = max(width, len(bids), len(asks))

    def pad(lists):
        out = np.full((len(lists), width), np.nan)
        for i, row in enumerate(lists):
            out[i, : len(row)] = row
        return out

    order = np.argsort(np.asarray(times, dtype=np.int64), kind="stable")
    return Books(
        view=view,
        time=np.asarray(times, dtype=np.int64)[order],
        bid_px=_to_units(pad(rows["bp"]), PX_SCALE, "price")[order],
        bid_sz=_to_units(pad(rows["bs"]), SZ_SCALE, "size")[order],
        ask_px=_to_units(pad(rows["ap"]), PX_SCALE, "price")[order],
        ask_sz=_to_units(pad(rows["as"]), SZ_SCALE, "size")[order],
        duplicates=dup,
    )


def load_bbo(sample: Sample) -> pd.DataFrame:
    """Best bid and best ask prices of every BBO message: columns time (ms), bid, ask (USD)."""
    t, b, a = [], [], []
    for raw in sample.records("bbo"):
        bid, ask = raw["bbo"]
        if bid is None or ask is None:
            continue
        t.append(raw["time"])
        b.append(float(bid["px"]))
        a.append(float(ask["px"]))
    out = pd.DataFrame({"time": np.asarray(t, dtype=np.int64), "bid": b, "ask": a})
    return out.sort_values("time", kind="stable").reset_index(drop=True)


def load_trades(sample: Sample) -> pd.DataFrame:
    """Every fill: time (ms), side (+1 the buyer was the taker, -1 the seller), px (USD), sz (BTC), tx, tid."""
    codes: dict[str, int] = {}
    t, s, p, q, h, i = [], [], [], [], [], []
    for raw in sample.records("trades"):
        for fill in raw if isinstance(raw, list) else [raw]:
            t.append(fill["time"])
            s.append(1 if fill["side"] == "B" else -1)
            p.append(float(fill["px"]))
            q.append(float(fill["sz"]))
            h.append(codes.setdefault(fill["hash"], len(codes)))
            i.append(fill["tid"])
    out = pd.DataFrame({"time": np.asarray(t, dtype=np.int64), "side": np.asarray(s, dtype=np.int8), "px": p,
                        "sz": q, "tx": np.asarray(h, dtype=np.int64), "tid": np.asarray(i, dtype=np.int64)})
    out.attrs["zero_tx"] = codes.get(ZERO_HASH, -1)
    return out


# --------------------------------------------------------------------------------------------------------------
# Chapter 1 - what each view sees
# --------------------------------------------------------------------------------------------------------------

def price_step(books: Books) -> int:
    """Price grid of a view, in price units: the greatest common divisor of every price it ever shows."""
    px = np.concatenate([books.bid_px.ravel(), books.ask_px.ravel()])
    return int(np.gcd.reduce(np.unique(px[px >= 0])))


def view_reach(books: Books) -> pd.DataFrame:
    """Per snapshot: mid (USD), distance from the mid to the last visible level (bps) and BTC shown, per side."""
    n = len(books)
    last = lambda px: px[np.arange(n), (px >= 0).sum(axis=1) - 1]  # noqa: E731
    mid = (books.bid_px[:, 0] + books.ask_px[:, 0]) / 2
    return pd.DataFrame({
        "time": books.time,
        "mid": mid / PX_SCALE,
        "reach_bid_bps": (mid - last(books.bid_px)) / mid * 1e4,
        "reach_ask_bps": (last(books.ask_px) - mid) / mid * 1e4,
        "btc_bid": np.where(books.bid_px >= 0, books.bid_sz, 0).sum(axis=1) / SZ_SCALE,
        "btc_ask": np.where(books.ask_px >= 0, books.ask_sz, 0).sum(axis=1) / SZ_SCALE,
    })


# --------------------------------------------------------------------------------------------------------------
# Chapter 2 - do the views agree?
# --------------------------------------------------------------------------------------------------------------

def bucket_of(px: int, step: int, side: str) -> int:
    """Price of the aggregated level that holds a finer price: bids are floored, asks are ceiled."""
    return (px // step) * step if side == "bid" else -((-px) // step) * step


def cross_check(fine: Books, coarse: Books) -> dict:
    """Compare two views on the snapshots that carry the same exchange time.

    Every level of the coarse view that lies entirely inside the price range shown by the fine view must hold
    exactly the sum of the fine levels that fall into it. Returns the counts and the largest gap found.
    """
    step, fine_step = price_step(coarse), price_step(fine)
    if step % fine_step:
        raise ValueError(f"{coarse.view} step is not a multiple of {fine.view} step")
    _, i_fine, i_coarse = np.intersect1d(fine.time, coarse.time, return_indices=True)
    width = fine.bid_px.shape[1]
    tested = equal = worst = 0
    for name in ("bid", "ask"):
        fpx_all, fsz_all = (a[i_fine].tolist() for a in fine.side(name))
        cpx_all, csz_all = (a[i_coarse].tolist() for a in coarse.side(name))
        for fpx, fsz, cpx, csz in zip(fpx_all, fsz_all, cpx_all, csz_all):
            sums: dict[int, int] = {}
            shown = 0
            for p, s in zip(fpx, fsz):
                if p < 0:
                    break
                shown += 1
                key = bucket_of(p, step, name)
                sums[key] = sums.get(key, 0) + s
            # a fine side with fewer levels than the view can show is the whole side: everything is covered
            limit = fpx[shown - 1] if shown == width else None
            for p, s in zip(cpx, csz):
                if p < 0:
                    break
                if limit is not None and (p < limit if name == "bid" else p > limit):
                    break
                tested += 1
                gap = abs(sums.get(p, 0) - s)
                equal += gap == 0
                worst = max(worst, gap)
    return {
        "coarse": coarse.view, "fine": fine.view,
        "coarse_step_usd": step / PX_SCALE, "fine_step_usd": fine_step / PX_SCALE,
        "snapshots_same_time": int(len(i_fine)),
        "levels_tested": int(tested), "levels_equal": int(equal),
        "largest_gap_btc": worst / SZ_SCALE,
    }


# --------------------------------------------------------------------------------------------------------------
# Chapter 3 - one depth curve from four views
# --------------------------------------------------------------------------------------------------------------

def stitch_side(levels: list[tuple[list[int], list[int]]], side: str) -> tuple[list[int], list[int], list[int]]:
    """Stitch one side of one snapshot. `levels` holds (prices, sizes) of each view, finest first.

    Each view gives exact points of the same curve: the size resting between the best price and the far edge
    of each of its levels. The finest view is used as far as it reaches, then the next one takes over.
    Returns the edge prices, the cumulative sizes and the index of the view that supplied each point.
    """
    sign = -1 if side == "bid" else 1
    edge, cum, src = [], [], []
    for v, (px, sz) in enumerate(levels):
        total = 0
        for p, s in zip(px, sz):
            if p < 0:
                break
            total += s
            if not edge or sign * (p - edge[-1]) > 0:
                if cum and total < cum[-1]:
                    raise ValueError("views disagree: cumulative size decreases along the stitched curve")
                edge.append(p)
                cum.append(total)
                src.append(v)
    return edge, cum, src


@dataclass
class Stitched:
    """Stitched books of every snapshot whose exchange time is common to all the views. -1 pads short rows."""

    time: np.ndarray       # (N,)
    mid: np.ndarray        # (N,) USD
    bid_edge: np.ndarray   # (N, K) price units, moving away from the mid
    bid_cum: np.ndarray    # (N, K) size units
    bid_src: np.ndarray    # (N, K) index into VIEWS
    ask_edge: np.ndarray
    ask_cum: np.ndarray
    ask_src: np.ndarray
    tick: int              # native price step, price units

    def __len__(self) -> int:
        return len(self.time)

    def side(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if name == "bid":
            return self.bid_edge, self.bid_cum, self.bid_src
        return self.ask_edge, self.ask_cum, self.ask_src


def common_times(books: dict[str, Books]) -> np.ndarray:
    times = books[VIEWS[0]].time
    for v in VIEWS[1:]:
        times = np.intersect1d(times, books[v].time)
    return times


def stitch(books: dict[str, Books]) -> Stitched:
    """Build the stitched book of every snapshot whose exchange time is present in the four views."""
    times = common_times(books)
    index = {v: np.searchsorted(books[v].time, times) for v in VIEWS}
    width = sum(books[v].bid_px.shape[1] for v in VIEWS)
    out = {f"{s}_{k}": np.full((len(times), width), -1, dtype=np.int64) for s in ("bid", "ask")
           for k in ("edge", "cum", "src")}
    for name in ("bid", "ask"):
        per_view = [[a[index[v]].tolist() for a in books[v].side(name)] for v in VIEWS]
        for i in range(len(times)):
            edge, cum, src = stitch_side([(px[i], sz[i]) for px, sz in per_view], name)
            k = len(edge)
            out[f"{name}_edge"][i, :k] = edge
            out[f"{name}_cum"][i, :k] = cum
            out[f"{name}_src"][i, :k] = src
    mid = (out["bid_edge"][:, 0] + out["ask_edge"][:, 0]) / 2 / PX_SCALE
    return Stitched(time=times, mid=mid, tick=price_step(books[VIEWS[0]]), **out)


def depth_within(st: Stitched, bps: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """BTC resting within `bps` of the mid, per snapshot: (at least, at most), NaN beyond the last point.

    Inside the native view the answer is exact. Farther out the size of an aggregated level sits somewhere
    between its two edges, hence the two bounds.
    """
    edge, cum, src = st.side(side)
    sign = -1 if side == "bid" else 1
    bps = np.asarray(bps, dtype=float)
    low = np.full((len(st), len(bps)), np.nan)
    high = np.full((len(st), len(bps)), np.nan)
    for i in range(len(st)):
        k = int((edge[i] >= 0).sum())
        dist = sign * (edge[i, :k] / PX_SCALE - st.mid[i]) / st.mid[i] * 1e4
        c = cum[i, :k] / SZ_SCALE
        j = np.searchsorted(dist, bps, side="right") - 1   # last point at or inside the distance
        covered = bps <= dist[-1]
        lo = np.where(j >= 0, c[np.clip(j, 0, k - 1)], 0.0)
        nxt = np.clip(j + 1, 0, k - 1)
        exact = (src[i, nxt] == 0) | (j == k - 1) | ((j >= 0) & (dist[np.clip(j, 0, k - 1)] == bps))
        hi = np.where(exact, lo, c[nxt])
        low[i] = np.where(covered, lo, np.nan)
        high[i] = np.where(covered, hi, np.nan)
    return low, high


# --------------------------------------------------------------------------------------------------------------
# Chapter 4 - cost of a market order
# --------------------------------------------------------------------------------------------------------------

def sweep_cost(edge_usd: np.ndarray, cum_btc: np.ndarray, src: np.ndarray, mid: float, tick_usd: float,
               sizes_btc: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Cost of taking `sizes_btc` from one side of one stitched book, in bps from the mid: (low, high).

    side="ask" prices a market buy, side="bid" a market sell. The size of an aggregated level is priced at
    its near edge for the low bound and at its far edge for the high bound; native levels have one price.
    Sizes larger than what the book shows return NaN.
    """
    sign = -1.0 if side == "bid" else 1.0
    seg = np.diff(cum_btc, prepend=0.0)
    far = edge_usd.astype(float)
    near = far.copy()
    coarse = src > 0
    near[coarse] = np.roll(far, 1)[coarse] + sign * tick_usd
    paid_far = np.concatenate([[0.0], np.cumsum(seg * far)])
    paid_near = np.concatenate([[0.0], np.cumsum(seg * near)])
    filled = np.concatenate([[0.0], cum_btc])
    sizes = np.asarray(sizes_btc, dtype=float)
    k = np.searchsorted(cum_btc, sizes, side="left")          # level that completes the order
    ok = k < len(cum_btc)
    kk = np.clip(k, 0, len(cum_btc) - 1)
    rest = sizes - filled[kk]
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_far = (paid_far[kk] + rest * far[kk]) / sizes
        avg_near = (paid_near[kk] + rest * near[kk]) / sizes
        hi = np.where(ok, sign * (avg_far - mid) / mid * 1e4, np.nan)
        lo = np.where(ok, sign * (avg_near - mid) / mid * 1e4, np.nan)
    return lo, hi


def sweep_costs(st: Stitched, sizes_btc: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """sweep_cost on every stitched snapshot: two (N, len(sizes)) arrays, low and high bound, in bps."""
    edge, cum, src = st.side(side)
    low = np.full((len(st), len(sizes_btc)), np.nan)
    high = np.full((len(st), len(sizes_btc)), np.nan)
    for i in range(len(st)):
        k = int((edge[i] >= 0).sum())
        low[i], high[i] = sweep_cost(edge[i, :k] / PX_SCALE, cum[i, :k] / SZ_SCALE, src[i, :k], st.mid[i],
                                     st.tick / PX_SCALE, sizes_btc, side)
    return low, high


# --------------------------------------------------------------------------------------------------------------
# Chapter 5 - the trades that happened
# --------------------------------------------------------------------------------------------------------------

def taker_orders(trades: pd.DataFrame) -> pd.DataFrame:
    """Group fills into taker orders: fills that share a transaction hash and a side.

    Fills that carry the all-zero hash are not tied to one transaction and are left out; their count and size
    are returned in `attrs`. Columns: time (first fill, ms), side, size (BTC), vwap (USD), fills, prices.
    """
    zero = trades["tx"] == trades.attrs.get("zero_tx", -1)
    kept = trades[~zero]
    notional = kept["px"] * kept["sz"]
    grouped = kept.assign(notional=notional).groupby(["tx", "side"], sort=False)
    out = grouped.agg(time=("time", "min"), size=("sz", "sum"), notional=("notional", "sum"),
                      fills=("sz", "size"), prices=("px", "nunique")).reset_index()
    out["vwap"] = out.pop("notional") / out["size"]
    out = out.drop(columns="tx").sort_values("time", kind="stable").reset_index(drop=True)
    out.attrs["zero_hash_fills"] = int(zero.sum())
    out.attrs["zero_hash_btc"] = float(trades.loc[zero, "sz"].sum())
    return out


def realized_cost(orders: pd.DataFrame, bbo: pd.DataFrame) -> pd.DataFrame:
    """Add to each order the mid of the last BBO strictly before it and its realized cost in bps from that mid."""
    j = np.searchsorted(bbo["time"].to_numpy(), orders["time"].to_numpy(), side="left") - 1
    mid = ((bbo["bid"] + bbo["ask"]) / 2).to_numpy()
    out = orders.copy()
    out["mid_before"] = np.where(j >= 0, mid[np.clip(j, 0, None)], np.nan)
    out["realized_bps"] = out["side"] * (out["vwap"] - out["mid_before"]) / out["mid_before"] * 1e4
    return out


def book_cost_before(orders: pd.DataFrame, st: Stitched) -> pd.DataFrame:
    """Add to each order the cost bounds of its size on the last stitched book strictly before it."""
    j = np.searchsorted(st.time, orders["time"].to_numpy(), side="left") - 1
    low = np.full(len(orders), np.nan)
    high = np.full(len(orders), np.nan)
    age = np.full(len(orders), np.nan)
    sizes = orders["size"].to_numpy()
    sides = orders["side"].to_numpy()
    times = orders["time"].to_numpy()
    for n, i in enumerate(j):
        if i < 0:
            continue
        name = "ask" if sides[n] > 0 else "bid"
        edge, cum, src = st.side(name)
        k = int((edge[i] >= 0).sum())
        lo, hi = sweep_cost(edge[i, :k] / PX_SCALE, cum[i, :k] / SZ_SCALE, src[i, :k], st.mid[i],
                            st.tick / PX_SCALE, np.array([sizes[n]]), name)
        low[n], high[n], age[n] = lo[0], hi[0], (times[n] - st.time[i]) / 1000
    out = orders.copy()
    out["book_low_bps"], out["book_high_bps"], out["book_age_s"] = low, high, age
    return out


# --------------------------------------------------------------------------------------------------------------
# Small helpers for reporting
# --------------------------------------------------------------------------------------------------------------

def quantiles(values: np.ndarray, qs=(0.1, 0.5, 0.9)) -> list[float]:
    """Quantiles that ignore NaN; NaN when nothing is left."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, q)) if len(v) else math.nan for q in qs]


def column_quantiles(values: np.ndarray, qs=(0.1, 0.5, 0.9)) -> list[np.ndarray]:
    """Quantiles of each column, ignoring NaN; a column with no value gives NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [np.nanquantile(values, q, axis=0) for q in qs]


def rounded(obj, digits: int = 6):
    """Round every float of a nested structure (stable JSON output); NaN becomes None."""
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, digits) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else round(float(obj), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
