"""Unit tests of hlbook.py on hand-made books (no download, no sample needed)."""
import gzip
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import hlbook
from hlbook import PX_SCALE, SZ_SCALE


def make_books(view, snapshots, width=4):
    """snapshots: list of (time, bids, asks) with levels as (price USD, size BTC), best first."""
    n = len(snapshots)
    arr = {k: np.full((n, width), -1, dtype=np.int64) for k in ("bp", "bs", "ap", "as")}
    for i, (_t, bids, asks) in enumerate(snapshots):
        for j, (p, s) in enumerate(bids[:width]):
            arr["bp"][i, j], arr["bs"][i, j] = round(p * PX_SCALE), round(s * SZ_SCALE)
        for j, (p, s) in enumerate(asks[:width]):
            arr["ap"][i, j], arr["as"][i, j] = round(p * PX_SCALE), round(s * SZ_SCALE)
    return hlbook.Books(view=view, time=np.array([s[0] for s in snapshots], dtype=np.int64),
                        bid_px=arr["bp"], bid_sz=arr["bs"], ask_px=arr["ap"], ask_sz=arr["as"])


# One snapshot. Native book (step 1 USD) and its exact aggregation on a 2 USD grid.
FINE = (1000, [(100, 1.0), (99, 2.0), (98, 3.0), (97, 4.0)], [(101, 1.5), (102, 2.5), (103, 3.5), (104, 4.5)])
# bids floored: 100->[100,102), 99 and 98->[98,100), 97->[96,98) plus an unseen 96 (5.0)
# asks ceiled: 101 and 102->(100,102], 103 and 104->(102,104], then 105/106 unseen (6.0)
COARSE = (1000, [(100, 1.0), (98, 5.0), (96, 9.0), (94, 7.0)], [(102, 4.0), (104, 8.0), (106, 6.0), (108, 2.0)])


def test_bucket_of_floors_bids_and_ceils_asks():
    assert hlbook.bucket_of(995, 20, "bid") == 980
    assert hlbook.bucket_of(980, 20, "bid") == 980
    assert hlbook.bucket_of(1005, 20, "ask") == 1020
    assert hlbook.bucket_of(1020, 20, "ask") == 1020


def test_price_step_is_the_grid_of_the_view():
    assert hlbook.price_step(make_books("REF", [FINE])) == 1 * PX_SCALE
    assert hlbook.price_step(make_books("M2", [COARSE])) == 2 * PX_SCALE


def test_cross_check_counts_only_levels_fully_inside_the_fine_range():
    res = hlbook.cross_check(make_books("REF", [FINE]), make_books("M2", [COARSE]))
    # bids: 100 and 98 are covered (>= 97 is the last fine bid); 96 is not (its bucket starts below 97)
    # asks: 102 and 104 are covered (<= 104 is the last fine ask); 106 is not
    assert res["snapshots_same_time"] == 1
    assert res["levels_tested"] == 4
    assert res["levels_equal"] == 4
    assert res["largest_gap_btc"] == 0


def test_cross_check_reports_a_wrong_level():
    wrong = (1000, [(100, 1.0), (98, 5.25), (96, 9.0), (94, 7.0)], COARSE[2])
    res = hlbook.cross_check(make_books("REF", [FINE]), make_books("M2", [wrong]))
    assert res["levels_tested"] == 4
    assert res["levels_equal"] == 3
    assert res["largest_gap_btc"] == pytest.approx(0.25)


def test_cross_check_ignores_snapshots_with_another_exchange_time():
    later = (2000, COARSE[1], COARSE[2])
    res = hlbook.cross_check(make_books("REF", [FINE]), make_books("M2", [later]))
    assert res["snapshots_same_time"] == 0 and res["levels_tested"] == 0


def test_cross_check_treats_a_short_fine_side_as_the_whole_side():
    fine = (1000, [(100, 1.0), (99, 2.0)], [(101, 1.5)])            # fewer levels than the view can show
    coarse = (1000, [(100, 1.0), (98, 2.0)], [(102, 1.5)])
    res = hlbook.cross_check(make_books("REF", [fine]), make_books("M2", [coarse]))
    assert res["levels_tested"] == 3 and res["levels_equal"] == 3


def units(levels):
    return [round(p * PX_SCALE) for p, _ in levels], [round(s * SZ_SCALE) for _, s in levels]


def test_stitch_side_uses_the_finest_view_first_then_extends_with_the_coarser_one():
    edge, cum, src = hlbook.stitch_side([units(FINE[2]), units(COARSE[2])], "ask")
    assert [e / PX_SCALE for e in edge] == [101, 102, 103, 104, 106, 108]
    assert [c / SZ_SCALE for c in cum] == [1.5, 4.0, 7.5, 12.0, 18.0, 20.0]
    assert src == [0, 0, 0, 0, 1, 1]
    edge, cum, src = hlbook.stitch_side([units(FINE[1]), units(COARSE[1])], "bid")
    assert [e / PX_SCALE for e in edge] == [100, 99, 98, 97, 96, 94]
    assert [c / SZ_SCALE for c in cum] == [1.0, 3.0, 6.0, 10.0, 15.0, 22.0]
    assert src == [0, 0, 0, 0, 1, 1]


def test_stitch_side_refuses_views_that_disagree():
    smaller = [(102, 4.0), (104, 8.0), (106, 0.1), (108, 0.1)]
    shrunk = ([round(p * PX_SCALE) for p, _ in smaller], [round(0.1 * SZ_SCALE)] * 4)
    with pytest.raises(ValueError):
        hlbook.stitch_side([units(FINE[2]), shrunk], "ask")


def stitched_one():
    books = {"REF": make_books("REF", [FINE]), "M2": make_books("M2", [COARSE])}
    for v in ("M5", "AGG4"):
        books[v] = make_books(v, [COARSE])
    return hlbook.stitch(books)


def test_stitch_keeps_only_times_common_to_the_four_views():
    st = stitched_one()
    assert len(st) == 1 and st.mid[0] == pytest.approx(100.5) and st.tick == PX_SCALE
    books = {v: make_books(v, [FINE if v == "REF" else COARSE]) for v in hlbook.VIEWS}
    books["M5"] = make_books("M5", [(999, COARSE[1], COARSE[2])])
    assert len(hlbook.stitch(books)) == 0


def test_depth_within_is_exact_in_the_native_range_and_bounded_beyond():
    st = stitched_one()
    bps = lambda usd: usd / 100.5 * 1e4  # noqa: E731
    grid = np.array([bps(0.4), bps(1.5), bps(3.5), bps(4.5), bps(5.5), bps(7.5), bps(8.0)])
    low, high = hlbook.depth_within(st, grid, "ask")
    # 0.4 USD from the mid: nothing; 1.5 -> 101 and 102; 3.5 -> up to 104 (end of the native view)
    assert low[0, :3].tolist() == [0.0, 4.0, 12.0] and high[0, :3].tolist() == [0.0, 4.0, 12.0]
    # 4.5 USD -> 105: inside the aggregated level (104, 106]: between 12 and 18 BTC
    assert (low[0, 3], high[0, 3]) == (12.0, 18.0)
    # 5.5 USD -> exactly 106, the edge of that level: exact again
    assert (low[0, 4], high[0, 4]) == (18.0, 18.0)
    # 7.5 USD -> exactly 108, the last point; 8.0 USD is beyond what the views show
    assert (low[0, 5], high[0, 5]) == (20.0, 20.0)
    assert np.isnan(low[0, 6]) and np.isnan(high[0, 6])


def test_sweep_cost_bounds_on_a_hand_made_book():
    edge = np.array([101.0, 102.0, 110.0])
    cum = np.array([1.0, 3.0, 10.0])
    src = np.array([0, 0, 1])
    low, high = hlbook.sweep_cost(edge, cum, src, 100.0, 1.0, np.array([2.0, 5.0, 10.0, 11.0]), "ask")
    assert low[0] == pytest.approx(150.0) and high[0] == pytest.approx(150.0)      # (101 + 102) / 2 = 101.5
    assert low[1] == pytest.approx(220.0)                                          # 2 BTC at the near edge 103
    assert high[1] == pytest.approx(500.0)                                         # 2 BTC at the far edge 110
    assert low[2] == pytest.approx((305 + 7 * 103) / 10 * 100 - 10000)
    assert high[2] == pytest.approx((305 + 7 * 110) / 10 * 100 - 10000)
    assert np.isnan(low[3]) and np.isnan(high[3])                                  # more than the book shows


def test_sweep_cost_sell_side_is_the_mirror():
    edge = np.array([99.0, 98.0, 90.0])
    cum = np.array([1.0, 3.0, 10.0])
    src = np.array([0, 0, 1])
    low, high = hlbook.sweep_cost(edge, cum, src, 100.0, 1.0, np.array([2.0, 5.0]), "bid")
    assert low[0] == pytest.approx(150.0) and high[0] == pytest.approx(150.0)
    assert low[1] == pytest.approx(220.0) and high[1] == pytest.approx(500.0)


def test_sweep_costs_runs_on_every_snapshot():
    st = stitched_one()
    low, high = hlbook.sweep_costs(st, np.array([1.0, 15.0, 50.0]), "ask")
    assert low.shape == (1, 3)
    assert low[0, 0] == pytest.approx((101 - 100.5) / 100.5 * 1e4)
    assert low[0, 1] < high[0, 1]            # the last 3 BTC come from an aggregated level
    assert np.isnan(low[0, 2])


def trades_frame(rows, zero_tx=-1):
    df = pd.DataFrame(rows, columns=["time", "side", "px", "sz", "tx", "tid"])
    df.attrs["zero_tx"] = zero_tx
    return df


def test_taker_orders_groups_by_transaction_and_side():
    trades = trades_frame([
        (10, 1, 100.0, 1.0, 7, 1), (10, 1, 101.0, 3.0, 7, 2),      # one buy order, two fills
        (11, -1, 99.0, 2.0, 8, 3),                                 # one sell order
        (12, 1, 100.0, 5.0, 0, 4), (40, -1, 98.0, 6.0, 0, 5),      # all-zero hash: left out
    ], zero_tx=0)
    orders = hlbook.taker_orders(trades)
    assert len(orders) == 2
    buy = orders[orders["side"] == 1].iloc[0]
    assert buy["size"] == 4.0 and buy["vwap"] == pytest.approx(100.75) and buy["fills"] == 2 and buy["prices"] == 2
    assert orders.attrs["zero_hash_fills"] == 2 and orders.attrs["zero_hash_btc"] == 11.0


def test_realized_cost_uses_the_last_bbo_strictly_before_the_order():
    orders = pd.DataFrame({"time": [10, 20], "side": [1, -1], "size": [1.0, 1.0], "vwap": [101.0, 99.0]})
    bbo = pd.DataFrame({"time": [5, 10, 15], "bid": [99.0, 100.0, 99.5], "ask": [101.0, 102.0, 100.5]})
    out = hlbook.realized_cost(orders, bbo)
    assert out["mid_before"].tolist() == [100.0, 100.0]            # the BBO stamped 10 is not before 10
    assert out["realized_bps"].tolist() == pytest.approx([100.0, 100.0])


def test_book_cost_before_uses_the_last_stitched_book_before_the_order():
    st = stitched_one()                                            # one book, stamped 1000
    orders = pd.DataFrame({"time": [900, 1500], "side": [1, -1], "size": [1.0, 1.0], "vwap": [0.0, 0.0]})
    out = hlbook.book_cost_before(orders, st)
    assert np.isnan(out["book_low_bps"][0])                        # no book before the first order
    assert out["book_high_bps"][1] == pytest.approx((100.5 - 100) / 100.5 * 1e4)
    assert out["book_age_s"][1] == pytest.approx(0.5)


def write_sample(tmp_path, events_by_stream):
    root = "SAMPLE"
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{root}/CONTINUITY.json", json.dumps({"streams": {}}))
        for stream, events in events_by_stream.items():
            body = "\n".join(json.dumps(e) for e in events).encode()
            z.writestr(f"{root}/data/BTC/{stream}/20260911_00.jsonl.gz", gzip.compress(body))
    return path


def test_readers_parse_the_recorded_format(tmp_path):
    level = lambda p, s: {"px": str(p), "sz": str(s), "n": 3}  # noqa: E731
    book = lambda t, b, a: {"event_type": "DATA", "raw": {"coin": "BTC", "time": t, "levels": [b, a]}}  # noqa: E731
    path = write_sample(tmp_path, {
        "l2Book_REF": [
            {"event_type": "CONNECT"},
            book(2000, [level("100.0", "0.5")], [level("101.0", "0.25")]),
            book(1000, [level("99.0", "1.0"), level("98.0", "2.0")], [level("100.0", "0.00001")]),
            book(2000, [level("1.0", "1.0")], [level("2.0", "1.0")]),            # same exchange time: dropped
        ],
        "bbo": [{"event_type": "DATA", "raw": {"coin": "BTC", "time": 5, "bbo": [level("99.0", "1"), level("100.0", "1")]}},
                {"event_type": "DATA", "raw": {"coin": "BTC", "time": 6, "bbo": [None, level("100.0", "1")]}}],
        "trades": [{"event_type": "DATA", "raw": [
            {"coin": "BTC", "side": "B", "px": "100.0", "sz": "0.5", "time": 7, "hash": "0xab", "tid": 1, "users": ["a", "b"]},
            {"coin": "BTC", "side": "A", "px": "99.0", "sz": "0.1", "time": 8, "hash": hlbook.ZERO_HASH, "tid": 2, "users": ["a", "b"]},
        ]}],
    })
    sample = hlbook.Sample(path)
    assert sample.streams() == ["bbo", "l2Book_REF", "trades"]
    books = hlbook.load_books(sample, "REF")
    assert books.time.tolist() == [1000, 2000] and books.duplicates == 1
    assert books.bid_px[0].tolist() == [990, 980] and books.bid_sz[0].tolist() == [100000, 200000]
    assert books.ask_sz[0, 0] == 1 and books.ask_px[1].tolist() == [1010, -1]
    bbo = hlbook.load_bbo(sample)
    assert bbo["time"].tolist() == [5] and bbo["ask"].tolist() == [100.0]
    trades = hlbook.load_trades(sample)
    assert trades["side"].tolist() == [1, -1] and trades.attrs["zero_tx"] == 1
    assert len(hlbook.taker_orders(trades)) == 1


def test_a_size_with_too_many_decimals_is_refused(tmp_path):
    book = {"event_type": "DATA", "raw": {"coin": "BTC", "time": 1, "levels": [
        [{"px": "100.0", "sz": "0.000001"}], [{"px": "101.0", "sz": "1.0"}]]}}
    with pytest.raises(ValueError):
        hlbook.load_books(hlbook.Sample(write_sample(tmp_path, {"l2Book_REF": [book]})), "REF")


def release_for(path, filename="sample.bin"):
    data = path.read_bytes()
    return {"sample": {"filename": filename, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}


def test_verify_sample_refuses_another_file(tmp_path):
    good = tmp_path / "good.bin"
    good.write_bytes(b"published bytes")
    release = release_for(good)
    assert hlbook.verify_sample(good, release) == release["sample"]["sha256"]
    same_size = tmp_path / "same_size.bin"
    same_size.write_bytes(b"PUBLISHED BYTES")
    with pytest.raises(hlbook.IntegrityError, match="SHA-256"):
        hlbook.verify_sample(same_size, release)
    shorter = tmp_path / "shorter.bin"
    shorter.write_bytes(b"published")
    with pytest.raises(hlbook.IntegrityError, match="bytes"):
        hlbook.verify_sample(shorter, release)


def test_download_sample_falls_back_and_never_keeps_a_wrong_file(tmp_path):
    origin, mirror, dest = tmp_path / "origin", tmp_path / "mirror", tmp_path / "dest"
    for d in (origin, mirror):
        d.mkdir()
    (mirror / "sample.bin").write_bytes(b"published bytes")
    (origin / "sample.bin").write_bytes(b"tampered bytes!")
    release = release_for(mirror / "sample.bin")
    sources = (origin.as_uri() + "/{filename}", (tmp_path / "missing").as_uri() + "/{filename}",
               mirror.as_uri() + "/{filename}")
    got = hlbook.download_sample(release, dest, sources)
    assert got.read_bytes() == b"published bytes"
    assert sorted(p.name for p in dest.iterdir()) == ["sample.bin"]              # no .part left behind
    with pytest.raises(hlbook.IntegrityError):
        hlbook.download_sample(release, tmp_path / "dest2", sources[:2])
    assert list((tmp_path / "dest2").iterdir()) == []


def test_rounded_makes_stable_json():
    assert hlbook.rounded({"a": [np.float64(1.23456789), float("nan")], "b": np.int64(3)}, 3) == {"a": [1.235, None], "b": 3}
