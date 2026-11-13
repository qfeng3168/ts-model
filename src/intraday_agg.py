"""Aggregate tick files to daily intraday features (v1, 13 features).

Reads data/ticks/{code}/{yyyymmdd}.parquet, writes one checkpoint parquet
per code to data/intraday_parts/{code}.parquet (resumable), then merges
to data/intraday_v1.parquet with the two 60d-relative features and the
reconciliation columns vs data/daily.parquet.

Definitions (validated on pilot 000001/20261106, see
reports/intraday_v1_pilot_check.md):
- buyorsell 0 = active buy, 1 = active sell, 8 = auction rebroadcast
  (vol=0), others neutral. A-group numerators count 0/1 only.
- auction window: time < 09:30; continuous: 09:30-11:30, 13:00-15:00.
- rows after 15:00 (after-hours fixed price) are excluded everywhere.
- vol unit is lots (100 shares); tick volume x100 reconciles vs daily.
"""
from __future__ import annotations

import os
import sys
import time as _time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TICKS = ROOT / "data" / "ticks"
PARTS = ROOT / "data" / "intraday_parts"

RAW_COLS = [
    "date", "code", "tick_vol", "tick_close", "tick_open", "amount",
    "ofi_day", "ofi_am", "ofi_pm", "ofi_last30", "buy_share",
    "large_share", "avg_trade_size", "tick_count",
    "vol_conc_am", "vol_conc_last30", "vwap_dev", "range_tick",
    "auction_vol_share", "auction_price",
]


def _minutes(t: np.ndarray) -> np.ndarray:
    s = pd.Series(t)
    return (s.str.slice(0, 2).astype(int) * 60
            + s.str.slice(3, 5).astype(int)).to_numpy()


def day_features(path: Path) -> dict | None:
    df = pd.read_parquet(path, columns=["time", "price", "vol", "buyorsell"])
    if len(df) == 0:
        return None
    mins = _minutes(df["time"].to_numpy())
    price = df["price"].to_numpy(dtype=np.float64)
    vol = df["vol"].to_numpy(dtype=np.float64)
    bs = df["buyorsell"].to_numpy()
    keep = mins <= 15 * 60  # drop after-hours rows
    mins, price, vol, bs = mins[keep], price[keep], vol[keep], bs[keep]
    if vol.sum() <= 0:
        return None
    traded = vol > 0
    total_vol = float(vol.sum())
    amount = float((price * vol).sum())  # in lots*price; x100 = yuan
    vwap = amount / total_vol
    close = float(price[traded][-1])
    # opening print: first traded row at/after 09:25 (auction match)
    op_idx = np.where(traded & (mins >= 9 * 60 + 25))[0]
    open_px = float(price[op_idx[0]]) if len(op_idx) else float(price[traded][0])
    auc = mins < 9 * 60 + 30
    auction_vol = float(vol[auc].sum())
    auc_traded = auc & traded
    auction_price = float(price[auc_traded][-1]) if auc_traded.any() else np.nan

    cont = (mins >= 9 * 60 + 30) & (mins <= 15 * 60)
    am = cont & (mins <= 11 * 60 + 30)
    pm = cont & (mins >= 13 * 60)
    first30 = cont & (mins < 10 * 60)
    last30 = cont & (mins >= 14 * 60 + 30)

    def ofi(mask: np.ndarray) -> float:
        b = float(vol[mask & (bs == 0)].sum())
        s = float(vol[mask & (bs == 1)].sum())
        tv = float(vol[mask].sum())
        return (b - s) / tv if tv > 0 else np.nan

    cvol = vol[cont]
    buy_c = float(vol[cont & (bs == 0)].sum())
    n_trades = int(cont.sum())
    if n_trades >= 10:
        sv = np.sort(cvol)[::-1]
        k = max(1, int(np.ceil(0.1 * n_trades)))
        large_share = float(sv[:k].sum() / cvol.sum()) if cvol.sum() > 0 else np.nan
    else:
        large_share = np.nan
    return {
        "date": path.stem,
        "tick_vol": total_vol * 100.0,
        "tick_close": close,
        "tick_open": open_px,
        "amount": amount * 100.0,
        "ofi_day": ofi(cont),
        "ofi_am": ofi(am),
        "ofi_pm": ofi(pm),
        "ofi_last30": ofi(last30),
        "buy_share": buy_c / float(cvol.sum()) if cvol.sum() > 0 else np.nan,
        "large_share": large_share,
        "avg_trade_size": float(cvol.mean()) if n_trades else np.nan,
        "tick_count": n_trades,
        "vol_conc_am": float(vol[first30].sum()) / total_vol,
        "vol_conc_last30": float(vol[last30].sum()) / total_vol,
        "vwap_dev": (close - vwap) / vwap,
        "range_tick": (float(price[traded].max()) - float(price[traded].min()))
                      / open_px,
        "auction_vol_share": auction_vol / total_vol,
        "auction_price": auction_price,
    }


def process_code(code: str) -> str:
    out = PARTS / f"{code}.parquet"
    if out.exists():
        return f"{code} skip"
    files = sorted((TICKS / code).glob("*.parquet"))
    rows = []
    for f in files:
        try:
            r = day_features(f)
        except Exception:
            r = None
        if r is not None:
            r["code"] = code
            rows.append(r)
    if not rows:
        return f"{code} EMPTY"
    df = pd.DataFrame(rows)[RAW_COLS]
    PARTS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return f"{code} ok {len(df)}"


def finalize() -> None:
    parts = [pd.read_parquet(p) for p in sorted(PARTS.glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", group_keys=False)
    df["avg_trade_size_rel"] = df["avg_trade_size"] / g["avg_trade_size"].transform(
        lambda s: s.rolling(60, min_periods=20).mean())
    df["tick_count_rel"] = df["tick_count"] / g["tick_count"].transform(
        lambda s: s.rolling(60, min_periods=20).mean())
    prev_close = g["tick_close"].shift(1)
    df["amihud_tick"] = (df["tick_close"] / prev_close - 1).abs() \
        / df["amount"] * 1e8
    df["auction_gap"] = df["auction_price"] / prev_close - 1
    # reconciliation vs daily.parquet (raw prices, 300 codes)
    daily = pd.read_parquet(ROOT / "data" / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    d = daily[["date", "code", "close", "volume"]].rename(
        columns={"close": "d_close", "volume": "d_vol"})
    df = df.merge(d, on=["date", "code"], how="left")
    df["recon_close_diff"] = (df["tick_close"] - df["d_close"]).abs() \
        / df["d_close"]
    df["recon_vol_diff"] = (df["tick_vol"] - df["d_vol"]).abs() / df["d_vol"]
    df.to_parquet(ROOT / "data" / "intraday_v1.parquet", index=False)
    ok_c = (df["recon_close_diff"] < 0.001).mean()
    ok_v = (df["recon_vol_diff"] < 0.01).mean()
    print(f"rows={len(df)} codes={df['code'].nunique()} "
          f"recon close<0.1%: {ok_c:.3%} vol<1%: {ok_v:.3%}")
    bad = df[df["recon_vol_diff"] >= 0.05]
    print(f"vol diff >=5%: {len(bad)} rows")
    if len(bad):
        print(bad[["date", "code", "tick_vol", "d_vol", "recon_vol_diff"]]
              .head(20).to_string())


def main() -> None:
    codes = sorted(p.name for p in TICKS.iterdir() if p.is_dir())
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        finalize()
        return
    if len(sys.argv) > 1:
        codes = sys.argv[1].split(",")
    t0 = _time.time()
    with Pool(2) as pool:
        for msg in pool.imap_unordered(process_code, codes):
            print(msg, f"[{_time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
