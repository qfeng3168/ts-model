"""Local forward adjustment (前复权) from raw bars + xdxr records.

For each ex-date event (category 1) with per-10-share fields from TDX:
  cash   = fenhong / 10                      (cash dividend per share)
  bonus  = songzhuangu / 10                  (bonus+transfer shares per share)
  rights = peigu / 10, rights_price = peigujia
  ex_ref = (prev_close - cash + rights * rights_price) / (1 + bonus + rights)
  ratio  = ex_ref / prev_close
The forward-adjust factor for a bar on date d is the product of ratios of
all events with ex_date > d, so the latest prices stay raw (same anchoring
as vendor qfq series). One factor table per stock, valid for daily AND
minute bars (factor is constant within a day).

Usage: python src/adjust.py [--check 600519,000001]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build_factors(code: str, raw: pd.DataFrame, xdxr: pd.DataFrame) -> pd.DataFrame:
    """raw: this stock's raw daily bars (date, close). Returns date -> factor."""
    ev = xdxr[(xdxr["code"] == code) & (xdxr["category"] == 1)].copy()
    dates = raw["date"].reset_index(drop=True)
    factor = np.ones(len(raw))
    if len(ev):
        ev["ex_date"] = pd.to_datetime(
            ev[["year", "month", "day"]].rename(
                columns={"year": "year", "month": "month", "day": "day"}))
        close = raw.set_index("date")["close"]
        for _, e in ev.sort_values("ex_date").iterrows():
            prev = close[close.index < e["ex_date"]]
            if len(prev) == 0:
                continue
            pc = float(prev.iloc[-1])
            cash = float(e["fenhong"] or 0) / 10
            bonus = float(e["songzhuangu"] or 0) / 10
            rights = float(e["peigu"] or 0) / 10
            rp = float(e["peigujia"] or 0)
            ex_ref = (pc - cash + rights * rp) / (1 + bonus + rights)
            ratio = ex_ref / pc
            factor[dates < e["ex_date"]] *= ratio
    return pd.DataFrame({"date": dates, "factor": factor, "code": code})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="600519,000001,300750")
    args = ap.parse_args()

    raw = pd.read_parquet(ROOT / "data" / "daily_raw.parquet")
    xfiles = sorted((ROOT / "data" / "xdxr").glob("*.parquet"))
    xdxr = pd.concat([pd.read_parquet(f) for f in xfiles], ignore_index=True)
    print(f"raw: {len(raw):,} rows; xdxr events: {len(xdxr):,} "
          f"(category1: {(xdxr['category'] == 1).sum():,})", flush=True)

    tables = []
    for code, g in raw.groupby("code"):
        tables.append(build_factors(code, g.sort_values("date"), xdxr))
    factors = pd.concat(tables, ignore_index=True)
    factors.to_parquet(ROOT / "data" / "qfq_factors.parquet", index=False)
    print(f"factors: {len(factors):,} rows -> data/qfq_factors.parquet", flush=True)

    # validation: locally adjusted close vs the Sina qfq daily already stored,
    # both normalized by their last value (same anchoring convention).
    daily = pd.read_parquet(ROOT / "data" / "daily.parquet")
    for code in args.check.split(","):
        r = raw[raw["code"] == code].sort_values("date")
        f = factors[factors["code"] == code].sort_values("date")
        mine = r.merge(f, on="date")
        mine["adj"] = mine["close"] * mine["factor"]
        ref = daily[daily["code"] == code][["date", "close"]].rename(
            columns={"close": "ref"})
        m = mine.merge(ref, on="date")
        a = m["adj"] / m["adj"].iloc[-1]
        b = m["ref"] / m["ref"].iloc[-1]
        dev = (a - b).abs()
        print(f"[check {code}] days={len(m)} max_dev={dev.max() * 100:.3f}% "
              f"mean_dev={dev.mean() * 100:.4f}%", flush=True)


if __name__ == "__main__":
    main()
