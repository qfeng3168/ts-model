"""Build data/daily_v2.parquet: current daily (300) + qfq bars for the
382 union extras (raw x local xdxr factors, the validated method).

Spot-checks 3 extras against Sina's own qfq series (normalized dev),
mirroring adjust.py's validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adjust import build_factors  # noqa: E402
import fetch_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parts = sorted((ROOT / "data/union_raw").glob("*.parquet"))
    xfiles = {p.stem: p for p in (ROOT / "data/xdxr").glob("*.parquet")}
    frames = []
    no_xdxr = []
    for p in parts:
        code = p.stem
        raw = pd.read_parquet(p).sort_values("date")
        if code in xfiles:
            x = pd.read_parquet(xfiles[code])
        else:
            x = pd.DataFrame()
            no_xdxr.append(code)
        f = build_factors(code, raw, x)
        d = raw.merge(f[["date", "factor"]], on="date", how="left")
        d["factor"] = d["factor"].fillna(1.0)
        for c in ["open", "high", "low", "close"]:
            d[c] = d[c] * d["factor"]
        frames.append(d[["date", "code", "open", "close", "high", "low",
                         "volume", "amount", "turnover"]])
    ext = pd.concat(frames, ignore_index=True)
    print(f"extras qfq: {len(ext):,} rows, {ext['code'].nunique()} stocks, "
          f"no_xdxr={len(no_xdxr)}")
    daily = pd.read_parquet(ROOT / "data/daily.parquet")
    v2 = pd.concat([daily, ext], ignore_index=True)
    v2 = v2.sort_values(["code", "date"]).reset_index(drop=True)
    v2.to_parquet(ROOT / "data/daily_v2.parquet", index=False)
    print(f"daily_v2: {len(v2):,} rows, {v2['code'].nunique()} stocks")

    for code in ["600004", "000063", "002024"]:
        if code not in set(ext["code"]):
            continue
        ref = fetch_data.fetch_one(code)
        mine = ext[ext["code"] == code][["date", "close"]]
        m = mine.merge(ref[["date", "close"]].rename(
            columns={"close": "ref"}), on="date")
        if len(m) == 0:
            print(f"[check {code}] no overlap")
            continue
        a = m["close"] / m["close"].iloc[-1]
        b = m["ref"] / m["ref"].iloc[-1]
        dev = (a - b).abs()
        print(f"[check {code}] days={len(m)} max_dev={dev.max()*100:.3f}% "
              f"mean_dev={dev.mean()*100:.4f}%")


if __name__ == "__main__":
    main()
