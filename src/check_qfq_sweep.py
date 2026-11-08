"""Sweep: Sina qfq close (data/daily.parquet) vs baostock qfq close.

Both series normalized by their last value; report max deviation per stock.
Motivation: 000001's Sina qfq proved corrupted (local factors agree with
baostock to 0.1% while Sina qfq deviates up to 22%). Output:
reports/qfq_sweep.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import baostock as bs
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    daily = pd.read_parquet(ROOT / "data" / "daily.parquet",
                            columns=["date", "code", "close"])
    codes = sorted(daily["code"].unique())
    bs.login()
    out, fail = [], []
    for i, code in enumerate(codes, 1):
        mkt = "sh" if code.startswith("6") else "sz"
        try:
            q = bs.query_history_k_data_plus(
                f"{mkt}.{code}", "date,close", start_date="2014-06-01",
                end_date="2026-11-06", frequency="d", adjustflag="2")
            rows = []
            while q.error_code == "0" and q.next():
                rows.append(q.get_row_data())
            b = pd.DataFrame(rows, columns=["date", "close"])
            b["date"] = pd.to_datetime(b["date"])
            b["close"] = b["close"].astype(float)
            s = daily[daily["code"] == code][["date", "close"]].rename(
                columns={"close": "sina"})
            m = s.merge(b.rename(columns={"close": "bs"}), on="date")
            dev = (m["sina"] / m["sina"].iloc[-1]
                   - m["bs"] / m["bs"].iloc[-1]).abs()
            out.append({"code": code, "days": len(m),
                        "max_dev_pct": dev.max() * 100,
                        "mean_dev_pct": dev.mean() * 100})
        except Exception as exc:  # noqa: BLE001
            fail.append(code)
            print(f"{code} FAIL {exc}", flush=True)
        if i % 50 == 0:
            print(f"[{i}/{len(codes)}] fail={len(fail)}", flush=True)
    bs.logout()
    df = pd.DataFrame(out).sort_values("max_dev_pct", ascending=False)
    df.to_csv(ROOT / "reports" / "qfq_sweep.csv", index=False)
    bad = df[df["max_dev_pct"] > 1.0]
    print(f"DONE sweep: {len(df)} stocks, fail={fail}")
    print(f">1% max dev: {len(bad)} stocks")
    print(bad.head(30).to_string())


if __name__ == "__main__":
    main()
