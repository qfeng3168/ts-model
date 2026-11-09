"""Caliber fixes v2, from dataset_union. Two artifacts:

dataset_pit2a.parquet — F10 family (13 F10 cols + sue + ind_share)
  takes effect one trading day later: values shifted by 1 row within
  each stock (announcements are mostly after close; same-day use was
  a leak). days_since_notice is aged by the shift's calendar delta so
  it stays consistent with the shifted values. Label unchanged.
dataset_pit2.parquet — pit2a + execution label: fwd_ret_5 column
  holds close(t+1) -> close(t+6) (signal at close t trades at t+1).
  Rows without a full exec window are dropped.

adata notice dates carry no intraday timestamp, so pre/post-market
announcements cannot be distinguished; the uniform +1d is the
conservative choice (disclosed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_v3 import F10  # noqa: E402
from backtest_v6 import SURP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHIFT_COLS = F10 + SURP


def main() -> None:
    ds = pd.read_parquet(ROOT / "data/dataset_union.parquet")
    ds = ds.sort_values(["code", "date"]).reset_index(drop=True)
    delta = ds.groupby("code")["date"].diff().dt.days
    ds[SHIFT_COLS] = ds.groupby("code")[SHIFT_COLS].shift(1)
    ds["days_since_notice"] = ds["days_since_notice"] + delta
    ds.to_parquet(ROOT / "data/dataset_pit2a.parquet", index=False)
    print(f"dataset_pit2a: {len(ds):,} rows", flush=True)

    daily = pd.read_parquet(ROOT / "data/daily_v2.parquet",
                            columns=["date", "code", "close"])
    daily = daily.sort_values(["code", "date"])
    daily["fwd5x"] = (daily.groupby("code")["close"].shift(-6)
                      / daily.groupby("code")["close"].shift(-1) - 1)
    out = ds.merge(daily[["date", "code", "fwd5x"]],
                   on=["date", "code"], how="left")
    before = len(out)
    out = out.dropna(subset=["fwd5x"]).reset_index(drop=True)
    out["fwd_ret_5"] = out["fwd5x"]
    out = out.drop(columns=["fwd5x"])
    out.to_parquet(ROOT / "data/dataset_pit2.parquet", index=False)
    print(f"dataset_pit2: {len(out):,} rows (dropped "
          f"{before - len(out):,} without exec window)")


if __name__ == "__main__":
    main()
