"""dataset_pit3 = pit2a (F10 delayed) with open-execution label:
fwd_ret_5 := close(t+6) / open(t+1) - 1. Signal at close t, buy at
the t+1 opening auction (realistic for a close-computed signal),
sell at close t+6. Sits between the optimistic close(t) caliber
and the conservative close(t+1) caliber of pit2.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ds = pd.read_parquet(ROOT / "data/dataset_pit2a.parquet")
    daily = pd.read_parquet(ROOT / "data/daily_v2.parquet",
                            columns=["date", "code", "open", "close"])
    daily = daily.sort_values(["code", "date"])
    daily["fwd5o"] = (daily.groupby("code")["close"].shift(-6)
                      / daily.groupby("code")["open"].shift(-1) - 1)
    out = ds.merge(daily[["date", "code", "fwd5o"]],
                   on=["date", "code"], how="left")
    before = len(out)
    out = out.dropna(subset=["fwd5o"]).reset_index(drop=True)
    out["fwd_ret_5"] = out["fwd5o"]
    out = out.drop(columns=["fwd5o"])
    out.to_parquet(ROOT / "data/dataset_pit3.parquet", index=False)
    print(f"dataset_pit3: {len(out):,} rows (dropped "
          f"{before - len(out):,})")


if __name__ == "__main__":
    main()
