"""Point-in-time HS300 membership from baostock snapshots.

baostock's query_hs300_stocks(date) returns the constituent list in
effect on that date (with updateDate). Monthly snapshots 2014-06 ->
2026-11 give membership as a step function; HS300 only adjusts in
June/December, so monthly resolution captures every change with at
most a few weeks of edge error (disclosed in the report).

Outputs:
  data/hs300_membership.parquet  (code, snap_date) raw snapshots
  data/hs300_union.parquet       (code, first_seen, last_seen, n_snaps)
"""
from __future__ import annotations

from pathlib import Path

import baostock as bs
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    lg = bs.login()
    print("login", lg.error_code, lg.error_msg, flush=True)
    snaps = []
    dates = pd.date_range("2014-06-30", "2026-11-30", freq="ME")
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        rs = bs.query_hs300_stocks(date=ds)
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            print(ds, "EMPTY", flush=True)
            continue
        df = pd.DataFrame(rows, columns=rs.fields)
        df["snap_date"] = d
        snaps.append(df)
        print(ds, len(df), "update", df["updateDate"].iloc[0], flush=True)
    bs.logout()
    all_snaps = pd.concat(snaps, ignore_index=True)
    all_snaps["code"] = all_snaps["code"].str.split(".").str[1]
    all_snaps.to_parquet(ROOT / "data/hs300_membership.parquet",
                         index=False)
    union = all_snaps.groupby("code").agg(
        first_seen=("snap_date", "min"), last_seen=("snap_date", "max"),
        n_snaps=("snap_date", "count")).reset_index()
    union.to_parquet(ROOT / "data/hs300_union.parquet", index=False)
    current = set(pd.read_parquet(ROOT / "data/daily.parquet",
                                  columns=["code"])["code"].unique())
    extra = sorted(set(union["code"]) - current)
    print(f"union={len(union)} current300={len(current)} "
          f"extra={len(extra)}")
    print("extra codes:", ",".join(extra))


if __name__ == "__main__":
    main()
