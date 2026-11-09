"""Assemble dataset_v4 = dataset_v3 + sector + macro + event features."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_extra import build_events, build_macro  # noqa: E402
from features_sector import build_sector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

SECTOR = ["ind_ret5", "ind_ret20", "ind_breadth5", "ind_volr"]
MACRO = ["gold_cny_r5", "gold_cny_r20", "gold_cny_z60", "gold_usd_r5",
         "gold_usd_r20", "usdcny_r5", "usdcny_r20", "copper_cny_r5",
         "copper_cny_r20", "copper_usd_r5", "copper_usd_r20", "mkt_flow5"]
EVENTS = ["dz_disc_mean", "dz_amt_sum", "dz_cnt_sum", "lhb_net_mean",
          "lhb_cnt_sum"]


def main() -> None:
    ds = pd.read_parquet(ROOT / "data/dataset_v3.parquet")
    sec = build_sector(ds)
    mac = build_macro(ds)
    ev = build_events(ds)
    out = ds.merge(sec, on=["date", "code"], how="left") \
            .merge(mac, on=["date", "code"], how="left") \
            .merge(ev, on=["date", "code"], how="left")
    out.to_parquet(ROOT / "data/dataset_v4.parquet", index=False)
    print(f"dataset_v4: {len(out):,} rows")
    print(out[SECTOR + MACRO + EVENTS].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
