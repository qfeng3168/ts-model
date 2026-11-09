"""Build dataset_v5 = dataset_v4 + macro2 daily + monthly + surprise +
industry-conditioned commodity exposures."""
from pathlib import Path

import pandas as pd

from features_cmdty import build_cmdty
from features_macro2 import build_macro2, build_monthly, build_surprise

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_parquet(ROOT / "data/dataset_v4.parquet")
for builder in (build_macro2, build_monthly, build_surprise, build_cmdty):
    add = builder(df)
    new = [c for c in add.columns if c not in ("date", "code")]
    df = df.merge(add, on=["date", "code"], how="left", validate="1:1")
    stats = {c: round(float(df[c].isna().mean()), 3) for c in new}
    print(builder.__name__, "->", stats, flush=True)
df.to_parquet(ROOT / "data/dataset_v5.parquet", index=False)
print("saved dataset_v5", df.shape)
