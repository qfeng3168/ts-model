"""dataset_v6 = dataset_v5 + TA pack + concept pack."""
from pathlib import Path

import pandas as pd

from features_concept import build_concept
from features_ta import build_ta

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_parquet(ROOT / "data/dataset_v5.parquet")
for builder in (build_ta, build_concept):
    add = builder(df)
    new = [c for c in add.columns if c not in ("date", "code")]
    df = df.merge(add, on=["date", "code"], how="left", validate="1:1")
    print(builder.__name__,
          {c: round(float(df[c].isna().mean()), 3) for c in new},
          flush=True)
df.to_parquet(ROOT / "data/dataset_v6.parquet", index=False)
print("saved dataset_v6", df.shape)
