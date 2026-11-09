"""P7: XWLB v2 (attention surprise x sector) ablations on the D6 base."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_v6 import D5, run  # noqa: E402
from features_ta import TA  # noqa: E402
from features_xwlb2 import X2_CPT, X2_IND  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D6 = D5 + TA
VARIANTS = {"D6": D6, "D6X_ind": D6 + X2_IND, "D6X_cpt": D6 + X2_CPT,
            "D6X_both": D6 + X2_IND + X2_CPT}


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_v7.parquet")
    df["year"] = df["date"].dt.year
    res = pd.concat([run(df, f, n) for n, f in VARIANTS.items()],
                    ignore_index=True)
    wide = res[res["year"] != "ALL"].pivot(index="year", columns="model",
                                           values="rank_ic")
    wide = wide.reindex(columns=list(VARIANTS))
    wide.loc["mean"] = wide.mean()
    allrows = res[res["year"] == "ALL"].set_index("model") \
        .reindex(list(VARIANTS))
    lines = ["# P7 消融：新闻联播 v2（注意力意外 × 板块）", "",
             "基底 D6。v2 = 标题/顺位加权计数 → 60 日基线 z-score 意外度 "
             "+ 头条子序列 + 与板块动量/宽度的显式交互。", "",
             "## 逐年 RankIC", "", wide.round(4).to_markdown(), "",
             "## 组合（全期拼接）", "",
             allrows[["total_return", "ann_return", "max_drawdown",
                      "sharpe"]].round(4).to_markdown(), ""]
    (ROOT / "reports/p7_xwlb2.md").write_text("\n".join(lines),
                                              encoding="utf-8")
    print(wide.round(4).to_string())
    print(allrows[["ann_return", "max_drawdown", "sharpe"]].round(4)
          .to_string())


if __name__ == "__main__":
    main()
