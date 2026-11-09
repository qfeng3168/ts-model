"""P5 ablations on dataset_v5, base D = P1 + F10 + sector momentum.

D2 = D + industry-conditioned commodity exposures
D3 = D + macro2 regime pack (rates/margin/northbound/index/breadth/monthly)
D4 = D + earnings surprise proxy (SUE) + industry revenue share
D5 = D + commodity exposures + surprise
Same walk-forward protocol as P3/P4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
from backtest_v3 import F10  # noqa: E402
from backtest_v4 import run  # noqa: E402
from build_v4 import SECTOR  # noqa: E402
from features import FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CMDTY = ["exp_cu_r5", "exp_cu_r20", "exp_au_r5", "exp_au_r20",
         "exp_oil_r5", "exp_oil_r20", "exp_fx_r20"]
REGIME = ["rate_shibor1w", "rate_shibor_chg20", "rate_cn10y",
          "rate_cn10y_chg20", "rate_spread_10y2y", "rate_lpr1y",
          "mgn_bal_chg20", "nb_flow5", "nb_flow20", "crude_r5",
          "crude_r20", "idx_ret20", "idx_vol20", "idx_dd60",
          "mkt_breadth5", "mkt_limitup",
          "mon_pmi", "mon_cpi_yoy", "mon_ppi_yoy", "mon_sf_yoy"]
SURP = ["sue", "ind_share"]
D = FEATURES + F10 + SECTOR
VARIANTS = {"D_base": D, "D2_cmdty": D + CMDTY, "D3_regime": D + REGIME,
            "D4_surp": D + SURP, "D5_both": D + CMDTY + SURP}


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_v5.parquet")
    df["year"] = df["date"].dt.year
    res = pd.concat([run(df, f, n) for n, f in VARIANTS.items()],
                    ignore_index=True)
    wide = res[res["year"] != "ALL"].pivot(index="year", columns="model",
                                           values="rank_ic")
    wide = wide.reindex(columns=list(VARIANTS))
    wide.loc["mean"] = wide.mean()
    allrows = res[res["year"] == "ALL"].set_index("model") \
        .reindex(list(VARIANTS))
    lines = ["# P5 消融：D 为新基底（板块动量胜出后）", "",
             "基底 D = P1 量价 + F10 + 板块动量（P4 胜出）。", "",
             "## 逐年 RankIC", "", wide.round(4).to_markdown(), "",
             "## 组合（全期拼接）", "",
             allrows[["total_return", "ann_return", "max_drawdown",
                      "sharpe"]].round(4).to_markdown(), ""]
    (ROOT / "reports/p5_ablation.md").write_text("\n".join(lines),
                                                 encoding="utf-8")
    print(wide.round(4).to_string())
    print(allrows[["ann_return", "max_drawdown", "sharpe"]].round(4)
          .to_string())


if __name__ == "__main__":
    main()
