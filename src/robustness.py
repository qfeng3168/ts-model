"""Robustness pack for a preds file (date, code, fwd_ret_5, pred).

- 5 rebalance phases (offsets 0-4): ann/sharpe/mdd spread.
- Turnover-weighted costs: cost_t = sell_frac*(comm+stamp+slip)
  + buy_frac*(comm+slip); tiers: realistic (slip 0.05%), and flat
  per-unit-turnover pressure tiers 0.25% / 0.35%. Legacy flat 0.15%
  shown for reference.
- Benchmark: equal-weight of the same day's universe (same label
  window); excess series reported.
- ICIR (daily ICs), decile monotonicity of realized label.
- Limit-up diagnostic: share of top20% picks that close limit-up
  at t+1 (unbuyable at that close) and their label contribution;
  share with no t+1 bar (suspended/delisted).
Usage: python src/robustness.py reports/preds_L3_exec.parquet tag
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMM = 0.00025
STAMP = 0.0005


def picks_by_date(d: pd.DataFrame, offset: int = 0):
    dates = sorted(d["date"].unique())
    rebal = dates[offset::5]
    out = []
    for dt in rebal:
        g = d[d["date"] == dt]
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * 0.2))
        out.append((dt, g.nlargest(k, "pred"), g))
    return out


def phase_stats(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for off in range(5):
        rets = [top["fwd_ret_5"].mean() - 0.0015
                for _, top, _ in picks_by_date(d, off)]
        st = ann_stats(pd.Series(rets))
        rows.append({"phase": off, "ann%": round(st["ann_return"] * 100, 1),
                     "sharpe": round(st["sharpe"], 2),
                     "mdd%": round(st["max_drawdown"] * 100, 1)})
    return pd.DataFrame(rows)


def costed_series(d: pd.DataFrame, mode: str) -> pd.Series:
    rets, prev = [], None
    for dt, top, _ in picks_by_date(d, 0):
        cur = set(top["code"])
        gross = top["fwd_ret_5"].mean()
        if prev is None:
            turn = 1.0
        else:
            turn = 1.0 - len(prev & cur) / max(len(cur), 1)
        if mode == "legacy":
            cost = 0.0015
        elif mode == "realistic":
            cost = turn * (COMM + STAMP + 0.0005) \
                + turn * (COMM + 0.0005)
        elif mode == "p25":
            cost = turn * 0.0025
        elif mode == "p35":
            cost = turn * 0.0035
        rets.append(gross - cost)
        prev = cur
    return pd.Series(rets)


def benchmark_excess(d: pd.DataFrame) -> dict:
    pr, br = [], []
    for _, top, g in picks_by_date(d, 0):
        pr.append(top["fwd_ret_5"].mean() - 0.0015)
        br.append(g["fwd_ret_5"].mean())
    pr, br = pd.Series(pr), pd.Series(br)
    ex = pr - br
    return {"port_ann%": round(ann_stats(pr)["ann_return"] * 100, 1),
            "bench_ann%": round(ann_stats(br)["ann_return"] * 100, 1),
            "excess_ann%": round(ann_stats(ex)["ann_return"] * 100, 1),
            "excess_sharpe": round(ann_stats(ex)["sharpe"], 2)}


def ic_pack(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["year"] = d["date"].dt.year
    out = {}
    daily = []
    for dt, g in d.groupby("date"):
        if len(g) >= 20:
            ic, _ = spearmanr(g["pred"], g["fwd_ret_5"])
            if np.isfinite(ic):
                daily.append((dt, ic))
    dd = pd.DataFrame(daily, columns=["date", "ic"])
    out["icir_all"] = round(dd["ic"].mean() / dd["ic"].std()
                            * np.sqrt(243), 2)
    out["ic_pos_share"] = round(float((dd["ic"] > 0).mean()), 3)
    dec = d.assign(dec=pd.qcut(d.groupby("date")["pred"].rank(pct=True),
                               10, labels=False))
    dm = dec.groupby("dec")["fwd_ret_5"].mean()
    out["decile_top_minus_bottom%"] = round(float(dm.iloc[-1] - dm.iloc[0])
                                            * 100, 2)
    out["decile_monotonicity"] = round(
        float(spearmanr(range(10), dm.values)[0]), 2)
    return out


def limitup_diag(d: pd.DataFrame) -> dict:
    daily = pd.read_parquet(ROOT / "data/daily_v2.parquet",
                            columns=["date", "code", "close"])
    daily = daily.sort_values(["code", "date"])
    daily["c1"] = daily.groupby("code")["close"].shift(-1)
    daily["r1x"] = daily["c1"] / daily["close"] - 1
    lk = daily.set_index(["date", "code"])["r1x"]
    n_all, n_lu, n_nobar = 0, 0, 0
    sum_all, sum_lu = 0.0, 0.0
    for _, top, _ in picks_by_date(d, 0):
        for _, r in top.iterrows():
            v = lk.get((r["date"], r["code"]), np.nan)
            n_all += 1
            sum_all += r["fwd_ret_5"]
            if not np.isfinite(v):
                n_nobar += 1
            else:
                thr = 0.198 if r["code"].startswith(("300", "301", "688")) \
                    else 0.098
                if v >= thr:
                    n_lu += 1
                    sum_lu += r["fwd_ret_5"]
    return {"picks": n_all,
            "limitup_t1_share%": round(100 * n_lu / n_all, 2),
            "limitup_label_share%": round(
                100 * sum_lu / sum_all, 1) if sum_all != 0 else None,
            "no_bar_t1_share%": round(100 * n_nobar / n_all, 2)}


def main() -> None:
    path, tag = sys.argv[1], sys.argv[2]
    d = pd.read_parquet(ROOT / path)
    print(f"== robustness {tag}: {len(d):,} preds")
    print(phase_stats(d).to_string(index=False))
    for mode in ["legacy", "realistic", "p25", "p35"]:
        st = ann_stats(costed_series(d, mode))
        print(f"cost[{mode}]: ann={st['ann_return']*100:.1f}% "
              f"sharpe={st['sharpe']:.2f} mdd={st['max_drawdown']*100:.1f}%")
    print("benchmark:", benchmark_excess(d))
    print("ic:", ic_pack(d))
    print("limitup:", limitup_diag(d))


if __name__ == "__main__":
    main()
