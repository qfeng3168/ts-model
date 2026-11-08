"""P3 feature build: P1 features + F10 fundamentals + XWLB policy signals.

F10 alignment is strictly point-in-time: for a trading date t, the latest
core_index report with notice_date <= t (announcements are public that
day). notice_date sentinel rows (1900) are dropped. Dividend yield uses
announced cash dividends (per-share) summed over the trailing 365 days.

XWLB alignment: the broadcast airs at 19:00, after the close, so features
for date t come from broadcasts with date < t (strict). Each stock is
mapped to a policy keyword group through its CSRC industry; the feature
is that group's mention intensity (hits per 1000 chars) plus 5/20
broadcast-day rolling means.

Output: data/dataset_v3.parquet (dataset.parquet + new columns).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

GROUPS: dict[str, list[str]] = {
    "tech": ["半导体", "芯片", "集成电路", "人工智能", "数字经济", "科技创新",
             "大数据", "云计算", "信息技术"],
    "military": ["国防", "强军", "军工", "武器装备", "军队建设"],
    "property": ["房地产", "保障性住房", "住房"],
    "finance": ["资本市场", "股票市场", "降准", "降息", "货币政策",
                "金融监管", "信贷", "证券市场"],
    "energy": ["光伏", "风电", "新能源", "核电", "储能", "电网", "煤炭",
               "石油", "天然气"],
    "agri": ["农业", "粮食", "乡村振兴", "种业", "春耕", "农民增收"],
    "pharma": ["医疗", "医药", "中医药", "医保", "疫苗", "公共卫生"],
    "consumption": ["消费", "旅游", "家电", "零售"],
    "infra": ["基础设施", "重大工程", "铁路建设", "水利工程", "交通建设"],
    "eco": ["生态文明", "碳达峰", "碳中和", "污染防治", "绿色发展", "环保"],
    "manufacturing": ["制造业", "智能制造", "装备制造", "工业经济",
                      "专精特新"],
}

_PREFIX_GROUP = {
    "C39": "tech", "C40": "tech", "C41": "tech",
    "I63": "tech", "I64": "tech", "I65": "tech",
    "J66": "finance", "J67": "finance", "J68": "finance", "J69": "finance",
    "D44": "energy", "D45": "energy", "B06": "energy", "B07": "energy",
    "C38": "energy",
    "K70": "property",
    "C27": "pharma",
    "A01": "agri", "A02": "agri", "A03": "agri", "A04": "agri",
    "A05": "agri", "C13": "agri",
    "C14": "consumption", "C15": "consumption", "C36": "consumption",
    "F51": "consumption", "F52": "consumption", "H61": "consumption",
    "H62": "consumption", "L72": "consumption",
    "E48": "infra", "E49": "infra", "E50": "infra", "G53": "infra",
    "G54": "infra", "G55": "infra", "G56": "infra", "G58": "infra",
    "G59": "infra", "G60": "infra", "C30": "infra",
    "C37": "military",
    "N76": "eco", "N77": "eco", "N78": "eco", "C42": "eco",
}


def industry_group(code3: str) -> str:
    return _PREFIX_GROUP.get(code3, "manufacturing")


def build_f10(ds: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    ci = pd.read_parquet(ROOT / "data/f10/core_index.parquet")
    ci = ci[ci["notice_date"] > "1990-01-01"].copy()
    ci = ci.sort_values(["stock_code", "notice_date"])
    cols = ["basic_eps", "net_asset_ps", "roe_wtd", "roa_wtd", "gross_margin",
            "net_margin", "total_rev_yoy_gr", "net_profit_yoy_gr",
            "net_profit_qoq_gr", "asset_liab_ratio", "oper_cf_ps"]
    px = daily.set_index(["code", "date"])["close"]
    out = []
    for code, g in ds.groupby("code"):
        reps = ci[ci["stock_code"] == code]
        dates = np.sort(g["date"].unique())
        if len(reps) == 0:
            continue
        nd = reps["notice_date"].to_numpy()
        pos = np.searchsorted(nd, dates, side="right") - 1
        rec = {"date": dates, "code": code}
        for c in cols:
            vals = reps[c].to_numpy(dtype=float)
            rec[c] = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
        rec["days_since_notice"] = np.where(
            pos >= 0, (dates - nd[np.clip(pos, 0, None)]).astype(
                "timedelta64[D]").astype(float), np.nan)
        close = np.array([px.get((code, d), np.nan) for d in dates])
        rec["ep_ttm"] = np.clip(rec.pop("basic_eps") / close, -1, 2)
        rec["bp"] = np.clip(rec.pop("net_asset_ps") / close, -1, 3)
        rec["cfp"] = np.clip(rec.pop("oper_cf_ps") / close, -2, 2)
        for c in ["total_rev_yoy_gr", "net_profit_yoy_gr", "net_profit_qoq_gr"]:
            rec[c] = np.clip(rec[c], -300, 300)
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def build_div(ds: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    dv = pd.read_parquet(ROOT / "data/f10/dividends.parquet")
    dv["ann"] = pd.to_datetime(dv["公告日期"])
    dv["cash_ps"] = pd.to_numeric(dv["派息"], errors="coerce") / 10.0
    dv = dv[(dv["cash_ps"] > 0) & dv["ann"].notna()]
    px = daily.set_index(["code", "date"])["close"]
    out = []
    for code, g in ds.groupby("code"):
        rows = dv[dv["code"] == code].sort_values("ann")
        dates = np.sort(g["date"].unique())
        if len(rows) == 0:
            out.append(pd.DataFrame({"date": dates, "code": code,
                                     "div_yield_ttm": np.nan}))
            continue
        ann = rows["ann"].to_numpy()
        cash = rows["cash_ps"].to_numpy()
        cum = np.concatenate([[0.0], np.cumsum(cash)])
        hi = np.searchsorted(ann, dates, side="right")
        lo = np.searchsorted(ann, dates - np.timedelta64(365, "D"),
                             side="right")
        ttm = cum[hi] - cum[lo]
        close = np.array([px.get((code, d), np.nan) for d in dates])
        out.append(pd.DataFrame({"date": dates, "code": code,
                                 "div_yield_ttm": ttm / close}))
    return pd.concat(out, ignore_index=True)


def build_xwlb(ds: pd.DataFrame) -> pd.DataFrame:
    recs = []
    with open(ROOT / "data/xwlb/xwlb.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            recs.append(r)
    days = pd.DataFrame(recs)
    days["date"] = pd.to_datetime(days["date"])
    days = days.sort_values("date").reset_index(drop=True)
    for gname, kws in GROUPS.items():
        pat = "|".join(map(re.escape, kws))
        hits = days["text"].str.count(pat)
        days[f"int_{gname}"] = hits * 1000.0 / days["text"].str.len()
        days[f"int5_{gname}"] = days[f"int_{gname}"].rolling(5).mean()
        days[f"int20_{gname}"] = days[f"int_{gname}"].rolling(20).mean()
    days["n_items"] = days["items"].apply(len)

    ind = pd.read_parquet(ROOT / "data/f10/industry.parquet")
    grp = {c: industry_group(i[:3]) for c, i in
           zip(ind["code"], ind["industry"])}
    bdates = days["date"].to_numpy()
    out = []
    for code, g in ds.groupby("code"):
        dates = np.sort(g["date"].unique())
        # strict lag: broadcast day must be < trading date
        pos = np.searchsorted(bdates, dates, side="left") - 1
        gname = grp.get(code, "manufacturing")
        rec = {"date": dates, "code": code,
               "xwlb_grp": np.where(pos >= 0, days[f"int_{gname}"].to_numpy()[
                   np.clip(pos, 0, None)], np.nan),
               "xwlb_grp5": np.where(pos >= 0, days[f"int5_{gname}"].to_numpy()[
                   np.clip(pos, 0, None)], np.nan),
               "xwlb_grp20": np.where(pos >= 0, days[f"int20_{gname}"].to_numpy()[
                   np.clip(pos, 0, None)], np.nan),
               "xwlb_fin": np.where(pos >= 0, days["int_finance"].to_numpy()[
                   np.clip(pos, 0, None)], np.nan),
               "xwlb_items": np.where(pos >= 0, days["n_items"].to_numpy()[
                   np.clip(pos, 0, None)], np.nan)}
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ds = pd.read_parquet(ROOT / "data/dataset.parquet")
    daily = pd.read_parquet(ROOT / "data/daily.parquet")
    f10 = build_f10(ds, daily)
    div = build_div(ds, daily)
    xw = build_xwlb(ds)
    out = ds.merge(f10, on=["date", "code"], how="left") \
            .merge(div, on=["date", "code"], how="left") \
            .merge(xw, on=["date", "code"], how="left")
    out.to_parquet(ROOT / "data/dataset_v3.parquet", index=False)
    new = [c for c in out.columns if c not in ds.columns]
    print(f"dataset_v3: {len(out):,} rows, +{len(new)} features: {new}")
    print("NaN share of new features:")
    print(out[new].isna().mean().round(4).to_string())


if __name__ == "__main__":
    main()
