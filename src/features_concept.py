"""Concept-board features (E version) from EM memberships (snapshot).

Memberships are the 2026-11 snapshot (462 of 531 boards fetched; the
rest were eastmoney-rate-limited and are being refetched). Historical
application of a current snapshot is an approximation - disclosed in
reports. Features per stock-day, all causal:
  cpt_ret5 / cpt_ret20: max over the stock's concepts of the peer
    (ex-self, HS300 members only) mean return.
  cpt_breadth5: max peer share with ret_5 > 0.
  cpt_news5: max over concepts of the trailing-5d XWLB mention count of
    the concept name, strictly lagged (broadcast day < t) - the v2
    attempt at policy/news linkage after the v1 industry-keyword
    version failed in P3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONCEPT = ["cpt_ret5", "cpt_ret20", "cpt_breadth5", "cpt_news5"]


def _news_counts(concepts: list[str]) -> pd.DataFrame:
    days = []
    with open(ROOT / "data/xwlb/xwlb.jsonl", encoding="utf-8") as fh:
        for line in fh:
            days.append(json.loads(line))
    rows = {}
    for d in days:
        text = d.get("text", "")
        rows[d["date"]] = {c: text.count(c) for c in concepts
                           if c in text}
    cnt = pd.DataFrame(0, index=sorted(rows), columns=concepts,
                       dtype=np.int32)
    for dt, hits in rows.items():
        for c, n in hits.items():
            cnt.loc[dt, c] = n
    cnt.index = pd.to_datetime(cnt.index)
    return cnt


def build_concept(ds: pd.DataFrame) -> pd.DataFrame:
    mem = pd.read_parquet(ROOT / "data/f10/concepts_em.parquet")
    stock_concepts: dict[str, list[str]] = {}
    for cpt, grp in mem.groupby("concept"):
        members = set(grp["code"])
        for code in members:
            stock_concepts.setdefault(code, []).append(cpt)
    base = ds[["date", "code", "ret_5", "ret_20"]].copy()
    pivot5 = base.pivot_table(index="date", columns="code",
                              values="ret_5")
    pivot20 = base.pivot_table(index="date", columns="code",
                               values="ret_20")
    concepts = sorted(mem["concept"].unique())
    news = _news_counts(concepts)
    news5 = news.rolling(5).sum().shift(1)
    dates = pivot5.index
    npos = {d: i for i, d in enumerate(dates)}
    peer5 = pd.DataFrame(np.nan, index=dates, columns=concepts)
    peer20 = pd.DataFrame(np.nan, index=dates, columns=concepts)
    peerb = pd.DataFrame(np.nan, index=dates, columns=concepts)
    for cpt in concepts:
        members = [c for c in mem.loc[mem["concept"] == cpt, "code"]
                   if c in pivot5.columns]
        if len(members) < 3:
            continue
        sub5 = pivot5[members]
        sub20 = pivot20[members]
        peer5[cpt] = sub5.mean(axis=1)
        peer20[cpt] = sub20.mean(axis=1)
        peerb[cpt] = (sub5 > 0).mean(axis=1)
    out = base[["date", "code"]].copy()
    di = out["date"].map(npos).to_numpy()
    for name, table in (("cpt_ret5", peer5), ("cpt_ret20", peer20),
                        ("cpt_breadth5", peerb)):
        vals = np.full(len(out), np.nan)
        arr = table.to_numpy()
        cidx = {c: i for i, c in enumerate(table.columns)}
        for r, (code, _) in enumerate(zip(out["code"], out["date"])):
            cps = stock_concepts.get(code)
            if not cps:
                continue
            js = [cidx[c] for c in cps if c in cidx]
            if js:
                v = arr[di[r], js]
                if np.isfinite(v).any():
                    vals[r] = np.nanmax(v)
        out[name] = vals
    narr = news5.to_numpy()
    ncidx = {c: i for i, c in enumerate(news5.columns)}
    uniq_dates = np.sort(out["date"].unique())
    upos = np.searchsorted(news5.index.to_numpy(), uniq_dates,
                           side="left") - 1
    row_of_date = {d: (upos[i] if upos[i] >= 0 else -1)
                   for i, d in enumerate(uniq_dates)}
    vals = np.full(len(out), np.nan)
    for r, (code, dt) in enumerate(zip(out["code"], out["date"])):
        cps = stock_concepts.get(code)
        i = row_of_date[dt]
        if not cps or i < 0:
            continue
        js = [ncidx[c] for c in cps if c in ncidx]
        if js:
            vals[r] = narr[i, js].max()
    out["cpt_news5"] = vals
    return out
