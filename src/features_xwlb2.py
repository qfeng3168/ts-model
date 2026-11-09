"""XWLB v2: policy-attention SURPRISE x sector, not raw keyword counts.

v1 (P3) failed with raw full-text mention intensity per industry group.
v2 fixes the construction:
  * items are used structurally - headline hits weigh 3x, and earlier
    items in the broadcast weigh more (policy priority order);
  * the signal is a surprise: z-score of today's weighted attention vs
    its trailing-60-broadcast baseline (chronic mentions are noise);
  * a separate lead series uses only the first 3 items (headlines);
  * explicit interactions with sector momentum / breadth - the
    "xwlb x sector" combination: policy attention confirming a sector
    move is a different animal from attention alone;
  * concept names (verbatim policy phrases) get the same treatment.
All series are built on broadcast dates and mapped to trading day t
with broadcast date < t (the 19:00 broadcast is usable from t+1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_v3 import GROUPS, industry_group  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
X2_IND = ["x2_attn_z", "x2_attn_ema", "x2_lead_z", "x2_attn_x_mom",
          "x2_attn_x_brd"]
X2_CPT = ["x2_cattn_z", "x2_cattn_ema", "x2_cattn_x_mom"]
X2 = X2_IND + X2_CPT


def _daily_counts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per broadcast day: group attention, group lead attention,
    concept attention (all position/title weighted)."""
    days = []
    with open(ROOT / "data/xwlb/xwlb.jsonl", encoding="utf-8") as fh:
        for line in fh:
            days.append(json.loads(line))
    mem = pd.read_parquet(ROOT / "data/f10/concepts_em.parquet")
    concepts = sorted(mem["concept"].unique())
    g_rows, lead_rows, c_rows, dates = [], [], [], []
    for d in days:
        items = d.get("items") or []
        g = {g: 0.0 for g in GROUPS}
        lead = {g: 0.0 for g in GROUPS}
        c = {cpt: 0.0 for cpt in concepts}
        for pos, it in enumerate(items):
            w = 1.0 / (1.0 + 0.25 * pos)
            title = it.get("title", "")
            content = it.get("content", "")
            for gname, kws in GROUPS.items():
                hits = sum(3 * title.count(k) + content.count(k)
                           for k in kws)
                if hits:
                    g[gname] += w * hits
                    if pos < 3:
                        lead[gname] += w * hits
            for cpt in concepts:
                h = 3 * title.count(cpt) + content.count(cpt)
                if h:
                    c[cpt] += w * h
        g_rows.append(g)
        lead_rows.append(lead)
        c_rows.append(c)
        dates.append(d["date"])
    idx = pd.to_datetime(dates)
    return (pd.DataFrame(g_rows, index=idx).sort_index(),
            pd.DataFrame(lead_rows, index=idx).sort_index(),
            pd.DataFrame(c_rows, index=idx).sort_index())


def _z(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.rolling(60, min_periods=20).mean()
    sd = df.rolling(60, min_periods=20).std()
    return ((df - mu) / sd.replace(0, np.nan)).clip(-1, 5)


def build_xwlb2(ds: pd.DataFrame) -> pd.DataFrame:
    g, lead, c = _daily_counts()
    gz, lz, cz = _z(g), _z(lead), _z(c)
    gema = gz.ewm(span=5, min_periods=1).mean()
    cema = cz.ewm(span=5, min_periods=1).mean()
    ind = pd.read_parquet(ROOT / "data/f10/industry.parquet")
    grp = {cd: industry_group(i[:3]) for cd, i in
           zip(ind["code"], ind["industry"])}
    mem = pd.read_parquet(ROOT / "data/f10/concepts_em.parquet")
    stock_concepts: dict[str, list[str]] = {}
    for cpt, sub in mem.groupby("concept"):
        for code in sub["code"]:
            stock_concepts.setdefault(code, []).append(cpt)

    def lagged(table: pd.DataFrame, dates: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(table.index.to_numpy(), dates,
                              side="left") - 1
        arr = table.to_numpy()
        out = np.full((len(dates), arr.shape[1]), np.nan)
        ok = pos >= 0
        out[ok] = arr[pos[ok]]
        return out

    out = ds[["date", "code"]].copy()
    dates = out["date"].to_numpy()
    gz_a = lagged(gz, dates)
    ge_a = lagged(gema, dates)
    lz_a = lagged(lz, dates)
    gidx = {gname: i for i, gname in enumerate(gz.columns)}
    grow = np.array([gidx[grp.get(cd, "manufacturing")]
                     for cd in out["code"]])
    rows = np.arange(len(out))
    out["x2_attn_z"] = gz_a[rows, grow]
    out["x2_attn_ema"] = ge_a[rows, grow]
    out["x2_lead_z"] = lz_a[rows, grow]
    # Concepts stay in broadcast-date space: materializing a
    # (rows x concepts) lagged matrix is ~2.7 GB and OOMs this VM.
    cpos = np.searchsorted(cz.index.to_numpy(), dates, side="left") - 1
    cz_a = cz.to_numpy()
    ce_a = cema.to_numpy()
    cidx = {cpt: i for i, cpt in enumerate(cz.columns)}
    czv = np.full(len(out), np.nan)
    cev = np.full(len(out), np.nan)
    for r, code in enumerate(out["code"]):
        cps = stock_concepts.get(code)
        bi = cpos[r]
        if not cps or bi < 0:
            continue
        js = [cidx[x] for x in cps if x in cidx]
        if js:
            v = cz_a[bi, js]
            if np.isfinite(v).any():
                czv[r] = np.nanmax(v)
            v = ce_a[bi, js]
            if np.isfinite(v).any():
                cev[r] = np.nanmax(v)
    out["x2_cattn_z"] = czv
    out["x2_cattn_ema"] = cev
    base = ds[["ind_ret5", "ind_breadth5", "cpt_ret5"]].reset_index(
        drop=True)
    out["x2_attn_x_mom"] = out["x2_attn_ema"] * base["ind_ret5"]
    out["x2_attn_x_brd"] = out["x2_attn_ema"] * base["ind_breadth5"]
    out["x2_cattn_x_mom"] = out["x2_cattn_ema"] * base["cpt_ret5"]
    return out
