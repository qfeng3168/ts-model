#!/usr/bin/env python3
"""扩展数据拉取（东方财富/交易所源，经 AKShare，G98 国内网络）。

家族：个股资金流、大宗交易、融资融券明细、北向持股、股东户数、机构调研、
限售解禁、股权质押、市场资金流。范围：沪深300 股票池、2020 起。
每个家族独立：checkpoint(done 文件)+周期落盘，单家失败不阻塞其他家。
输出 /vol2/extra/<family>.parquet，失败键记 <family>.failed.txt。
"""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import akshare as ak

OUT = "/vol2/extra"
os.makedirs(OUT, exist_ok=True)
CODES = open("/vol2/ts-ticks/tdx_codes.txt").read().split()
CODESET = set(CODES)
DATES = open("/vol2/ts-ticks/tdx_dates_2020.txt").read().split()  # YYYYMMDD
lock = threading.Lock()


def code_col(df):
    for c in df.columns:
        if "代码" in str(c):
            return c
    return None


def norm_codes(s):
    return (s.astype(str).str.replace(r"\.0$", "", regex=True)
             .str.strip().str.zfill(6))


def collect(name, items, fn, workers=4, flush_every=200):
    """fn(key) -> DataFrame|None。断点续传 + 周期落盘。"""
    path = os.path.join(OUT, f"{name}.parquet")
    done_path = os.path.join(OUT, f"{name}.done.txt")
    fail_path = os.path.join(OUT, f"{name}.failed.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    rows = [pd.read_parquet(path)] if os.path.exists(path) else []
    todo = [k for k in items if k not in done]
    print(f"[{name}] total={len(items)} done={len(done)} todo={len(todo)}", flush=True)
    cnt = {"n": 0}

    def work(key):
        df = None
        for a in range(3):
            try:
                df = fn(key)
                break
            except Exception:
                if a == 2:
                    with lock:
                        open(fail_path, "a").write(f"{key}\n")
                time.sleep(2 * (a + 1))
        time.sleep(0.15)
        with lock:
            if df is not None and len(df):
                rows.append(df)
            open(done_path, "a").write(f"{key}\n")
            cnt["n"] += 1
            if cnt["n"] % flush_every == 0:
                if rows:
                    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
                print(f"[{name}] {cnt['n']}/{len(todo)} rows={sum(len(r) for r in rows)}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
    print(f"[{name}] DONE rows={sum(len(r) for r in rows)}", flush=True)


def months(start_i=0):
    seen = []
    for d in DATES:
        ym = d[:6]
        if ym not in seen:
            seen.append(ym)
    for ym in seen:
        ds = [d for d in DATES if d.startswith(ym)]
        yield ym, ds[0], ds[-1]


# 1) 个股资金流（深度受接口限制，约最近 120 个交易日）
def f_fund_flow(code):
    df = ak.stock_individual_fund_flow(
        stock=code, market="sh" if code[0] in "69" else "sz")
    df["code"] = code
    return df


# 2) 大宗交易（按月，全市场拉后过滤股票池）
def f_dzjy(key):
    ym, sd, ed = key.split("|")
    df = ak.stock_dzjy_mrmx(symbol="A股", start_date=sd, end_date=ed)
    cc = code_col(df)
    if cc is None:
        return None
    df[cc] = norm_codes(df[cc])
    return df[df[cc].isin(CODESET)]


# 3) 融资融券明细（按交易日，沪+深）
def f_margin(date):
    frames = []
    for fn, tag in ((ak.stock_margin_detail_sse, "sse"),
                    (ak.stock_margin_detail_szse, "szse")):
        try:
            df = fn(date=date)
        except Exception:
            continue
        cc = code_col(df)
        if cc is None:
            continue
        df[cc] = norm_codes(df[cc])
        df = df[df[cc].isin(CODESET)]
        df["exchange"] = tag
        df["date"] = date
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


# 4) 北向个股持股历史
def f_hsgt(code):
    df = ak.stock_hsgt_individual_em(symbol=code)
    df["code"] = code
    return df


# 5) 股东户数历史
def f_gdhs(code):
    df = ak.stock_zh_a_gdhs_detail_em(symbol=code)
    df["code"] = code
    return df


# 6) 机构调研明细（按交易日，全市场拉后过滤；接口签名是 date=单日）
def f_jgdy(date):
    df = ak.stock_jgdy_detail_em(date=date)
    cc = code_col(df)
    if cc is None:
        return None
    df[cc] = norm_codes(df[cc])
    df = df[df[cc].isin(CODESET)]
    df["date"] = date
    return df


# 7) 限售解禁（按年，全市场拉后过滤）
def f_restricted(year):
    ds = [d for d in DATES if d.startswith(year)]
    if not ds:
        return None
    df = ak.stock_restricted_release_detail_em(start_date=ds[0], end_date=ds[-1])
    cc = code_col(df)
    if cc is None:
        return None
    df[cc] = norm_codes(df[cc])
    return df[df[cc].isin(CODESET)]


# 8) 股权质押比例（接口无参、仅当前快照；历史无公开接口）
def f_pledge(key):
    df = ak.stock_gpzy_pledge_ratio_detail_em()
    cc = code_col(df)
    if cc is None:
        return None
    df[cc] = norm_codes(df[cc])
    df = df[df[cc].isin(CODESET)]
    df["snapshot_date"] = "current"
    return df


only = set(sys.argv[1:]) or None

# 9) 市场资金流（单次调用；走 push2，有过滤时跳过以免碰限流）
if not only:
    try:
        mf = ak.stock_market_fund_flow()
        mf.to_parquet(os.path.join(OUT, "market_fund_flow.parquet"), index=False)
        print(f"[market_fund_flow] DONE rows={len(mf)} cols={list(mf.columns)}", flush=True)
    except Exception as e:
        print(f"[market_fund_flow] FAILED {e!s:.120}", flush=True)

jobs = [
    ("dzjy", [f"{ym}|{sd}|{ed}" for ym, sd, ed in months()], f_dzjy, 4),
    ("hsgt_hold", CODES, f_hsgt, 4),
    ("gdhs", CODES, f_gdhs, 4),
    ("jgdy", DATES, f_jgdy, 4),
    ("restricted", [str(y) for y in range(2020, 2027)], f_restricted, 2),
    ("pledge", ["current"], f_pledge, 1),
    ("margin", DATES, f_margin, 4),
    # push2 接口易触发东方财富 IP 限流：放最后、低并发
    ("fund_flow", CODES, f_fund_flow, 2),
]
for name, items, fn, wk in jobs:
    if only and name not in only:
        print(f"[{name}] SKIPPED (filter)", flush=True)
        continue
    try:
        collect(name, items, fn, workers=wk,
                flush_every=200 if name != "margin" else 100)
    except Exception as e:
        print(f"[{name}] FAMILY FAILED {e!s:.200}", flush=True)

print("ALL FINISHED", flush=True)
