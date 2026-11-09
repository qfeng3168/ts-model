#!/usr/bin/env python3
"""龙虎榜拉取（东方财富，经 AKShare）：汇总层（全市场）+ 席位层（限股票池）。

必须在国内网络跑（G98）：东方财富从海外 IP 连不上。
- 汇总： ak.stock_lhb_detail_em 按月分段，2020-01-01 起，全市场保存。
- 席位： 对股票池内每个 (code, 上榜日) 调 stock_lhb_stock_detail_em（买入/卖出各一次），
  断点续传（done_flags.txt），失败记 failed.txt 不阻塞。
输出 /vol2/lhb/: lhb_summary.parquet, lhb_seats.parquet, run.log 由 nohup 重定向。
"""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import akshare as ak

OUT = "/vol2/lhb"
CODES_FILE = "/vol2/ts-ticks/tdx_codes.txt"
START, END = "20200101", "20261106"
os.makedirs(OUT, exist_ok=True)

lock = threading.Lock()
done_path = os.path.join(OUT, "done_flags.txt")
failed_path = os.path.join(OUT, "failed.txt")
done = set()
if os.path.exists(done_path):
    done = set(open(done_path).read().split())

codes = set(open(CODES_FILE).read().split())
print(f"universe={len(codes)}", flush=True)


def months(start, end):
    y, m = int(start[:4]), int(start[4:6])
    ey, em = int(end[:4]), int(end[4:6])
    while (y, m) <= (ey, em):
        last = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(1)
        yield f"{y}{m:02d}01", min(last.strftime("%Y%m%d"), end)
        m += 1
        if m == 13:
            y, m = y + 1, 1


# ---------- 汇总层 ----------
sp = os.path.join(OUT, "lhb_summary.parquet")
if os.path.exists(sp):
    summary = pd.read_parquet(sp)
    print(f"summary exists rows={len(summary)}", flush=True)
else:
    frames = []
    for sd, ed in months(START, END):
        for attempt in range(3):
            try:
                df = ak.stock_lhb_detail_em(start_date=sd, end_date=ed)
                break
            except Exception as e:
                print(f"summary {sd} attempt{attempt} err {e!s:.80}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"summary {sd} failed")
        frames.append(df)
        print(f"summary {sd}-{ed} rows={len(df)} total={sum(len(f) for f in frames)}", flush=True)
        time.sleep(0.5)
    summary = pd.concat(frames, ignore_index=True)
    summary["代码"] = summary["代码"].astype(str).str.zfill(6)
    summary.to_parquet(sp, index=False)
    print(f"summary saved rows={len(summary)}", flush=True)

summary["代码"] = summary["代码"].astype(str).str.zfill(6)
filt = summary[summary["代码"].isin(codes)]
pairs = sorted({(r["代码"], str(r["上榜日"])[:10].replace("-", ""))
                for _, r in filt.iterrows()})
print(f"universe stock-days on LHB: {len(pairs)} -> seat calls {2*len(pairs)}", flush=True)

# ---------- 席位层 ----------
seat_rows = []
sp2 = os.path.join(OUT, "lhb_seats.parquet")
counter = {"n": 0, "total": 2 * len(pairs)}


def fetch_flag(code, date, flag):
    key = f"{code}|{date}|{flag}"
    if key in done:
        with lock:
            counter["n"] += 1
        return
    df = None
    for attempt in range(3):
        try:
            df = ak.stock_lhb_stock_detail_em(symbol=code, date=date, flag=flag)
            break
        except Exception as e:
            if attempt == 2:
                with lock:
                    open(failed_path, "a").write(key + "\n")
            time.sleep(2 * (attempt + 1))
    with lock:
        if df is not None and len(df):
            d = df.copy()
            d["code"] = code
            d["date"] = date
            d["flag"] = flag
            seat_rows.append(d)
        with open(done_path, "a") as f:
            f.write(key + "\n")
        done.add(key)
        counter["n"] += 1
        if counter["n"] % 500 == 0:
            print(f"seats {counter['n']}/{counter['total']} rows={sum(len(x) for x in seat_rows)}", flush=True)
            if seat_rows:
                pd.concat(seat_rows, ignore_index=True).to_parquet(sp2, index=False)


with ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(fetch_flag, c, d, fl) for c, d in pairs for fl in ("买入", "卖出")]
    for f in futs:
        f.result()

if seat_rows:
    pd.concat(seat_rows, ignore_index=True).to_parquet(sp2, index=False)
print(f"FINISHED seats rows={sum(len(x) for x in seat_rows)} pairs={len(pairs)}", flush=True)
