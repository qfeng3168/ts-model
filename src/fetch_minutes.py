"""分钟线采集（5 分钟，前复权，新浪源）：沪深300成分股，逐股合并落盘、按时间戳去重。

同一脚本同时服务两个用途：
- 首次运行 = 拉回最近约 2 个月 5 分钟线（新浪上限约 1970 根）做小样本；
- 每日收盘后运行 = 增量补当天，与已存数据合并去重，历史深度逐日累积。
"""
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "minute5" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
PERIOD = "5"


def _prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def fetch_one(code: str) -> int:
    fp = RAW / f"{code}.parquet"
    try:
        df = ak.stock_zh_a_minute(symbol=f"{_prefix(code)}{code}", period=PERIOD, adjust="qfq")
    except Exception as e:
        print(f"  {code} failed: {e}", flush=True)
        return 0
    if df is None or df.empty:
        return 0
    df = df.rename(columns={"day": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["code"] = code
    df = df[["ts", "open", "high", "low", "close", "volume", "code"]]
    if fp.exists():
        old = pd.read_parquet(fp)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df.to_parquet(fp, index=False)
    return len(df)


def main() -> None:
    cons = ak.index_stock_cons_csindex(symbol="000300")
    codes = sorted(cons["成分券代码"].astype(str).str.zfill(6).unique())
    print(f"minute{PERIOD} constituents: {len(codes)}", flush=True)
    ok = 0
    for i, code in enumerate(codes, 1):
        if fetch_one(code) > 0:
            ok += 1
        if i % 50 == 0:
            print(f"[{i}/{len(codes)}] ok={ok}", flush=True)
        time.sleep(0.15)
    files = list(RAW.glob("*.parquet"))
    total = sum(len(pd.read_parquet(p, columns=["ts"])) for p in files)
    print(f"done: files={len(files)} total_bars={total}", flush=True)


if __name__ == "__main__":
    main()
