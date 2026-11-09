"""Second-wave fetch: everything the factor checklist marked missing.

Sections (independent, failures logged, never fatal):
  restricted_sina   限售解禁 (sina queue; eastmoney was empty from this IP)
  ggcg              股东增减持 (eastmoney)
  hold_change       增减持 (cninfo, latest quarter probe)
  repurchase        股票回购 (eastmoney)
  forecast_ths      同花顺一致预期 EPS, per HS300 stock (checkpointed)
  qvix300           沪深300 中国波指 (implied vol)
  gdp / unemployment 宏观补充
  fundflow          个股资金流历史, per HS300 stock (checkpointed)
  futures           大宗商品期货主连一批 (sina): 产业链价格传导层
  holder_history    股东人数历史截面 (cninfo, 按季调用, 每次全市场)
Output: data/extra/*.parquet, data/extra/futures/<SYM>.parquet
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "extra"
LOG = ROOT / "reports" / "fetch_missing.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save(name: str, df) -> None:
    if df is None or len(df) == 0:
        log(f"{name}: EMPTY")
        return
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    log(f"{name}: {len(df):,} rows, cols={list(df.columns)[:6]}")


def hs300() -> list[str]:
    return sorted(pd.read_parquet(
        ROOT / "data/daily.parquet", columns=["code"])["code"].unique())


def section(name, fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        log(f"{name}: FAIL {traceback.format_exc(limit=1)[-200:]}")
    time.sleep(1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    section("restricted_sina", lambda: save(
        "restricted_release_sina",
        ak.stock_restricted_release_queue_sina(symbol="全部股票")))
    section("ggcg", lambda: save(
        "ggcg", ak.stock_ggcg_em(symbol="全部股票")))
    section("repurchase", lambda: save(
        "repurchase", ak.stock_repurchase_em()))
    section("qvix300", lambda: save(
        "qvix_300etf", ak.index_option_300etf_qvix()))
    section("gdp", lambda: save("gdp", ak.macro_china_gdp()))
    section("unemployment", lambda: save(
        "unemployment", ak.macro_china_urban_unemployment()))

    def futures():
        fut_dir = OUT / "futures"
        fut_dir.mkdir(exist_ok=True)
        syms = ["RB0", "HC0", "I0", "J0", "JM0", "MA0", "TA0", "PP0",
                "L0", "V0", "AL0", "ZN0", "NI0", "SN0", "AG0", "FU0",
                "RU0", "BU0", "M0", "Y0", "P0", "C0", "SR0", "CF0",
                "OI0", "RM0", "AP0", "EG0", "UR0", "SA0"]
        ok, bad = 0, []
        for s in syms:
            p = fut_dir / f"{s}.parquet"
            if p.exists():
                ok += 1
                continue
            try:
                d = ak.futures_main_sina(symbol=s, start_date="20150101",
                                         end_date="20261109")
                if d is not None and len(d):
                    d.to_parquet(p, index=False)
                    ok += 1
                else:
                    bad.append(s)
            except Exception:  # noqa: BLE001
                bad.append(s)
            time.sleep(0.5)
        log(f"futures: ok={ok}/{len(syms)} bad={bad}")
    section("futures", futures)

    def holder_history():
        parts = []
        qends = pd.date_range("2015-03-31", "2026-09-30", freq="QE")
        for q in qends:
            try:
                d = ak.stock_hold_num_cninfo(date=q.strftime("%Y%m%d"))
                if d is not None and len(d):
                    parts.append(d)
            except Exception:  # noqa: BLE001
                log(f"holder {q.date()} fail")
            time.sleep(0.5)
        if parts:
            save("holder_history", pd.concat(parts, ignore_index=True))
        else:
            log("holder_history: EMPTY")
    section("holder_history", holder_history)

    def per_stock(name, fn, key):
        parts_dir = OUT / f"{name}_parts"
        parts_dir.mkdir(exist_ok=True)
        codes = hs300()
        done = {p.stem for p in parts_dir.glob("*.parquet")}
        todo = [c for c in codes if c not in done]
        log(f"{name}: todo={len(todo)}/{len(codes)}")
        for i, code in enumerate(todo, 1):
            try:
                d = fn(code)
                if d is not None and len(d):
                    d = d.copy()
                    d["code"] = code
                    d.to_parquet(parts_dir / f"{code}.parquet", index=False)
            except Exception:  # noqa: BLE001
                pass
            if i % 60 == 0:
                log(f"{name}: [{i}/{len(todo)}]")
            time.sleep(0.4)
        parts = [pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet"))]
        if parts:
            save(name, pd.concat(parts, ignore_index=True))
    section("forecast_ths", lambda: per_stock(
        "consensus_eps",
        lambda c: ak.stock_profit_forecast_ths(
            symbol=c, indicator="预测年报每股收益"), "code"))
    section("fundflow", lambda: per_stock(
        "fund_flow_hist",
        lambda c: ak.stock_individual_fund_flow(
            stock=c, market="sh" if c.startswith("6") else "sz"), "code"))
    log("ALL DONE")


if __name__ == "__main__":
    main()
