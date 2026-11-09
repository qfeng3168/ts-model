#!/usr/bin/env python3
"""Fetch historical tick (fenbi) data from Tongdaxin (TDX) HQ servers via pytdx.

Free, no account. Must run on a CN network (TDX servers reset overseas IPs).
mootdx's high-level transaction() returns 0 for history dates (verified
2026-11-09) - use the raw pytdx get_history_transaction_data with paging.

Layout: OUT/{code}/{yyyymmdd}.parquet, same schema as data/ticks/ from
fetch_ticks.py: time, price, vol, buyorsell, volume(cumsum), date, code.
Progress is tracked in OUT/done.txt (code date rows) so reruns resume.
Failed stock-days land in OUT/failed.txt and never stall the crawl.
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pytdx.hq import TdxHq_API

SERVERS = [
    ("119.147.212.81", 7709),
    ("112.74.214.43", 7727),
    ("221.231.141.60", 7709),
    ("101.227.73.20", 7709),
    ("101.227.77.254", 7709),
    ("14.215.128.18", 7709),
    ("59.173.18.140", 7709),
    ("218.75.126.9", 7709),
]
PAGE = 2000


def market_of(code: str) -> int:
    return 1 if code.startswith(("6", "9")) else 0


class Worker:
    """One TDX connection; rotates servers on failure."""

    def __init__(self, seed: int):
        self.srv = seed % len(SERVERS)
        self.api = None
        self.connect()

    def connect(self):
        last = None
        for _ in range(len(SERVERS)):
            ip, port = SERVERS[self.srv % len(SERVERS)]
            self.srv += 1
            try:
                api = TdxHq_API(heartbeat=True)
                if api.connect(ip, port, time_out=5):
                    self.api = api
                    return
            except Exception as exc:  # try next server
                last = exc
        raise RuntimeError(f"no TDX server reachable: {last}")

    def fetch_day(self, code: str, date: int) -> list:
        """All tick rows for one stock-day, chronological. Retries w/ reconnect."""
        for attempt in range(5):
            try:
                # Pages tile the day backwards: page 0 is the newest chunk,
                # each chunk internally chronological. Correct assembly is
                # chunks in reverse fetch order, each as-is. (A flat reverse
                # scrambles the day - verified in the 2026-11-09 pilot.)
                chunks, start = [], 0
                while True:
                    data = self.api.get_history_transaction_data(
                        market_of(code), code, start, PAGE, date)
                    if not data:
                        break
                    chunks.append(data)
                    if len(data) < PAGE:
                        break
                    start += len(data)
                    time.sleep(0.03)
                return [r for chunk in reversed(chunks) for r in chunk]
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)
                try:
                    self.api.disconnect()
                except Exception:
                    pass
                self.connect()
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--dates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="max stock-days (pilot)")
    args = ap.parse_args()

    codes = [l.strip() for l in open(args.codes) if l.strip()]
    dates = [l.strip() for l in open(args.dates) if l.strip()]
    out = args.out
    os.makedirs(out, exist_ok=True)
    done_path = os.path.join(out, "done.txt")
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            parts = line.split()
            if len(parts) >= 2:
                done.add((parts[0], parts[1]))

    tasks = [(c, d) for d in dates for c in codes if (c, d) not in done]
    if args.limit:
        tasks = tasks[: args.limit]
    total = len(tasks)
    print(f"stock-days todo: {total} (already done: {len(done)})", flush=True)
    if not total:
        print("ALL DONE", flush=True)
        return

    lock = threading.Lock()
    state = {"n": 0, "rows": 0, "t0": time.time()}
    done_f = open(done_path, "a")
    fail_f = open(os.path.join(out, "failed.txt"), "a")
    tls = threading.local()

    def get_worker() -> Worker:
        if not hasattr(tls, "worker"):
            tls.worker = Worker(threading.get_ident() % len(SERVERS))
        return tls.worker

    def process(task):
        code, dstr = task
        try:
            rows = get_worker().fetch_day(code, int(dstr))
        except Exception as exc:
            with lock:
                fail_f.write(f"{code} {dstr} {type(exc).__name__}\n")
                fail_f.flush()
                state["n"] += 1
            return
        if rows:
            df = pd.DataFrame(rows)[["time", "price", "vol", "buyorsell"]]
            df["volume"] = df["vol"].cumsum()
            df["date"] = dstr
            df["code"] = code
            cdir = os.path.join(out, code)
            os.makedirs(cdir, exist_ok=True)
            tmp = os.path.join(cdir, f".{dstr}.tmp.parquet")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, os.path.join(cdir, f"{dstr}.parquet"))
        with lock:
            done_f.write(f"{code} {dstr} {len(rows)}\n")
            done_f.flush()
            state["rows"] += len(rows)
            state["n"] += 1
            if state["n"] % 200 == 0 or state["n"] == total:
                el = time.time() - state["t0"]
                rate = state["n"] / el
                eta = (total - state["n"]) / rate / 3600 if rate else -1
                print(f"[{state['n']}/{total}] rows={state['rows']:,} "
                      f"rate={rate:.2f}/s eta={eta:.1f}h", flush=True)

    if args.workers <= 1:
        for t in tasks:
            process(t)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(process, tasks))

    print(f"FINISHED n={state['n']} rows={state['rows']:,}", flush=True)


if __name__ == "__main__":
    main()
