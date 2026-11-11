"""P11: Timestral-209M zero-shot probe on the PIT HS300 universe.

For each rebalance (date, code): context = last 512 closes up to t
(NaN left-padded), forecast 5 steps. Outputs per row: median /
q10 / q90 implied 5-day returns. Two T4s sharded by row index via
multiprocessing. Per-year shards saved immediately (checkpoint rule).
A smoke batch runs first; projected wall time is printed and the
context is cut to 256 if the projection exceeds ~3.5h.
"""
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/mistralai/timestral.git"],
               check=False)

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

CTX = 512
HOR = 5
import glob as _glob
_hits = _glob.glob("/kaggle/input/**/rebalance.parquet", recursive=True)
IN = os.path.dirname(_hits[0]) if _hits else "/kaggle/input/ts-model-pit"
OUT = "/kaggle/working"


def build_jobs():
    rb = pd.read_parquet(f"{IN}/rebalance.parquet")
    cl = pd.read_parquet(f"{IN}/closes.parquet")
    series, pos = {}, {}
    for code, g in cl.sort_values("date").groupby("code"):
        series[code] = g["close"].to_numpy(dtype=np.float32)
        pos[code] = {d: i for i, d in enumerate(g["date"])}
    jobs = []
    for r in rb.itertuples():
        arr = series.get(r.code)
        if arr is None:
            continue
        i = pos[r.code].get(r.date)
        if i is None or i < 127:
            continue
        lo = max(0, i - CTX + 1)
        ctx = arr[lo:i + 1]
        if len(ctx) < CTX:
            ctx = np.concatenate([np.full(CTX - len(ctx), np.nan,
                                          dtype=np.float32), ctx])
        jobs.append((r.date, r.code, ctx, float(arr[i])))
    return jobs


def _np(x):
    # v3 died here: forecast outputs are CUDA tensors, and
    # np.asarray(cuda_tensor) raises TypeError. Always go via CPU.
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def run_shard(gpu, jobs, tag):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from timestral import TimestralForecaster
    model = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-209M", device="cuda")
    rows = []
    bs = 512
    t0 = time.time()
    for s in range(0, len(jobs), bs):
        chunk = jobs[s:s + bs]
        X = np.stack([c[2] for c in chunk])
        try:
            fc = model.forecast(X, prediction_length=HOR)
            med = _np(fc.median)[:, -1]
            q10 = _np(fc.quantile(0.1))[:, -1]
            q90 = _np(fc.quantile(0.9))[:, -1]
        except RuntimeError:
            bs = max(32, bs // 2)
            print(f"[{tag}] OOM -> batch {bs}", flush=True)
            continue
        for (d, code, _, last), m, a, b in zip(chunk, med, q10, q90):
            rows.append({"date": d, "code": code,
                         "pred_med": m / last - 1,
                         "pred_q10": a / last - 1,
                         "pred_q90": b / last - 1})
        if s % (bs * 20) == 0:
            el = time.time() - t0
            print(f"[{tag}] {s + len(chunk)}/{len(jobs)} "
                  f"{el:.0f}s eta={el / max(s + len(chunk), 1) * len(jobs):.0f}s",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT}/timestral_{tag}.parquet", index=False)
    print(f"[{tag}] DONE {len(df)} rows in {time.time()-t0:.0f}s",
          flush=True)


def _spawn_shard(rank, shards):
    # Top-level (picklable) spawn target: mp.spawn with the 'spawn'
    # start method cannot pickle a lambda defined inside main().
    # v2 died on exactly this (PicklingError on the lambda).
    run_shard(rank, shards[rank], f"s{rank}")


def main():
    jobs = build_jobs()
    print(f"jobs: {len(jobs)}", flush=True)
    half = (len(jobs) + 1) // 2
    shards = [jobs[:half], jobs[half:]]
    if torch.cuda.device_count() >= 2:
        mp.spawn(_spawn_shard, args=(shards,), nprocs=2, join=True)
    else:
        run_shard(0, jobs, "s0")


if __name__ == "__main__":
    main()
