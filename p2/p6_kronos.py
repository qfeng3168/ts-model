"""P6: zero-shot Kronos-small scoring, sharded across all visible GPUs.

For each rebalance date (every 5th trading day, 2020-2026) and stock:
context = past 90 daily OHLCV bars, sample 5 future bars x5 paths,
score = mean predicted close[-1] / last close - 1.
Dates are sharded across worker processes (one per GPU; CPU fallback).
Each shard checkpoints its CSV every 20 dates.
Output: p6_predictions.csv + summary.
"""
import glob
import multiprocessing as mp
import subprocess
import sys
import time

subprocess.run(["git", "clone", "-q", "--depth", "1",
                "https://github.com/shiyu-coder/Kronos",
                "/kaggle/working/Kronos"], check=False)
sys.path.insert(0, "/kaggle/working/Kronos")

import numpy as np
import pandas as pd
import torch

_hits = glob.glob("/kaggle/input/**/daily.parquet", recursive=True)
DATA = _hits[0]
CTX, HORIZON, SAMPLES = 90, 5, 5


def load():
    daily = pd.read_parquet(DATA)
    daily["dkey"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    daily["fwd_ret_5"] = daily.groupby("code")["close"].transform(
        lambda s: s.shift(-5) / s - 1)
    return daily


def shard(shard_id, dates, device, out_csv):
    from model import Kronos, KronosPredictor, KronosTokenizer
    daily = load()
    by_code = {c: g.reset_index(drop=True)
               for c, g in daily.groupby("code")}
    date_pos = {c: {d: i for i, d in enumerate(g["dkey"])}
                for c, g in by_code.items()}
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pm = KronosPredictor(model, tok, device=device, max_context=512)
    rows = []
    t0 = time.time()
    for di, d in enumerate(dates):
        frames, xs, ys, meta = [], [], [], []
        for code, g in by_code.items():
            i = date_pos[code].get(d)
            if i is None or i < CTX or i + HORIZON >= len(g):
                continue
            ctx = g.iloc[i - CTX + 1: i + 1]
            frames.append(ctx[["open", "high", "low", "close", "volume",
                               "amount"]].astype(np.float32))
            xs.append(pd.Series(ctx["date"]).reset_index(drop=True))
            ys.append(pd.Series(pd.bdate_range(
                pd.Timestamp(d) + pd.Timedelta(days=1), periods=HORIZON)))
            meta.append((code, float(ctx["close"].iloc[-1]),
                         float(g["fwd_ret_5"].iloc[i])))
        if not frames:
            continue
        out = pm.predict_batch(frames, xs, ys, pred_len=HORIZON, T=1.0,
                               top_p=0.9, sample_count=SAMPLES,
                               verbose=False)
        for (code, last_close, fwd), o in zip(meta, out):
            rows.append({"date": d, "code": code,
                         "pred": float(o["close"].iloc[-1]) / last_close - 1,
                         "fwd_ret_5": fwd})
        if di % 20 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"shard{shard_id} [{di}/{len(dates)}] rows={len(rows)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"shard{shard_id} DONE rows={len(rows)}", flush=True)


def main():
    daily = load()
    all_dates = np.sort(daily["dkey"].unique())
    test_dates = [d for d in all_dates if d[:4] >= "2020"][::5]
    n_gpu = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_gpu)] or ["cpu"]
    print(f"devices={devices} dates={len(test_dates)}", flush=True)
    procs = []
    for k, dev in enumerate(devices):
        p = mp.Process(target=shard,
                       args=(k, test_dates[k::len(devices)], dev,
                             f"p6_part{k}.csv"))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    parts = [pd.read_csv(f) for f in glob.glob("p6_part*.csv")]
    res = pd.concat(parts, ignore_index=True)
    res.to_csv("p6_predictions.csv", index=False)

    def spearman(a, b):
        return pd.Series(a).rank().corr(pd.Series(b).rank())

    ics = [spearman(g["pred"], g["fwd_ret_5"])
           for _, g in res.groupby("date") if len(g) >= 20]
    print("P6 Kronos zero-shot: n_dates", len(ics),
          "mean RankIC", round(float(np.mean(ics)), 4), flush=True)
    rets = []
    for d in sorted(res["date"].unique()):
        g = res[res["date"] == d]
        k = max(1, int(len(g) * 0.2))
        rets.append(g.nlargest(k, "pred")["fwd_ret_5"].mean() - 0.0015)
    r = np.array(rets)
    cum = np.cumprod(1 + r)
    print(f"portfolio: ann={cum[-1] ** (50 / len(r)) - 1:.3f} "
          f"sharpe={r.mean() / r.std() * np.sqrt(50):.2f} "
          f"total={cum[-1] - 1:.2f}", flush=True)


if __name__ == "__main__":
    main()
