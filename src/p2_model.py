"""P2 自研架构 v1：PatchTST 式通道独立 patch Transformer（截面排序）。

与 P1 同口径对照：同标签 fwd_ret_5（取自 dataset.parquet，保证一致）、
同 walk-forward 按年测试、同组合评估（复用 backtest.py 的函数）。
差异只在模型：60 日窗口 × 6 个因果通道 → patch(10/5) → 共享 Transformer
编码（通道独立）→ 截面分数。训练目标为日期内秩百分位（[-.5,.5]），
与 P1 的原始收益回归的差异在报告中注明。

用法：
  python src/p2_model.py [--years 2020,...,2026] [--epochs 30] [--smoke]
  Kaggle 上自动识别 /kaggle/input/ts-model-ashare 作为数据根。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
except ImportError:  # standalone (Kaggle kernel): verbatim copies from backtest.py
    from scipy.stats import spearmanr

    def rank_ic(d: pd.DataFrame, pred_col: str) -> float:
        ics = []
        for _, g in d.groupby("date"):
            if len(g) >= 20:
                ic, _ = spearmanr(g[pred_col], g["fwd_ret_5"])
                if np.isfinite(ic):
                    ics.append(ic)
        return float(np.mean(ics)) if ics else float("nan")

    def portfolio(d: pd.DataFrame, pred_col: str) -> pd.Series:
        dates = sorted(d["date"].unique())
        rebal = dates[::5]
        rets = []
        for dt in rebal:
            g = d[d["date"] == dt]
            if len(g) < 20:
                continue
            k = max(1, int(len(g) * 0.2))
            top = g.nlargest(k, pred_col)
            rets.append(top["fwd_ret_5"].mean() - 0.0015)
        return pd.Series(rets, index=pd.to_datetime(rebal[: len(rets)]))

    def ann_stats(rets: pd.Series, periods_per_year: int = 50) -> dict:
        if len(rets) == 0:
            return {}
        cum = (1 + rets).cumprod()
        total = cum.iloc[-1] - 1
        years = len(rets) / periods_per_year
        ann = (1 + total) ** (1 / years) - 1 if years > 0 else float("nan")
        dd = (cum / cum.cummax() - 1).min()
        sharpe = rets.mean() / rets.std() * np.sqrt(periods_per_year) \
            if rets.std() > 0 else float("nan")
        return {"total_return": float(total), "ann_return": float(ann),
                "max_drawdown": float(dd), "sharpe": float(sharpe),
                "periods": int(len(rets))}

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 60
PATCH, STRIDE = 10, 5
N_PATCH = (WINDOW - PATCH) // STRIDE + 1
CHANNELS = ["ret1", "range", "body", "gap", "volr", "turn"]
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def data_root() -> Path:
    for cand in (Path("/kaggle/input/ts-model-ashare"),
                 Path("/kaggle/input/datasets/willowwaelchi/ts-model-ashare"),
                 ROOT / "data"):
        if (cand / "daily.parquet").exists():
            return cand
    return ROOT / "data"


def build_series(root: Path):
    """Per-stock causal channel arrays + label lookup from dataset.parquet."""
    daily = pd.read_parquet(root / "daily.parquet").sort_values(["code", "date"])
    ds = pd.read_parquet(root / "dataset.parquet",
                         columns=["date", "code", "fwd_ret_5"])
    # date keys as ISO strings: Timestamp vs np.datetime64 hashing differs
    # across pandas versions (bit us on Kaggle), strings are stable.
    labels = {(c, d): v for (c, d), v in
              zip(zip(ds["code"], ds["date"].dt.strftime("%Y-%m-%d")),
                  ds["fwd_ret_5"])}
    series = {}
    for code, g in daily.groupby("code"):
        g = g.reset_index(drop=True)
        close, vol = g["close"], g["volume"]
        ch = np.column_stack([
            close.pct_change().fillna(0).to_numpy(),
            ((g["high"] - g["low"]) / close).to_numpy(),
            (close / g["open"] - 1).to_numpy(),
            (g["open"] / close.shift(1) - 1).fillna(0).to_numpy(),
            (vol / vol.rolling(20).mean() - 1).to_numpy(),
            np.log1p(g["turnover"]).to_numpy(),
        ]).astype(np.float32)
        ch[~np.isfinite(ch)] = 0.0
        # all length-WINDOW windows, per-window per-channel z-scored (causal:
        # statistics come from inside the window only). win[i] ends at pos i+59.
        win = np.lib.stride_tricks.sliding_window_view(
            ch, WINDOW, axis=0)                       # (n, C, L)
        mu = win.mean(axis=2, keepdims=True)
        sd = win.std(axis=2, keepdims=True) + 1e-6
        series[code] = {"dates": g["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                        "win": ((win - mu) / sd).astype(np.float32)}
    return series, labels


def make_samples(series, labels, years: set[int]):
    """(code, end_pos) samples whose date's year is in `years` and label exists."""
    idx = []
    for code, s in series.items():
        dates = s["dates"]
        for pos in range(WINDOW - 1, len(dates)):
            d = dates[pos]
            if pd.Timestamp(d).year in years and (code, d) in labels:
                idx.append((code, pos))
    return idx


def rank_targets(series, labels, idx):
    """Within-date rank percentile of the label, mapped to [-0.5, 0.5]."""
    by_date: dict = {}
    for code, pos in idx:
        d = series[code]["dates"][pos]
        by_date.setdefault(d, []).append((code, pos))
    tgt = {}
    for d, items in by_date.items():
        vals = np.array([labels[(c, series[c]["dates"][p])] for c, p in items])
        order = vals.argsort().argsort()
        pct = order / max(1, len(vals) - 1) - 0.5
        for (c, p), v in zip(items, pct):
            tgt[(c, p)] = float(v)
    return tgt


class PatchTS(nn.Module):
    def __init__(self, n_ch=6, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(PATCH, d_model)
        self.pos = nn.Parameter(torch.randn(N_PATCH, d_model) * 0.02)
        self.chan = nn.Parameter(torch.randn(n_ch, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                         dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(),
                                  nn.Linear(32, 1))

    def forward(self, x):  # x: (B, C, L)
        b, c, _ = x.shape
        patches = x.unfold(2, PATCH, STRIDE)              # (B, C, P, patch)
        h = self.embed(patches) + self.pos + self.chan[:, None, :]
        h = h.reshape(b * c, N_PATCH, -1)
        h = self.enc(h).mean(1).reshape(b, c, -1).mean(1)  # pool patches+channels
        return self.head(h).squeeze(-1)


def fold_tensor(series, idx):
    """One (N, C, L) float32 tensor for a whole fold, in sample order."""
    parts = [series[c]["win"][p - WINDOW + 1] for c, p in idx]
    return torch.from_numpy(np.stack(parts))


@torch.no_grad()
def predict(model, xt: torch.Tensor, device, bs=8192):
    model.eval()
    out = np.empty(len(xt), dtype=np.float64)
    for i in range(0, len(xt), bs):
        out[i:i + bs] = model(xt[i:i + bs].to(device)).cpu().numpy()
    return out


def frame_of(series, labels, idx, pred):
    rows = [(series[c]["dates"][p], c, labels[(c, series[c]["dates"][p])], pr)
            for (c, p), pr in zip(idx, pred)]
    return pd.DataFrame(rows, columns=["date", "code", "fwd_ret_5", "pred"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=",".join(map(str, TEST_YEARS)))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch, first test year only, subset of train dates")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",")]
    if args.smoke:
        years = years[:1]
        args.epochs = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)
    root = data_root()
    series, labels = build_series(root)
    print(f"stocks={len(series)} labels={len(labels)}", flush=True)

    rows, all_test = [], []
    for y in years:
        t0 = time.time()
        train_idx = make_samples(series, labels, set(range(2015, y - 1)))
        val_idx = make_samples(series, labels, {y - 1})
        test_idx = make_samples(series, labels, {y})
        if args.smoke:
            train_idx = train_idx[:20000]
            val_idx = val_idx[:5000]
        tgt = rank_targets(series, labels, train_idx)
        y_train = torch.tensor([tgt[k] for k in train_idx], dtype=torch.float32)
        x_train = fold_tensor(series, train_idx).to(device)
        y_train = y_train.to(device)
        x_val = fold_tensor(series, val_idx)
        x_test = fold_tensor(series, test_idx)
        print(f"  y{y}: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)}", flush=True)

        torch.manual_seed(42)
        model = PatchTS(len(CHANNELS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        best_ic, best_state = -1e9, None
        n = len(train_idx)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, 4096):
                sel = perm[i:i + 4096]
                loss = nn.functional.mse_loss(model(x_train[sel]), y_train[sel])
                opt.zero_grad()
                loss.backward()
                opt.step()
            vic = rank_ic(frame_of(series, labels, val_idx,
                                   predict(model, x_val, device)), "pred")
            if vic > best_ic:
                best_ic = vic
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  y{y} ep{ep}: val_rank_ic={vic:.4f}", flush=True)
        if best_state is not None:
            model.load_state_dict(best_state)
        test_frame = frame_of(series, labels, test_idx,
                              predict(model, x_test, device))
        ic = rank_ic(test_frame, "pred")
        dir_acc = float(((test_frame["pred"] > 0) ==
                         (test_frame["fwd_ret_5"] > 0)).mean())
        rows.append({"model": "patchtst", "year": y, "rank_ic": ic,
                     "dir_acc": dir_acc, "n": int(len(test_frame))})
        test_frame["model"] = "patchtst"
        all_test.append(test_frame)
        ckpt = Path("/kaggle/working") if Path("/kaggle/working").exists() \
            else ROOT / "reports"
        test_frame.to_parquet(ckpt / f"p2_preds_{y}.parquet", index=False)
        if best_state is not None:
            torch.save(best_state, ckpt / f"p2_model_{y}.pt")
        pd.DataFrame(rows).to_csv(ckpt / "p2_results_partial.csv",
                                  index=False)
        print(f"year {y}: rank_ic={ic:.4f} dir_acc={dir_acc:.4f} "
              f"n={len(test_frame)} ({time.time() - t0:.0f}s)", flush=True)

    cat = pd.concat(all_test, ignore_index=True)
    rets = portfolio(cat, "pred")
    stats = ann_stats(rets)
    rows.append({"model": "patchtst", "year": "ALL", **stats})
    rep = pd.DataFrame(rows)
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() \
        else ROOT / "reports"
    lines = ["# P2 PatchTST 式模型回测（A股日线 / 沪深300 / 未来5日收益）", "",
             "评估口径与 P1 一致（walk-forward 按年测试、top20% 每5日调仓、单边成本 0.15%）；",
             "模型差异：60日×6通道 patch Transformer，训练目标为日期内秩百分位；",
             "训练集为测试年前两年起的扩张窗口，测试前一年单独留作早停验证（P1 无此划分）。",
             "注：成分股为当前名单，存在幸存者偏差。", "",
             rep.to_markdown(index=False, floatfmt=".4f"), ""]
    (out_dir / "p2_patchtst.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame({"patchtst": (1 + rets).cumprod()}).to_csv(
        out_dir / "p2_curves.csv")
    cat.to_parquet(out_dir / "p2_predictions.parquet", index=False)
    print(rep.to_string(index=False))
    print(json.dumps({"out_dir": str(out_dir), "stats": stats}))


if __name__ == "__main__":
    main()
