"""Kaggle GPU smoke test: proves the kernel push channel, phone verification,
GPU availability, and that the ts-model-ashare dataset mounts readable."""
import json
import os

result = {"gpu": False, "dataset": None}

try:
    import torch

    result["torch"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        result["gpu"] = True
        result["device"] = torch.cuda.get_device_name(0)
        result["matmul_sum"] = float(y.sum().item())
except Exception as exc:  # noqa: BLE001
    result["torch_error"] = repr(exc)

for root in ("/kaggle/input/ts-model-ashare", "/kaggle/input/datasets/willowwaelchi/ts-model-ashare"):
    if os.path.isdir(root):
        import pandas as pd

        ds = pd.read_parquet(os.path.join(root, "dataset.parquet"))
        result["dataset"] = {"path": root, "rows": len(ds), "cols": len(ds.columns)}
        break

with open("smoke_result.json", "w") as fh:
    json.dump(result, fh, indent=2)
print("SMOKE_RESULT", json.dumps(result))
print("SMOKE_OK" if result["gpu"] and result["dataset"] else "SMOKE_INCOMPLETE")
