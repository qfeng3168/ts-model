"""Concept board memberships via AKShare / eastmoney.

(THS's info endpoint returns board quotes, not constituents, in this
akshare version; EM cons works.) Checkpointed: finished concepts are
recorded in reports/fetch_concepts.done and skipped on rerun.
Output: data/f10/concepts_em.parquet (concept, code) - all A-share pairs.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "reports" / "fetch_concepts.log"
DONE = ROOT / "reports" / "fetch_concepts.done"
TMP = ROOT / "data" / "f10" / "concepts_parts"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    done = set()
    if DONE.exists():
        done = set(DONE.read_text(encoding="utf-8").split())
    names = ak.stock_board_concept_name_em()
    todo = [n for n in names["板块名称"] if n not in done]
    log(f"concepts total={len(names)} done={len(done)} todo={len(todo)}")
    fail = []
    for i, name in enumerate(todo, 1):
        try:
            cons = ak.stock_board_concept_cons_em(symbol=name)
            codes = cons["代码"].astype(str).str.zfill(6)
            part = pd.DataFrame({"concept": name, "code": codes})
            part.to_parquet(TMP / f"{len(done) + i:04d}.parquet",
                            index=False)
            with open(DONE, "a", encoding="utf-8") as fh:
                fh.write(name + "\n")
        except Exception:  # noqa: BLE001
            fail.append(name)
            log(f"[{i}] {name} FAIL {traceback.format_exc(limit=1)[-120:]}")
        if i % 40 == 0:
            log(f"[{i}/{len(todo)}] fail={len(fail)}")
        time.sleep(2.5)
    parts = [pd.read_parquet(p) for p in sorted(TMP.glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    df.to_parquet(ROOT / "data/f10/concepts_em.parquet", index=False)
    log(f"DONE pairs={len(df):,} stocks={df['code'].nunique()} "
        f"concepts={df['concept'].nunique()} fail={fail}")


if __name__ == "__main__":
    main()
