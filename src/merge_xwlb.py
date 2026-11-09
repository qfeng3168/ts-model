#!/usr/bin/env python3
"""Merge all non-official Xinwen Lianbo sources into data/xwlb/xwlb.jsonl
plus coverage.json. Priority per date:
  1. GitHub archives (_repos.jsonl) when the record looks complete
     (>=8 items and >=3000 chars of text)
  2. mirror re-fetch of suspect repo days (_repair_mirror.jsonl) when it is
     more complete than the repo record
  3. mirror (cn.govopendata.com) for dates the archives do not cover
  4. KehaoWu corpus (_kehao.jsonl) as last-resort filler
The discarded official-source scrape (_official_discarded.jsonl) is never read.
"""
import datetime as dt
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "xwlb"
START = dt.date(2015, 1, 1)
END = dt.date(2026, 11, 7)


def load(name):
    out = {}
    p = OUT / name
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["date"]] = r
    return out


def complete(r):
    return len(r["items"]) >= 8 and len(r["text"]) >= 3000


def main():
    repos = load("_repos.jsonl")
    mirror = load("_mirror.jsonl")
    repair = load("_repair_mirror.jsonl")
    kehao = load("_kehao.jsonl")
    final = {}
    for d in sorted(set(repos) | set(mirror) | set(repair) | set(kehao)):
        try:
            dt.date.fromisoformat(d)
        except ValueError:
            print(f"skip invalid date key: {d!r}")
            continue
        cand = None
        if d in repos and complete(repos[d]):
            cand = repos[d]
        elif d in repair and d in repos and (
                (len(repair[d]["items"]), len(repair[d]["text"])) >
                (len(repos[d]["items"]), len(repos[d]["text"]))):
            cand = repair[d]
        elif d in mirror:
            cand = mirror[d]
        elif d in repos:
            cand = repos[d]
        elif d in repair:
            cand = repair[d]
        elif d in kehao:
            cand = kehao[d]
        if cand is not None and START.isoformat() <= d <= END.isoformat():
            final[d] = cand
    with open(OUT / "xwlb.jsonl", "w", encoding="utf-8") as f:
        for d in sorted(final):
            f.write(json.dumps(final[d], ensure_ascii=False) + "\n")
    per_year, sources = {}, {}
    for d, r in final.items():
        per_year[d[:4]] = per_year.get(d[:4], 0) + 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    missing, low_text = [], []
    d = START
    while d <= END:
        ds = d.isoformat()
        if ds not in final:
            missing.append(ds)
        elif len(final[ds]["items"]) < 5 or len(final[ds]["text"]) < 1500:
            low_text.append({"date": ds, "items": len(final[ds]["items"]),
                             "text_chars": len(final[ds]["text"]),
                             "source": final[ds]["source"]})
        d += dt.timedelta(days=1)
    cov = {"range": [START.isoformat(), END.isoformat()],
           "total_days_in_range": (END - START).days + 1,
           "records": len(final),
           "per_year": dict(sorted(per_year.items())),
           "by_source": sources,
           "missing_count": len(missing),
           "missing_dates": missing,
           "low_text_days": low_text}
    (OUT / "coverage.json").write_text(
        json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: cov[k] for k in
                      ("records", "per_year", "by_source", "missing_count",
                       "missing_dates")}, ensure_ascii=False))
    print("low_text_days:", len(low_text))


if __name__ == "__main__":
    main()
