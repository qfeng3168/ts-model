#!/usr/bin/env python3
"""Fetch Xinwen Lianbo full text from the third-party mirror
cn.govopendata.com for every date in 2015-01-01..2026-11-07 that the
GitHub archives in data/xwlb/_repos.jsonl do not cover.

Day page: https://cn.govopendata.com/xinwenlianbo/{YYYYMMDD}/
Items are <h2> headings followed by <p> paragraphs inside #main-content.

Politeness: single thread, 0.5 s between requests, abort after 8
consecutive failed days. Resumable: appends to data/xwlb/_mirror.jsonl.
Also supports: fetch specific extra dates for cross-source verification
(--dates a,b,c --out FILE).
"""

import argparse
import datetime as dt
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "xwlb"
START = dt.date(2015, 1, 1)
END = dt.date(2026, 11, 7)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
SLEEP = 0.5
session = requests.Session()
session.headers.update(HEADERS)


def load_jsonl(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["date"]] = rec
    return out


def clean_ws(s):
    return re.sub(r"[ \t]+", " ", s).strip()


def fetch_mirror_day(date):
    """Return (record, None) or (None, reason)."""
    url = f"https://cn.govopendata.com/xinwenlianbo/{date.strftime('%Y%m%d')}/"
    r = session.get(url, timeout=25)
    if r.status_code != 200:
        return None, f"http {r.status_code}"
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")
    main = soup.find(id="main-content") or soup.body
    if main is None:
        return None, "no main content"
    items = []
    articles = main.find_all("article", class_="content-section")
    if articles:
        for art in articles:
            h2 = art.find("h2")
            body_div = art.find("div", class_="content-body")
            if h2 is None or body_div is None:
                continue
            paras = [clean_ws(pp.get_text()) for pp in body_div.find_all("p")]
            paras = [t for t in paras if t]
            items.append({"title": clean_ws(h2.get_text()),
                          "content": "\n".join(paras) if paras else clean_ws(body_div.get_text())})
    else:
        for h2 in main.find_all("h2"):
            title = clean_ws(h2.get_text())
            if not title:
                continue
            paras = []
            for sib in h2.find_next_siblings():
                if sib.name == "h2":
                    break
                if sib.name == "p":
                    t = clean_ws(sib.get_text())
                    if t:
                        paras.append(t)
                elif sib.name == "div":
                    t = clean_ws(sib.get_text())
                    if t:
                        paras.append(t)
            items.append({"title": title, "content": "\n".join(paras)})
    items = [i for i in items if i["content"]]
    if not items:
        return None, "no items parsed"
    text = "\n\n".join(f"\u3010{i['title']}\u3011\n{i['content']}" for i in items)
    return {"date": date.isoformat(),
            "title": f"\u300a\u65b0\u95fb\u8054\u64ad\u300b {date.isoformat()}",
            "text": text, "items": items,
            "source": "mirror:cn.govopendata.com"}, None


def run(dates, out_path, log_missing=True):
    done = load_jsonl(out_path)
    todo = [d for d in dates if d.isoformat() not in done]
    print(f"mirror: {len(todo)} days to fetch -> {out_path.name}", flush=True)
    fails = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, d in enumerate(todo, 1):
            time.sleep(SLEEP)
            try:
                rec, err = fetch_mirror_day(d)
            except requests.RequestException as e:
                rec, err = None, f"exc {e}"[:100]
            if rec is None:
                fails += 1
                print(f"[{i}/{len(todo)}] {d} FAIL {err} (consec={fails})",
                      flush=True)
                if log_missing:
                    with open(OUT / "_missing.log", "a", encoding="utf-8") as mf:
                        mf.write(f"{d}\tmirror: {err}\n")
                if fails >= 8:
                    print("ABORT: 8 consecutive failed days", flush=True)
                    return
                continue
            fails = 0
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"[{i}/{len(todo)}] {d} ok items={len(rec['items'])}",
                      flush=True)
    print("mirror: done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="")
    ap.add_argument("--out", default="_mirror.jsonl")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.dates:
        dates = [dt.date.fromisoformat(s) for s in args.dates.split(",")]
    else:
        have = set(load_jsonl(OUT / "_repos.jsonl"))
        have |= set(load_jsonl(OUT / "_mirror.jsonl"))
        dates = []
        d = START
        while d <= END:
            if d.isoformat() not in have:
                dates.append(d)
            d += dt.timedelta(days=1)
    run(dates, OUT / args.out)


if __name__ == "__main__":
    main()
