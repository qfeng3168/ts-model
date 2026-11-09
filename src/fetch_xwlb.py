#!/usr/bin/env python3
"""Fetch CCTV Xinwen Lianbo full-text archive, 2015-01-01..2026-11-07.

Sources
-------
- Bulk: two GitHub archives transparently scraped from tv.cctv.com
  (xiaoyu2er/xwlb for 2020-2022, DuckBurnIncense/xin-wen-lian-bo for 2022-2026).
- Gap fill + verification: CCTV official pages, URL patterns per AKShare
  news_cctv (akshare/alternative/alternative_news_cctv.py):
    date >= 20160203: https://tv.cctv.com/lm/xwlb/day/{YYYYMMDD}.shtml
    date <  20160203: https://cctv.cntv.cn/lm/xinwenlianbo/{YYYYMMDD}.shtml
  (20160203 itself routed to the modern pattern; AKShare has no branch for it.)

Politeness: single thread, >=0.35 s between requests, abort after 8
consecutive failed days (possible block).

Outputs under data/xwlb/: _repos.jsonl, _official.jsonl (resumable),
_verify.json, xwlb.jsonl (final), coverage.json, _missing.log

Usage: fetch_xwlb.py {repos|verify|merge|scrape [--start D] [--end D]}
"""

import argparse
import datetime as dt
import difflib
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "xwlb"
REPO_XIAOYU = Path("/tmp/xw2/news")
REPO_DUCK = Path("/tmp/xw1/news")
START = dt.date(2015, 1, 1)
END = dt.date(2026, 11, 7)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
SLEEP = 0.35
session = requests.Session()
session.headers.update(HEADERS)


def daterange(a, b):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def clean_ws(s):
    return re.sub(r"[ \t]+", " ", s).strip()


# ---------------------------------------------------------------- repos
def parse_duck(path):
    raw = path.read_text(encoding="utf-8")
    items = []
    m = re.search(r"## \u8be6\u7ec6\u65b0\u95fb\s*\n(.*)\Z", raw, re.S)
    if m:
        for part in re.split(r"\n### ", m.group(1)):
            part = part.strip()
            if not part:
                continue
            title, _, body = part.partition("\n")
            title = title.removeprefix("### ").strip()
            body = re.sub(r"\*\*\u592e\u89c6\u7f51\u6d88\u606f\*\*\uff08\u65b0\u95fb\u8054\u64ad\uff09\uff1a", "", body).strip()
            items.append({"title": clean_ws(title), "content": clean_ws(body)})
    text = re.sub(r"^# \d{8}\.md\s*\n", "", raw)
    text = re.sub(r"^# \u300a\u65b0\u95fb\u8054\u64ad\u300b \(\d{8}\)\s*\n", "", text).strip()
    return items, text


def parse_xiaoyu(path):
    raw = path.read_text(encoding="utf-8")
    items = []
    for part in re.split(r"\n## ", raw):
        part = part.strip()
        if not part or part.startswith("# \u65b0\u95fb\u8054\u64ad"):
            continue
        title, _, body = part.partition("\n")
        title = title.strip()
        if title.startswith("\u300a\u65b0\u95fb\u8054\u64ad\u300b"):
            continue
        body = re.sub(r"\n?\*\[(.*?)\]\(https?://[^)]*\)\*", "", body)
        body = re.sub(r"\*\*\u592e\u89c6\u7f51\u6d88\u606f\*\*\uff08\u65b0\u95fb\u8054\u64ad\uff09\uff1a", "", body).strip()
        if title and body:
            items.append({"title": clean_ws(title), "content": clean_ws(body)})
    text = re.sub(r"^# \u65b0\u95fb\u8054\u64ad \d{4}-\d{2}-\d{2}\s*\n", "", raw)
    text = re.sub(r"\n?\*\[(.*?)\]\(https?://[^)]*\)\*", "", text).strip()
    return items, text


def cmd_repos():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for path in sorted(REPO_XIAOYU.glob("*.md")):
        m = re.match(r"(\d{4}-\d{2}-\d{2})\.md", path.name)
        if not m:
            continue
        date = m.group(1)
        items, text = parse_xiaoyu(path)
        records[date] = {
            "date": date,
            "title": f"\u300a\u65b0\u95fb\u8054\u64ad\u300b {date}",
            "text": text,
            "items": items,
            "source": "github:xiaoyu2er/xwlb",
        }
    for path in sorted(REPO_DUCK.glob("*.md")):
        m = re.match(r"(\d{8})\.md", path.name)
        if not m:
            continue
        d = dt.datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()
        if d in records:
            continue
        items, text = parse_duck(path)
        records[d] = {
            "date": d,
            "title": f"\u300a\u65b0\u95fb\u8054\u64ad\u300b {d}",
            "text": text,
            "items": items,
            "source": "github:DuckBurnIncense/xin-wen-lian-bo",
        }
    with open(OUT / "_repos.jsonl", "w", encoding="utf-8") as f:
        for date in sorted(records):
            f.write(json.dumps(records[date], ensure_ascii=False) + "\n")
    print(f"repos: {len(records)} days ({min(records)} .. {max(records)})")


# ------------------------------------------------------------- official
def _get(url):
    r = session.get(url, timeout=25)
    r.encoding = "utf-8"
    return r


def _clean_title(t):
    return t.strip().removeprefix("[\u89c6\u9891]").strip().replace("\n", " ")


_PREFIXES = ("\u592e\u89c6\u7f51\u6d88\u606f(\u65b0\u95fb\u8054\u64ad)\uff1a",
             "\u592e\u89c6\u7f51\u6d88\u606f\uff08\u65b0\u95fb\u8054\u64ad\uff09\uff1a",
             "(\u65b0\u95fb\u8054\u64ad)\uff1a",
             "\uff08\u65b0\u95fb\u8054\u64ad\uff09\uff1a")


def _clean_body(c):
    c = c.strip()
    for p in _PREFIXES:
        if c.startswith(p):
            c = c[len(p):]
            break
    return clean_ws(c)


_STUB_RE = re.compile(r"window\.location\.href='([^']+)'")


def _get_article(url):
    """Fetch an article page, following the JS-redirect stub that old
    news.cntv.cn pages serve (they point at the same path on tv.cctv.com)."""
    r = _get(url)
    if r.status_code == 200 and len(r.text) < 1000:
        m = _STUB_RE.search(r.text)
        if m:
            r = _get(m.group(1))
    return r


def fetch_official_day(date):
    """Return (items, None) or (None, reason)."""
    ds = date.strftime("%Y%m%d")
    if date >= dt.date(2016, 2, 3):
        url = f"https://tv.cctv.com/lm/xwlb/day/{ds}.shtml"
        r = _get(url)
        if r.status_code != 200:
            return None, f"list http {r.status_code}"
        soup = BeautifulSoup(r.text, "lxml")
        links = []
        for li in soup.find_all("li")[1:]:
            a = li.find("a")
            if a and a.get("href"):
                links.append(a["href"])
    else:
        url = f"https://cctv.cntv.cn/lm/xinwenlianbo/{ds}.shtml"
        r = _get(url)
        if r.status_code != 200:
            return None, f"list http {r.status_code}"
        soup = BeautifulSoup(r.text, "lxml")
        div = soup.find("div", attrs={"id": "contentELMT1368521805488378"})
        if div is None:
            return None, "list div missing"
        links = []
        for li in div.find_all("li")[1:]:
            a = li.find("a")
            if a and a.get("href"):
                links.append(a["href"])
    if not links:
        return None, "no item links"
    items = []
    for link in links:
        time.sleep(SLEEP)
        if link.startswith("//"):
            link = "https:" + link
        try:
            rp = _get_article(link)
        except requests.RequestException as e:
            items.append({"title": "", "content": "", "error": str(e)[:80]})
            continue
        if rp.status_code != 200:
            items.append({"title": "", "content": "", "error": f"http {rp.status_code}"})
            continue
        sp = BeautifulSoup(rp.text, "lxml")
        t = sp.find("h3")
        if t is None:
            t = sp.find("div", class_="tit")
        b = sp.find("div", class_="cnt_bd")
        if b is None:
            b = sp.find("div", class_="content_area")
        if t is None or b is None:
            items.append({"title": "", "content": "", "error": "parse miss"})
            continue
        items.append({"title": _clean_title(t.get_text()),
                      "content": _clean_body(b.get_text())})
    ok = [i for i in items if i["content"]]
    if not ok:
        return None, "all items failed"
    return items, None


def _load_jsonl(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["date"]] = rec
    return out


def cmd_scrape(a, b):
    OUT.mkdir(parents=True, exist_ok=True)
    have = set(_load_jsonl(OUT / "_repos.jsonl"))
    done = _load_jsonl(OUT / "_official.jsonl")
    have |= set(done)
    todo = [d for d in daterange(a, b) if d.isoformat() not in have]
    print(f"scrape: {len(todo)} days to fetch ({a}..{b})", flush=True)
    fails = 0
    with open(OUT / "_official.jsonl", "a", encoding="utf-8") as f:
        for i, d in enumerate(todo, 1):
            time.sleep(SLEEP)
            try:
                items, err = fetch_official_day(d)
            except requests.RequestException as e:
                items, err = None, f"list exc {e}"[:100]
            if items is None:
                fails += 1
                print(f"[{i}/{len(todo)}] {d} FAIL {err} (consec={fails})", flush=True)
                with open(OUT / "_missing.log", "a", encoding="utf-8") as mf:
                    mf.write(f"{d}\t{err}\n")
                if fails >= 8:
                    print("ABORT: 8 consecutive failed days", flush=True)
                    return
                continue
            fails = 0
            text = "\n\n".join(
                f"\u3010{it['title']}\u3011\n{it['content']}"
                for it in items if it["content"])
            rec = {"date": d.isoformat(),
                   "title": f"\u300a\u65b0\u95fb\u8054\u64ad\u300b {d.isoformat()}",
                   "text": text, "items": items,
                   "source": "cctv-official"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"[{i}/{len(todo)}] {d} ok items={len(items)}", flush=True)
    print("scrape: done", flush=True)


# -------------------------------------------------------------- verify
def _norm(s):
    return re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)


def cmd_verify():
    repos = _load_jsonl(OUT / "_repos.jsonl")
    samples = ["2020-03-15", "2021-06-01", "2022-01-28", "2022-10-16",
               "2023-02-28", "2024-08-06", "2025-05-01", "2026-11-06"]
    results = []
    for s in samples:
        d = dt.date.fromisoformat(s)
        rec = repos.get(s)
        if rec is None:
            results.append({"date": s, "note": "not in repos"})
            continue
        items, err = fetch_official_day(d)
        time.sleep(1)
        if items is None:
            results.append({"date": s, "note": f"official fetch failed: {err}"})
            continue
        off = [i for i in items if i["content"]]
        repo_text = _norm(rec["text"])
        hit = sum(1 for i in off if _norm(i["title"]) and _norm(i["title"]) in repo_text)
        repo_items = {_norm(i["title"]): _norm(i["content"]) for i in rec["items"]}
        sims = []
        for it in off:
            rt = repo_items.get(_norm(it["title"]))
            if rt:
                sims.append(difflib.SequenceMatcher(
                    None, _norm(it["content"])[:3000], rt[:3000]).ratio())
        results.append({
            "date": s, "source": rec["source"],
            "official_items": len(off),
            "titles_found_in_repo_text": hit,
            "matched_items": len(sims),
            "mean_text_similarity": round(sum(sims) / len(sims), 4) if sims else None,
        })
        print(results[-1], flush=True)
    (OUT / "_verify.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------- merge
def cmd_merge():
    merged = _load_jsonl(OUT / "_repos.jsonl")
    for date, rec in _load_jsonl(OUT / "_official.jsonl").items():
        merged.setdefault(date, rec)
    with open(OUT / "xwlb.jsonl", "w", encoding="utf-8") as f:
        for date in sorted(merged):
            if START.isoformat() <= date <= END.isoformat():
                f.write(json.dumps(merged[date], ensure_ascii=False) + "\n")
    per_year, sources = {}, {}
    in_range = {d: r for d, r in merged.items()
                if START.isoformat() <= d <= END.isoformat()}
    for date, rec in in_range.items():
        per_year[date[:4]] = per_year.get(date[:4], 0) + 1
        sources[rec["source"]] = sources.get(rec["source"], 0) + 1
    missing = [d.isoformat() for d in daterange(START, END)
               if d.isoformat() not in in_range]
    cov = {"range": [START.isoformat(), END.isoformat()],
           "total_days_in_range": (END - START).days + 1,
           "records": len(in_range),
           "per_year": dict(sorted(per_year.items())),
           "by_source": sources,
           "missing_count": len(missing),
           "missing_dates": missing}
    (OUT / "coverage.json").write_text(
        json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: cov[k] for k in
                      ("records", "per_year", "by_source", "missing_count")},
                     ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("repos")
    sub.add_parser("verify")
    sub.add_parser("merge")
    sp = sub.add_parser("scrape")
    sp.add_argument("--start", default=START.isoformat())
    sp.add_argument("--end", default=END.isoformat())
    args = ap.parse_args()
    if args.cmd == "repos":
        cmd_repos()
    elif args.cmd == "verify":
        cmd_verify()
    elif args.cmd == "merge":
        cmd_merge()
    elif args.cmd == "scrape":
        cmd_scrape(dt.date.fromisoformat(args.start),
                   dt.date.fromisoformat(args.end))


if __name__ == "__main__":
    main()
