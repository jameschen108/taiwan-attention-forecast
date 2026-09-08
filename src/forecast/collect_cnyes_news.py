"""Collect Cnyes headline news (timestamped) for P4 news features."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

API = "https://api.cnyes.com/media/api/v1/newslist/category/headline"
TZ = ZoneInfo("Asia/Taipei")


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _window_paths(out_dir: Path, start: datetime, end: datetime) -> Path:
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    return out_dir / tag


MAX_PAGE = 30


def fetch_window(session: requests.Session, start: datetime, end: datetime, out_dir: Path,
                 pause: tuple[float, float] = (0.25, 0.5)) -> int:
    win_dir = _window_paths(out_dir, start, end)
    win_dir.mkdir(parents=True, exist_ok=True)
    done_marker = win_dir / "_complete.json"
    if done_marker.exists():
        meta = json.loads(done_marker.read_text(encoding="utf-8"))
        return int(meta.get("n_articles", 0))

    start_at = _ts(start)
    end_at = _ts(end)
    page = 1
    total_saved = 0
    last_page = 1

    while page <= MAX_PAGE:
        out_path = win_dir / f"page_{page:04d}.json"
        if out_path.exists():
            doc = json.loads(out_path.read_text(encoding="utf-8"))
            items = doc.get("items") or {}
            last_page = int(items.get("last_page") or page)
            total_saved += len(items.get("data") or [])
            if page >= min(last_page, MAX_PAGE):
                break
            page += 1
            continue

        params = {"startAt": start_at, "endAt": end_at, "limit": 30, "page": page}
        resp = session.get(API, params=params, timeout=60)
        if resp.status_code == 422 and page > MAX_PAGE:
            break
        if resp.status_code != 200:
            raise RuntimeError(f"Cnyes HTTP {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        items = body.get("items") or {}
        rows = items.get("data") or []
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        out_path.write_bytes(payload)
        (out_path.with_suffix(".sha256")).write_text(
            hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8",
        )
        total_saved += len(rows)
        last_page = int(items.get("last_page") or page)
        if page >= min(last_page, MAX_PAGE) or not rows:
            break
        page += 1
        time.sleep(random.uniform(*pause))

    truncated = last_page > MAX_PAGE
    if truncated:
        # Split dense window and fetch remainder recursively.
        mid = start + (end - start) / 2
        if (end - start).total_seconds() > 3600:
            total_saved += fetch_window(session, start, mid, out_dir, pause)
            total_saved += fetch_window(session, mid + timedelta(seconds=1), end, out_dir, pause)
            truncated = False

    done_marker.write_text(json.dumps({
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_articles": total_saved,
        "last_page": last_page,
        "truncated": truncated,
        "fetched_on": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return total_saved


def iter_windows(start: datetime, end: datetime, days: int = 7):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect Cnyes headline news")
    parser.add_argument("--start", default="2015-05-01")
    parser.add_argument("--end", default="2025-03-31")
    parser.add_argument("--window-days", type=int, default=1)
    parser.add_argument("--out", default="data/raw/cnyes/headline")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(args.start).replace(tzinfo=TZ)
    end = datetime.fromisoformat(args.end).replace(hour=23, minute=59, second=59, tzinfo=TZ)
    session = requests.Session()
    grand = 0
    windows = list(iter_windows(start, end, days=args.window_days))
    print(f"Cnyes: {len(windows)} windows {args.start} ~ {args.end}", flush=True)

    for i, (ws, we) in enumerate(windows, 1):
        n = fetch_window(session, ws, we, out_dir)
        grand += n
        if i % 20 == 0 or i == len(windows):
            print(f"  [{i}/{len(windows)}] {ws.date()}~{we.date()} cumulative {grand:,}", flush=True)


if __name__ == "__main__":
    main()
