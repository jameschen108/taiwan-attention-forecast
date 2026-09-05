"""TWSE 除權除息計算結果表（PRD §4.1「除權息參考價 — 報酬還原的必要輸入」）。

端點支援日期區間且為**全市場**，因此 11 年只需約 45 次請求，遠優於逐檔抓取。
「除權息前收盤價」與「除權息參考價」的比值即為還原因子。
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; contact via repo)"}

_NUM = re.compile(r"[^\d.\-]")


def _to_float(value: str) -> float | None:
    s = _NUM.sub("", str(value or ""))
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v > 0 else None


def _roc_date(value: str) -> pd.Timestamp | None:
    """「109年01月16日」→ Timestamp。"""
    m = re.match(r"(\d{2,3})年(\d{1,2})月(\d{1,2})日", str(value).strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))
    return pd.Timestamp(year=y, month=mo, day=d)


def fetch_range(start: str, end: str, session: requests.Session,
                max_retries: int = 5) -> list[list]:
    delay = 5.0
    for attempt in range(max_retries):
        resp = session.get(URL, params={"startDate": start, "endDate": end,
                                        "response": "json"},
                           headers=HEADERS, timeout=60)
        if resp.status_code == 200:
            body = resp.json()
            if body.get("stat") == "OK":
                return body.get("data") or []
            if "無" in str(body.get("stat", "")):
                return []
        if attempt == max_retries - 1:
            raise RuntimeError(f"TWT49U {start}~{end}: HTTP {resp.status_code}")
        time.sleep(delay)
        delay = min(delay * 2, 120)
    return []


def collect(start_year: int, end_year: int, out_path: Path,
            raw_dir: Path | None = None) -> pd.DataFrame:
    session = requests.Session()
    rows: list[list] = []
    for year in range(start_year, end_year + 1):
        for q_start, q_end in (("0101", "0331"), ("0401", "0630"),
                               ("0701", "0930"), ("1001", "1231")):
            s, e = f"{year}{q_start}", f"{year}{q_end}"
            chunk = fetch_range(s, e, session)
            rows.extend(chunk)
            if raw_dir is not None:
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"twt49u_{s}_{e}.json").write_text(
                    json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
            print(f"  {s}~{e}: {len(chunk)} 筆", flush=True)
            time.sleep(random.uniform(1.5, 3.0))

    df = pd.DataFrame(rows, columns=[
        "date_roc", "ticker", "name", "before_price", "reference_price",
        "value", "kind", "limit_up", "limit_down", "open_base",
        "adj_ref_price", "detail", "decl_period", "decl_bvps", "decl_eps",
    ][:len(rows[0])] if rows else None)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "before_price",
                                     "after_price", "factor"])

    df["date"] = df["date_roc"].map(_roc_date)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["before_price"] = df["before_price"].map(_to_float)
    df["after_price"] = df["reference_price"].map(_to_float)
    df = df[df["ticker"].str.fullmatch(r"\d{4}")]
    df = df.dropna(subset=["date", "before_price", "after_price"])
    df["factor"] = df["after_price"] / df["before_price"]

    out = df[["ticker", "date", "before_price", "after_price", "factor",
              "kind"]].sort_values(["ticker", "date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"除權息事件 {len(out):,} 筆，{out['ticker'].nunique()} 檔，"
          f"{out['date'].min().date()} ~ {out['date'].max().date()}")
    return out


if __name__ == "__main__":
    collect(2014, 2025, Path("data/interim/ex_rights.csv"),
            Path("data/raw/twse_exrights"))
