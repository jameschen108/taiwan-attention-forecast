"""TWSE 股權分散：發行股數與外資及陸資持股比率（PRD §3.8、§4.1）。

MI_QFIIS 是**全市場單日**表，因此每個抽樣日只需一次請求。發行股數與外資持股比率
都是低頻變數（月內幾乎不變），週抽樣即足以支撐市值、周轉率與 H6 的調節變數。

命名紀律（PRD §11）：本欄位只能稱為「外資及陸資持股比率 proxy」，不得寫成
「機構持股比例」——它並非全體機構投資人。
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; contact via repo)"}
_NUM = re.compile(r"[^\d.\-]")


def _num(value) -> float | None:
    s = _NUM.sub("", str(value or ""))
    try:
        return float(s)
    except ValueError:
        return None


def fetch_day(date: pd.Timestamp, session: requests.Session,
              max_retries: int = 4) -> list[list]:
    delay = 5.0
    for attempt in range(max_retries):
        resp = session.get(URL, params={"date": date.strftime("%Y%m%d"),
                                        "selectType": "ALLBUT0999",
                                        "response": "json"},
                           headers=HEADERS, timeout=60)
        if resp.status_code == 200:
            body = resp.json()
            if body.get("stat") == "OK":
                return body.get("data") or []
            return []          # 非交易日回傳「很抱歉，沒有符合條件的資料」
        if attempt == max_retries - 1:
            return []
        time.sleep(delay)
        delay = min(delay * 2, 120)
    return []


def collect(trading_days_csv: Path, out_path: Path, tickers: set[str],
            every_n_days: int = 5, raw_dir: Path | None = None) -> pd.DataFrame:
    """對交易日曆每 `every_n_days` 個交易日抽樣一次（預設約每週一次）。"""
    cal = pd.read_csv(trading_days_csv, parse_dates=["date"])
    sample_days = cal["date"].iloc[::every_n_days]
    session = requests.Session()

    rows = []
    for i, day in enumerate(sample_days, 1):
        cached = None
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            cache = raw_dir / f"qfiis_{day:%Y%m%d}.json"
            if cache.exists():
                cached = json.loads(cache.read_text(encoding="utf-8"))
        data = cached if cached is not None else fetch_day(day, session)
        if cached is None and raw_dir is not None:
            (raw_dir / f"qfiis_{day:%Y%m%d}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8")
            time.sleep(random.uniform(1.2, 2.4))

        for r in data:
            ticker = str(r[0]).strip()
            if ticker not in tickers:
                continue
            rows.append({
                "ticker": ticker, "date": day,
                "shares_outstanding": _num(r[3]),
                "foreign_holding_pct": _num(r[7]),
            })
        if i % 50 == 0 or i == len(sample_days):
            print(f"  {i}/{len(sample_days)} 抽樣日，累計 {len(rows):,} 列", flush=True)

    df = pd.DataFrame(rows).dropna(subset=["shares_outstanding"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"股權分散 {len(df):,} 列，{df['ticker'].nunique()} 檔，"
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


if __name__ == "__main__":
    uni = pd.read_csv("data/external/universe.csv", dtype={"ticker": str})
    collect(Path("data/interim/trading_days.csv"),
            Path("data/interim/shareholding.csv"),
            set(uni["ticker"]), every_n_days=5,
            raw_dir=Path("data/raw/twse_qfiis"))
