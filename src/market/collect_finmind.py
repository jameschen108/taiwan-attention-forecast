"""從 FinMind 抓取 267 檔的日成交、三大法人、除權息、股權分散。

PRD §4.3：原始回應原封落地，附抓取日期與 SHA-256；斷點續傳；速率限制退避；
缺檔明確報錯而非當成零。本模組只寫入 data/raw/，不做任何轉換。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import requests

API = "https://api.finmindtrade.com/api/v4/data"

# dataset 名稱 → 落地子目錄
DATASETS = {
    "price": "TaiwanStockPrice",
    "inst": "TaiwanStockInstitutionalInvestorsBuySell",
    "dividend": "TaiwanStockDividendResult",
    "shareholding": "TaiwanStockShareholding",
}

# 價格與法人需向前多取，供 25 週動能與 8 週回顧窗使用（PRD §3.8）
START_DATE = "2014-01-01"
END_DATE = "2025-03-31"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(dataset: str, ticker: str, session: requests.Session,
          start: str = START_DATE, end: str = END_DATE,
          max_retries: int = 8) -> dict:
    """單次抓取，對 402/429 做指數退避。回傳原始 JSON。"""
    params = {
        "dataset": dataset,
        "data_id": ticker,
        "start_date": start,
        "end_date": end,
    }
    delay = 30.0
    for attempt in range(max_retries):
        try:
            resp = session.get(API, params=params, timeout=90)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            print(f"    網路錯誤 {exc}；{delay:.0f}s 後重試")
            time.sleep(delay)
            delay = min(delay * 2, 900)
            continue

        if resp.status_code in (402, 429):
            # FinMind 免費額度用盡，等待額度重置
            print(f"    {resp.status_code} 額度限制；{delay:.0f}s 後重試")
            time.sleep(delay)
            delay = min(delay * 2, 900)
            continue
        if resp.status_code != 200:
            if attempt == max_retries - 1:
                resp.raise_for_status()
            time.sleep(delay)
            delay = min(delay * 2, 900)
            continue

        body = resp.json()
        if body.get("status") != 200:
            if attempt == max_retries - 1:
                raise RuntimeError(f"{dataset}/{ticker}: {body.get('msg')}")
            time.sleep(delay)
            delay = min(delay * 2, 900)
            continue
        return body

    raise RuntimeError(f"{dataset}/{ticker}: 重試次數耗盡")


def collect(tickers: list[str], out_root: Path, kinds: list[str] | None = None,
            pause: tuple[float, float] = (1.0, 2.5)) -> None:
    """對每檔股票抓取指定資料集，已存在者跳過（斷點續傳）。"""
    kinds = kinds or list(DATASETS)
    session = requests.Session()
    fetched_on = time.strftime("%Y-%m-%d")

    for kind in kinds:
        dataset = DATASETS[kind]
        out_dir = out_root / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [t for t in tickers if not (out_dir / f"{t}.json").exists()]
        print(f"[{kind}] {len(todo)}/{len(tickers)} 待抓")

        for i, ticker in enumerate(todo, 1):
            body = fetch(dataset, ticker, session)
            rows = body.get("data") or []
            payload = json.dumps(
                {
                    "dataset": dataset,
                    "data_id": ticker,
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "fetched_on": fetched_on,
                    "n_rows": len(rows),
                    "data": rows,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            path = out_dir / f"{ticker}.json"
            path.write_bytes(payload)
            (out_dir / f"{ticker}.sha256").write_text(_sha256(payload) + "\n")
            if i % 20 == 0 or i == len(todo):
                print(f"  [{kind}] {i}/{len(todo)} 最新 {ticker} ({len(rows)} 列)", flush=True)
            time.sleep(random.uniform(*pause))


def main() -> None:
    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="data/universe_267.csv")
    parser.add_argument("--out", default="data/raw/finmind")
    parser.add_argument("--kinds", nargs="*", default=None, choices=list(DATASETS))
    args = parser.parse_args()

    with open(args.universe, encoding="utf-8-sig") as fh:
        tickers = [row["ticker"].strip() for row in csv.DictReader(fh)]
    collect(tickers, Path(args.out), args.kinds)


if __name__ == "__main__":
    main()
