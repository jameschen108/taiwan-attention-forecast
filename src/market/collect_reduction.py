"""TWSE 股票減資恢復買賣參考價格（TWTAUU）。

**為什麼需要這張表**：除權除息計算結果表（TWT49U）只涵蓋除權與除息，**不含減資**。
減資會讓股本縮減、股價機械性跳升（長榮 2603 於 2022-09-19 單日「上漲」109%），
若不還原，該檔的報酬序列在減資日會出現一筆完全虛假的極端值。

實測：只用 TWT49U 還原時，主樣本內仍有 **127 筆 |日報酬| > 11%** 的觀測（台股漲跌幅
上限為 10%，超過即定義上不可能是價格變動），涉及 59 檔。這是由既有 Yahoo 快取的
交叉比對揭露的——Yahoo 的收盤價有還原股本變動但未還原股利，正好與本專案互補。

還原因子 = 恢復買賣參考價 / 停止買賣前收盤價格。
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU"
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
    """「111/01/06」→ Timestamp。"""
    m = re.match(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", str(value).strip())
    if not m:
        return None
    return pd.Timestamp(year=int(m.group(1)) + 1911, month=int(m.group(2)),
                        day=int(m.group(3)))


def fetch_range(start: str, end: str, session: requests.Session,
                max_retries: int = 5) -> list[list]:
    delay = 10.0
    for attempt in range(max_retries):
        try:
            resp = session.get(URL, params={"startDate": start, "endDate": end,
                                            "response": "json"},
                               headers=HEADERS, timeout=90)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"TWTAUU {start}~{end}: {exc}") from exc
            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue
        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                body = {}
            if body.get("stat") == "OK":
                return body.get("data") or []
            return []
        if attempt == max_retries - 1:
            raise RuntimeError(f"TWTAUU {start}~{end}: HTTP {resp.status_code}")
        time.sleep(delay)
        delay = min(delay * 2, 300)
    return []


def collect(start_year: int, end_year: int, out_path: Path,
            raw_dir: Path | None = None) -> pd.DataFrame:
    session = requests.Session()
    rows: list[list] = []
    for year in range(start_year, end_year + 1):
        s, e = f"{year}0101", f"{year}1231"
        cache = (raw_dir / f"twtauu_{year}.json") if raw_dir else None
        if cache is not None and cache.exists():
            chunk = json.loads(cache.read_text(encoding="utf-8"))
        else:
            chunk = fetch_range(s, e, session)
            if cache is not None:
                raw_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(chunk, ensure_ascii=False),
                                 encoding="utf-8")
            time.sleep(random.uniform(2.0, 4.0))
        rows.extend(chunk)
        print(f"  {year}: {len(chunk)} 筆", flush=True)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "before_price",
                                     "after_price", "factor", "reason"])

    df = pd.DataFrame(rows).iloc[:, :10]
    df.columns = ["date_roc", "ticker", "name", "before_price", "after_price",
                  "limit_up", "limit_down", "open_base", "ex_right_ref",
                  "reason"][:df.shape[1]]
    df["date"] = df["date_roc"].map(_roc_date)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["before_price"] = df["before_price"].map(_to_float)
    df["after_price"] = df["after_price"].map(_to_float)
    df = df[df["ticker"].str.fullmatch(r"\d{4}")]
    df = df.dropna(subset=["date", "before_price", "after_price"])
    # 減資使股價上調，因子 > 1（與除權息的 < 1 相反）
    df["factor"] = df["after_price"] / df["before_price"]

    out = df[["ticker", "date", "before_price", "after_price", "factor",
              "reason"]].sort_values(["ticker", "date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"減資事件 {len(out):,} 筆，{out['ticker'].nunique()} 檔，"
          f"{out['date'].min().date()} ~ {out['date'].max().date()}")
    return out


if __name__ == "__main__":
    collect(2011, 2026, Path("data/interim/capital_reductions.csv"),
            Path("data/raw/twse_reduction"))
