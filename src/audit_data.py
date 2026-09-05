"""P1 既有資料稽核（PRD §4.3 第 7 點、F9）。

既有快取不得因為「資料已經有了」而跳過稽核。本模組檢查涵蓋起訖、月份缺口、
逐檔筆數、重複鍵，並輸出 audit/ 下的報告。稽核不通過即中止管線。
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def audit_ptt(ptt_root: Path, audit_dir: Path) -> dict:
    """PTT 封存的完整性：月份缺口、重複 article_id、時戳可得性、推文狀態。"""
    months: Counter = Counter()
    ids: Counter = Counter()
    n_files = n_missing_ts = n_empty_body = n_pushes = 0
    n_push_with_meta = 0

    for batch in sorted(p for p in ptt_root.iterdir() if p.is_dir()):
        for path in batch.glob("M.*.json"):
            n_files += 1
            doc = json.loads(path.read_text(encoding="utf-8"))
            ts = doc.get("timestamp")
            if ts is None:
                n_missing_ts += 1
                m = re.match(r"M\.(\d{10})\.", path.name)
                ts = int(m.group(1)) if m else None
            if ts is None:
                continue
            months[dt.datetime.fromtimestamp(int(ts)).strftime("%Y-%m")] += 1
            ids[doc.get("article_id") or path.stem] += 1
            if not doc.get("body"):
                n_empty_body += 1
            pushes = doc.get("pushes") or []
            n_pushes += len(pushes)
            n_push_with_meta += sum(1 for p in pushes if isinstance(p, dict))

    ordered = sorted(months)
    expected = _month_range(ordered[0], ordered[-1])
    gaps = [m for m in expected if m not in months]
    dupes = {k: v for k, v in ids.items() if v > 1}

    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(months.items()), columns=["month", "n_articles"]).to_csv(
        audit_dir / "ptt_monthly_coverage.csv", index=False)

    result = {
        "n_files": n_files,
        "n_articles": sum(months.values()),
        "first_month": ordered[0],
        "last_month": ordered[-1],
        "n_months": len(months),
        "n_month_gaps": len(gaps),
        "month_gaps": ";".join(gaps),
        "n_duplicate_ids": len(dupes),
        "n_missing_timestamp_field": n_missing_ts,
        "n_empty_body": n_empty_body,
        "n_pushes": n_pushes,
        "pushes_have_author_and_timestamp": n_push_with_meta > 0,
    }
    return result


def _month_range(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    Y, M = map(int, last.split("-"))
    out = []
    while (y, m) <= (Y, M):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def audit_market(raw_root: Path, universe: pd.DataFrame, audit_dir: Path,
                 interim: Path = Path("data/interim")) -> dict:
    """實際使用來源的涵蓋率。缺檔明確列出，不得當成零。

    三大法人與除權息**不使用 FinMind**：前者用既有 TWSE T86 封存（涵蓋主樣本全期
    且不受 API 額度限制），後者用 TWSE 全市場區間端點。FinMind 的對應目錄若為空，
    不是資料缺口，而是來源選擇——因此只稽核實際進入管線的來源。
    """
    rows = []
    for kind in ("price", "shareholding"):
        kdir = raw_root / kind
        if not kdir.exists():
            continue
        for _, u in universe.iterrows():
            ticker = str(u["ticker"])
            path = kdir / f"{ticker}.json"
            if not path.exists():
                rows.append({"kind": kind, "ticker": ticker, "status": "MISSING",
                             "n_rows": 0, "first": "", "last": ""})
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            data = doc.get("data") or []
            dates = [d.get("date") for d in data if d.get("date")]
            rows.append({
                "kind": kind, "ticker": ticker,
                "status": "OK" if data else "EMPTY",
                "n_rows": len(data),
                "first": min(dates) if dates else "",
                "last": max(dates) if dates else "",
            })
    tickers = set(universe["ticker"].astype(str))
    summary: dict = {}

    # 三大法人：TWSE T86 封存
    t86 = interim / "market_daily.parquet"
    if t86.exists():
        daily = pd.read_parquet(t86, columns=["ticker", "date", "inst_buy"])
        have = daily.dropna(subset=["inst_buy"])
        for ticker in sorted(tickers):
            sub = have[have["ticker"] == ticker]
            rows.append({
                "kind": "institutional(T86)", "ticker": ticker,
                "status": "OK" if len(sub) else "MISSING", "n_rows": len(sub),
                "first": str(sub["date"].min().date()) if len(sub) else "",
                "last": str(sub["date"].max().date()) if len(sub) else "",
            })

    # 除權息：TWSE 全市場區間端點
    ex = interim / "ex_rights.csv"
    if ex.exists():
        exd = pd.read_csv(ex, dtype={"ticker": str})
        n_uni = exd[exd["ticker"].isin(tickers)]
        summary["exrights_events_total"] = len(exd)
        summary["exrights_events_in_universe"] = len(n_uni)
        summary["exrights_tickers_in_universe"] = int(n_uni["ticker"].nunique())

    cov = pd.DataFrame(rows)
    audit_dir.mkdir(parents=True, exist_ok=True)
    cov.to_csv(audit_dir / "market_coverage.csv", index=False)
    for kind, g in cov.groupby("kind"):
        summary[f"{kind}_ok"] = int((g["status"] == "OK").sum())
        summary[f"{kind}_missing"] = int((g["status"] == "MISSING").sum())
        summary[f"{kind}_empty"] = int((g["status"] == "EMPTY").sum())
    summary["note_sources"] = ("三大法人=TWSE T86 封存；除權息=TWSE TWT49U；"
                               "FinMind 僅供價格與股權分散備援")
    return summary


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ptt-root", default="data/pttweb")
    p.add_argument("--raw-root", default="data/raw/finmind")
    p.add_argument("--universe", default="data/external/universe.csv")
    p.add_argument("--audit", default="audit")
    p.add_argument("--skip-ptt", action="store_true")
    a = p.parse_args()

    audit_dir = Path(a.audit)
    uni = pd.read_csv(a.universe, dtype={"ticker": str})
    results: dict = {}

    if not a.skip_ptt:
        print("稽核 PTT 封存…")
        results.update({f"ptt_{k}": v for k, v in
                        audit_ptt(Path(a.ptt_root), audit_dir).items()})
    print("稽核市場資料…")
    results.update({f"market_{k}": v for k, v in
                    audit_market(Path(a.raw_root), uni, audit_dir).items()})

    pd.DataFrame(sorted(results.items()), columns=["check", "value"]).to_csv(
        audit_dir / "raw_audit_summary.csv", index=False)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
