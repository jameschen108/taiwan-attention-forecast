"""PTT 文章 → ticker-article 配對（PRD §3.4、§3.5）。

輸出 data/interim/ptt_matches.parquet：ticker, timestamp, category, is_reply,
match_mode, effort, window（日曆切法）, session（交易時段切法）, week。
同文多檔輸出多列。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from src.features.sessions import (
    assign_calendar_window, assign_session_window, attention_week,
)
from src.ptt.parse import effort_tier, iter_articles
from src.universe.name_matching import build_matcher


def build_matches(ptt_root: Path, universe_cfg: Path, trading_days_csv: Path,
                  out_path: Path, audit_dir: Path,
                  progress_every: int = 20000) -> pd.DataFrame:
    cfg = yaml.safe_load(universe_cfg.read_text(encoding="utf-8"))
    matcher = build_matcher(cfg)

    cal = pd.read_csv(trading_days_csv, parse_dates=["date"])
    trading_days = {d.date() for d in cal["date"]}
    trading_sorted = sorted(trading_days)

    rows: list[dict] = []
    excluded: list[dict] = []
    n_articles = 0
    cat_counts: dict[str, int] = {}
    n_pushes_total = 0

    for art in iter_articles(ptt_root):
        n_articles += 1
        cat_counts[art.category] = cat_counts.get(art.category, 0) + 1
        n_pushes_total += art.n_pushes

        ts = pd.Timestamp(art.timestamp, unit="s", tz="Asia/Taipei").tz_localize(None)
        week = attention_week(ts, trading_sorted)
        if week is None:
            # 期末收盤後無法指派下一交易日者，明確列入排除清單（PRD §3.1 終點規則）
            excluded.append({"article_id": art.article_id, "timestamp": art.timestamp,
                             "reason": "no_next_trading_day"})
            continue

        found = matcher.match(art.title, art.body)
        if not found:
            continue
        effort = effort_tier(art.category, art.is_reply)
        window = assign_calendar_window(ts)
        session = assign_session_window(ts, trading_days)
        for ticker, mode in found.items():
            rows.append({
                "ticker": ticker,
                "article_id": art.article_id,
                "timestamp": ts,
                "week": week,
                "category": art.category,
                "is_reply": art.is_reply,
                "match_mode": mode,
                "effort": effort,
                "window": window,
                "session": session,
                "n_pushes": art.n_pushes,
            })
        if progress_every and n_articles % progress_every == 0:
            print(f"  已處理 {n_articles:,} 篇，累計配對 {len(rows):,}", flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(excluded).to_csv(audit_dir / "ptt_excluded_articles.csv", index=False)
    pd.DataFrame(
        sorted(cat_counts.items(), key=lambda kv: -kv[1]), columns=["category", "n"]
    ).to_csv(audit_dir / "ptt_category_distribution.csv", index=False)

    matched_articles = df["article_id"].nunique() if not df.empty else 0
    print(f"文章 {n_articles:,}；有配對 {matched_articles:,} "
          f"({matched_articles / max(n_articles, 1):.1%})；"
          f"配對列 {len(df):,}；排除 {len(excluded)}；推文總計 {n_pushes_total:,}（無時戳，不使用）")
    return df


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ptt-root", default="data/pttweb")
    p.add_argument("--universe", default="config/universe.yaml")
    p.add_argument("--trading-days", default="data/interim/trading_days.csv")
    p.add_argument("--out", default="data/interim/ptt_matches.parquet")
    p.add_argument("--audit", default="audit")
    a = p.parse_args()
    build_matches(Path(a.ptt_root), Path(a.universe), Path(a.trading_days),
                  Path(a.out), Path(a.audit))


if __name__ == "__main__":
    main()
