"""Match raw Cnyes headlines to universe tickers → news_matched.parquet."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.universe.name_matching import build_matcher


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_raw_news(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for complete in sorted(raw_dir.glob("*/_complete.json")):
        win_dir = complete.parent
        for path in sorted(win_dir.glob("page_*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            items = (doc.get("items") or {}).get("data") or []
            for item in items:
                pub = item.get("publishAt")
                if pub is None:
                    continue
                title = item.get("title") or ""
                summary = _strip_html(item.get("summary") or "")
                rows.append({
                    "news_id": item.get("newsId"),
                    "title": title,
                    "summary": summary,
                    "publish_at": pd.to_datetime(int(pub), unit="s", utc=True).tz_convert(None),
                    "category": item.get("categoryName"),
                    "source": item.get("source"),
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["news_id"])
    return df.sort_values("publish_at").reset_index(drop=True)


def match_to_tickers(articles: pd.DataFrame, cfg_path: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    matcher = build_matcher(cfg)
    matched_rows: list[dict] = []
    for _, row in articles.iterrows():
        text = f"{row['title']}\n{row.get('summary', '')}"
        hits = matcher.match(row["title"], row.get("summary", ""))
        for ticker, mode in hits.items():
            matched_rows.append({
                "news_id": row["news_id"],
                "ticker": ticker,
                "publish_at": row["publish_at"],
                "title": row["title"],
                "match_mode": mode,
            })
    if not matched_rows:
        return pd.DataFrame(columns=["news_id", "ticker", "publish_at", "title", "match_mode"])
    out = pd.DataFrame(matched_rows)
    return out.sort_values(["publish_at", "ticker"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Match Cnyes news to tickers")
    parser.add_argument("--raw", default="data/raw/cnyes/headline")
    parser.add_argument("--universe-cfg", default="config/universe.yaml")
    parser.add_argument("--out", default="data/forecast/supplemental/news_matched.parquet")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    raw_dir = root / args.raw
    articles = load_raw_news(raw_dir)
    print(f"Loaded {len(articles):,} unique articles", flush=True)
    if articles.empty:
        print("No raw news found; run collect_cnyes_news first.", flush=True)
        return

    matched = match_to_tickers(articles, root / args.universe_cfg)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(out_path, index=False)
    summary = {
        "n_articles": int(len(articles)),
        "n_matched_rows": int(len(matched)),
        "n_tickers": int(matched["ticker"].nunique()) if not matched.empty else 0,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path.parent / "news_match_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
