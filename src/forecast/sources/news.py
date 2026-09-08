"""News count features from timestamped headlines (Cnyes or compatible parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forecast.config import load_forecast_config
from src.forecast.time_contract import as_of_from_feature_week
from src.features.sessions import week_of


def _load_news(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "publish_at" not in df.columns and "publishAt" in df.columns:
        df["publish_at"] = pd.to_datetime(df["publishAt"], unit="ms", errors="coerce")
    else:
        df["publish_at"] = pd.to_datetime(df["publish_at"], errors="coerce")
    if "ticker" not in df.columns:
        df["ticker"] = df.get("stock_id", pd.NA)
    df["ticker"] = df["ticker"].astype(str)
    return df.dropna(subset=["publish_at"])


def _counts_for_as_of(news: pd.DataFrame, as_of: pd.Timestamp, tickers: set[str]) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    feat_week_end = as_of - pd.Timedelta(days=1)  # Sunday before Monday as_of
    w1_start = feat_week_end - pd.Timedelta(days=6)
    w4_start = feat_week_end - pd.Timedelta(days=27)
    sub = news[news["publish_at"] < as_of].copy()
    rows = []
    for ticker in tickers:
        tnews = sub[sub["ticker"] == ticker]
        c1 = int(((tnews["publish_at"] >= w1_start) & (tnews["publish_at"] <= feat_week_end)).sum())
        c4 = int(((tnews["publish_at"] >= w4_start) & (tnews["publish_at"] <= feat_week_end)).sum())
        rows.append({
            "ticker": ticker,
            "news_count_1w": c1,
            "news_count_4w": c4,
        })
    return pd.DataFrame(rows)


def build_news_features(
    panel: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = cfg or load_forecast_config()
    root = Path(cfg["_root"])
    news_path = root / cfg.get("p4", {}).get(
        "news_path", "data/forecast/supplemental/news_matched.parquet",
    )
    news = _load_news(news_path)
    meta = {"source": "news", "available": not news.empty, "n_rows": len(news)}

    panel = panel.copy()
    panel["week"] = pd.to_datetime(panel["week"])
    panel["as_of"] = panel["week"].map(as_of_from_feature_week)

    if news.empty:
        for col in ("log1p_news_count_1w", "log1p_news_count_4w", "missing_news"):
            panel[col] = np.nan if col != "missing_news" else 1
        panel["missing_news"] = 1
        return panel, meta

    pieces = []
    for as_of, grp in panel.groupby("as_of", sort=True):
        counts = _counts_for_as_of(news, as_of, set(grp["ticker"].astype(str)))
        merged = grp.merge(counts, on="ticker", how="left")
        merged["news_count_1w"] = merged["news_count_1w"].fillna(0)
        merged["news_count_4w"] = merged["news_count_4w"].fillna(0)
        merged["log1p_news_count_1w"] = np.log1p(merged["news_count_1w"])
        merged["log1p_news_count_4w"] = np.log1p(merged["news_count_4w"])
        merged["missing_news"] = 0
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True)
    meta["n_matched_rows"] = int(len(out))
    return out, meta
