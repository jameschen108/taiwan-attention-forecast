"""Build supplemental P4 features (news, revenue) and attach to panel."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.forecast.config import load_forecast_config
from src.forecast.sources.coverage import coverage_row, write_source_coverage
from src.forecast.sources.monthly_revenue import build_revenue_features
from src.forecast.sources.news import build_news_features


def build_supplemental_features(
    panel: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    cfg = cfg or load_forecast_config()
    out = panel.copy()
    coverage: list[dict] = []

    news_panel, news_meta = build_news_features(out, cfg)
    rev_panel, rev_meta = build_revenue_features(out, cfg)

    extra_cols = [
        "log1p_news_count_1w", "log1p_news_count_4w", "missing_news",
        "revenue_yoy_latest", "revenue_mom_latest", "log_revenue_latest",
        "days_since_revenue_announce", "missing_revenue",
    ]
    merge_keys = ["ticker", "week"]
    out = out.drop(columns=[c for c in extra_cols if c in out.columns], errors="ignore")
    out = out.merge(
        news_panel[merge_keys + [c for c in extra_cols[:3] if c in news_panel.columns]],
        on=merge_keys, how="left",
    )
    out = out.merge(
        rev_panel[merge_keys + [c for c in extra_cols[3:] if c in rev_panel.columns]],
        on=merge_keys, how="left",
    )

    if "as_of" in news_panel.columns:
        for as_of in sorted(news_panel["as_of"].dropna().unique()):
            coverage.append(coverage_row(
                "news", as_of,
                available=bool(news_meta.get("available")),
                n_records=int(news_meta.get("n_rows", 0)),
                notes="" if news_meta.get("available") else "news parquet missing",
            ))
            coverage.append(coverage_row(
                "month_revenue", as_of,
                available=bool(rev_meta.get("available")),
                n_records=int(rev_meta.get("n_rows", 0)),
                notes="" if rev_meta.get("available") else "month_revenue raw/cache missing",
            ))
            break

    if coverage:
        write_source_coverage(coverage, cfg)
    return out, [news_meta, rev_meta]
