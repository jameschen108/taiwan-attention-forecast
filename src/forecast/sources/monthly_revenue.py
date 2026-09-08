"""Monthly revenue features by announcement date (not revenue month)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forecast.config import load_forecast_config
from src.forecast.time_contract import as_of_from_feature_week
from src.features.sessions import week_of


def _load_finmind_month_revenue(raw_dir: Path) -> pd.DataFrame:
    frames = []
    kind_dir = raw_dir / "month_revenue"
    if not kind_dir.exists():
        return pd.DataFrame()
    for path in sorted(kind_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = doc.get("data") or []
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"stock_id": "ticker"})
    df["ticker"] = df["ticker"].astype(str)
    # FinMind: `date` is announcement/publication date
    df["announce_date"] = pd.to_datetime(df["date"])
    for col in ("revenue", "Revenue", "revenue_month", "RevenueMonth"):
        if col in df.columns and "revenue" not in df.columns:
            df["revenue"] = df[col]
    yoy_cols = [c for c in df.columns if c.lower() in {"revenueyoy", "revenue_yoy", "yoy"}]
    if yoy_cols:
        df["revenue_yoy"] = pd.to_numeric(df[yoy_cols[0]], errors="coerce")
    mom_cols = [c for c in df.columns if c.lower() in {"revenuemom", "revenue_mom", "mom"}]
    if mom_cols:
        df["revenue_mom"] = pd.to_numeric(df[mom_cols[0]], errors="coerce")
    df["revenue"] = pd.to_numeric(df.get("revenue"), errors="coerce")
    df = df.sort_values(["ticker", "announce_date"])
    # Compute YoY/MoM when not provided by API
    if "revenue_yoy" not in df.columns or df["revenue_yoy"].isna().all():
        df["revenue_yoy"] = df.groupby("ticker")["revenue"].pct_change(12)
    if "revenue_mom" not in df.columns or df["revenue_mom"].isna().all():
        df["revenue_mom"] = df.groupby("ticker")["revenue"].pct_change(1)
    for col in ("revenue_yoy", "revenue_mom", "log_revenue_latest"):
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def latest_revenue_known_before(
    revenue: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Most recent announced revenue strictly before as_of for each ticker."""
    as_of = pd.Timestamp(as_of)
    sub = revenue[revenue["announce_date"] < as_of].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ticker", "revenue_yoy_latest", "revenue_mom_latest",
                                     "log_revenue_latest", "days_since_revenue_announce"])
    sub = sub.sort_values(["ticker", "announce_date"]).groupby("ticker", sort=False).tail(1)
    sub = sub.rename(columns={
        "revenue_yoy": "revenue_yoy_latest",
        "revenue_mom": "revenue_mom_latest",
    })
    sub["log_revenue_latest"] = np.log(sub["revenue"].where(sub["revenue"] > 0))
    sub["days_since_revenue_announce"] = (as_of - sub["announce_date"]).dt.days
    return sub[["ticker", "revenue_yoy_latest", "revenue_mom_latest",
                "log_revenue_latest", "days_since_revenue_announce", "announce_date"]]


def build_revenue_features(
    panel: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = cfg or load_forecast_config()
    root = Path(cfg["_root"])
    p4 = cfg.get("p4", {})
    cached = root / p4.get("month_revenue_cache", "data/forecast/supplemental/monthly_revenue.parquet")
    raw_dir = root / p4.get("month_revenue_raw_dir", "data/raw/finmind/month_revenue")

    if cached.exists():
        revenue = pd.read_parquet(cached)
        revenue["announce_date"] = pd.to_datetime(revenue["announce_date"])
    else:
        revenue = _load_finmind_month_revenue(raw_dir)

    meta = {"source": "month_revenue", "available": not revenue.empty, "n_rows": len(revenue)}
    panel = panel.copy()
    panel["week"] = pd.to_datetime(panel["week"])
    panel["as_of"] = panel["week"].map(as_of_from_feature_week)

    if revenue.empty:
        for col in ("revenue_yoy_latest", "revenue_mom_latest", "log_revenue_latest",
                    "days_since_revenue_announce", "missing_revenue"):
            panel[col] = np.nan if col != "missing_revenue" else 1
        panel["missing_revenue"] = 1
        return panel, meta

    cached.parent.mkdir(parents=True, exist_ok=True)
    revenue.to_parquet(cached, index=False)

    pieces = []
    for as_of, grp in panel.groupby("as_of", sort=True):
        latest = latest_revenue_known_before(revenue, as_of)
        merged = grp.merge(latest, on="ticker", how="left")
        merged["missing_revenue"] = merged["revenue_yoy_latest"].isna().astype(int)
        for col in ("revenue_yoy_latest", "revenue_mom_latest", "log_revenue_latest"):
            if col in merged.columns:
                merged[col] = merged[col].replace([np.inf, -np.inf], np.nan)
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True)
    meta["n_tickers_with_revenue"] = int((out["missing_revenue"] == 0).sum())
    return out, meta
