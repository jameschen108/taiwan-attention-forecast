"""Assemble forecast features/labels tables from research panel + benchmark."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from src.forecast.config import load_forecast_config, load_whitelist, resolve_path
from src.forecast.features import engineer_forecast_features, feature_lists
from src.forecast.labels import build_label_frame, load_or_build_benchmark_weekly
from src.forecast.time_contract import as_of_from_feature_week


def _load_trading_days(path: Path) -> list[dt.date]:
    cal = pd.read_csv(path, parse_dates=["date"])
    return sorted(d.date() for d in cal["date"])


def build_forecast_tables(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_forecast_config()
    root = Path(cfg["_root"])
    panel_path = resolve_path(cfg, "panel_path")
    daily_path = resolve_path(cfg, "daily_path")  # reserved for future PIT daily feats
    _ = daily_path
    trading_days_path = resolve_path(cfg, "trading_days_path")
    bench_price = resolve_path(cfg, "benchmark_price_path")
    out_dir = resolve_path(cfg, "output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    panel["ticker"] = panel["ticker"].astype(str)
    panel["week"] = pd.to_datetime(panel["week"])

    trading_days = _load_trading_days(trading_days_path)
    exrights = root / "data" / "interim" / "ex_rights.csv"
    bench_weekly = load_or_build_benchmark_weekly(
        bench_price, exrights, out_dir / "benchmark_weekly.parquet",
    )

    feat_src = engineer_forecast_features(panel, bench_weekly)
    feat_src["as_of"] = feat_src["week"].map(as_of_from_feature_week)
    feat_src["feature_version"] = cfg["features"]["whitelist_version"]
    feat_src["availability_mode"] = cfg["data"]["availability_mode"]
    feat_src["universe_mode"] = cfg["universe"]["development_mode"]

    a_cols, b_cols, forbidden = feature_lists(load_whitelist())
    meta = [
        "ticker", "week", "as_of", "feature_version", "availability_mode",
        "universe_mode", "sparsity_tier", "listing_date", "sector",
    ]
    feature_cols = list(dict.fromkeys(a_cols + b_cols))
    # Ensure all engineered cols exist
    for c in feature_cols:
        if c not in feat_src.columns:
            feat_src[c] = pd.NA

    features = feat_src[meta + feature_cols].copy()
    # Hard guard: never persist forbidden columns in features table
    drop_forbidden = [c for c in features.columns if c in forbidden or c in {
        "ret_next", "ret_fwd2", "ret_fwd3", "ret_fwd4", "ret_fwd5", "ret_fwd6",
        "ret_fwd7", "ret_fwd8", "non_inst_roi_next", "turnover_next",
    }]
    features = features.drop(columns=drop_forbidden, errors="ignore")

    labels = build_label_frame(
        panel,
        trading_days,
        bench_weekly,
        label_definition=cfg["forecast"].get("label_definition", "adjusted_price_proxy"),
    )

    features.to_parquet(out_dir / "features.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)

    # Sample exclusion report
    excl = pd.DataFrame([
        {
            "n_feature_rows": len(features),
            "n_label_rows": len(labels),
            "n_tickers": features["ticker"].nunique(),
            "n_as_of": features["as_of"].nunique(),
            "n_labels_ok": int((labels["label_status"] == "ok").sum()),
            "n_schedule_skip": int((labels["label_status"] == "schedule_skip").sum()),
            "n_benchmark_missing": int((labels["label_status"] == "benchmark_missing").sum()),
            "n_stock_missing": int(
                (labels["label_status"] == "entry_unavailable_or_missing_stock").sum()),
            "first_as_of": str(features["as_of"].min().date()),
            "last_as_of": str(features["as_of"].max().date()),
            "whitelist_version": cfg["features"]["whitelist_version"],
            "availability_mode": cfg["data"]["availability_mode"],
            "label_definition": cfg["forecast"].get("label_definition"),
        }
    ])
    excl.to_csv(out_dir / "sample_exclusion_report.csv", index=False)

    return {"features": features, "labels": labels, "benchmark_weekly": bench_weekly}


def join_xy(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: str = "A",
) -> pd.DataFrame:
    """Join features with mature-capable labels; does not drop latest unlabeled rows from features."""
    a_cols, b_cols, _ = feature_lists()
    cols = a_cols if feature_set.upper() == "A" else b_cols
    keys = ["ticker", "week", "as_of"]
    lab = labels[[
        "ticker", "week", "as_of", "y_excess_1w", "y_outperform_1w",
        "label_end_at", "label_available_at", "label_status",
    ]]
    out = features[keys + ["sparsity_tier"] + cols].merge(lab, on=keys, how="left")
    out["feature_set"] = feature_set.upper()
    return out
