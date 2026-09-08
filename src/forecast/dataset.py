"""Assemble forecast features/labels tables from research panel + benchmark."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from src.forecast.config import load_forecast_config, load_whitelist, resolve_path
from src.forecast.features import columns_for_set, engineer_forecast_features, feature_lists
from src.forecast.labels import build_label_frame, load_or_build_benchmark_weekly
from src.forecast.time_contract import as_of_from_feature_week


def _load_trading_days(path: Path) -> list[dt.date]:
    cal = pd.read_csv(path, parse_dates=["date"])
    return sorted(d.date() for d in cal["date"])


def build_forecast_tables(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_forecast_config()
    root = Path(cfg["_root"])
    panel_path = resolve_path(cfg, "panel_path")
    daily_path = resolve_path(cfg, "daily_path")
    trading_days_path = resolve_path(cfg, "trading_days_path")
    bench_price = resolve_path(cfg, "benchmark_price_path")
    out_dir = resolve_path(cfg, "output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    panel["ticker"] = panel["ticker"].astype(str)
    panel["week"] = pd.to_datetime(panel["week"])

    stock_daily = pd.read_parquet(daily_path)
    stock_daily["date"] = pd.to_datetime(stock_daily["date"])
    stock_daily["ticker"] = stock_daily["ticker"].astype(str)

    trading_days = _load_trading_days(trading_days_path)
    exrights = root / "data" / "interim" / "ex_rights.csv"
    bench_weekly, bench_daily = load_or_build_benchmark_weekly(
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
        stock_daily=stock_daily,
        bench_daily=bench_daily,
        label_definition=cfg["forecast"].get("label_definition", "adjusted_price_proxy"),
    )

    n_1w = int((labels["horizon"] == "1w").sum()) if "horizon" in labels.columns else len(labels)
    n_4w = int((labels["horizon"] == "4w").sum()) if "horizon" in labels.columns else 0

    features.to_parquet(out_dir / "features.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)

    # Sample exclusion report
    excl = pd.DataFrame([
        {
            "n_feature_rows": len(features),
            "n_label_rows": len(labels),
            "n_label_1w": n_1w,
            "n_label_4w": n_4w,
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

    return {
        "features": features,
        "labels": labels,
        "benchmark_weekly": bench_weekly,
        "stock_daily": stock_daily,
        "bench_daily": bench_daily,
    }


def join_xy(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: str = "A",
    horizon: str = "1w",
) -> pd.DataFrame:
    """Join features with labels for a given horizon."""
    a_cols, b_cols, _ = feature_lists()
    if feature_set.upper() in {"A", "B"}:
        cols = a_cols if feature_set.upper() == "A" else b_cols
    else:
        cols = columns_for_set(feature_set)
    keys = ["ticker", "week", "as_of"]
    hz = horizon.lower()
    y_excess = f"y_excess_{hz}"
    y_out = f"y_outperform_{hz}"
    lab_cols = ["ticker", "week", "as_of", "label_end_at", "label_available_at", "label_status"]
    for c in (y_excess, y_out):
        if c in labels.columns:
            lab_cols.append(c)
    lab = labels[labels["horizon"] == hz][lab_cols].drop_duplicates(keys)
    out = features[keys + ["sparsity_tier"] + cols].merge(lab, on=keys, how="left")
    out["feature_set"] = feature_set.upper()
    out["horizon"] = hz
    if hz == "1w" and y_excess in out.columns:
        out["y_excess_1w"] = out[y_excess]
        out["y_outperform_1w"] = out[y_out]
    return out
