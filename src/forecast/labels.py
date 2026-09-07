"""1-week excess labels vs benchmark (adjusted-price proxy for P0/P1)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast.benchmark import load_benchmark_prices, weekly_benchmark_returns
from src.forecast.time_contract import (
    as_of_from_feature_week,
    entry_at_for_feature_week,
    label_available_at,
    label_end_1w,
)


def build_label_frame(
    panel: pd.DataFrame,
    trading_days_sorted: list[dt.date],
    benchmark_weekly: pd.DataFrame,
    label_definition: str = "adjusted_price_proxy",
) -> pd.DataFrame:
    """Build labels.parquet rows from research panel + benchmark weekly returns.

    Primary label:
        y_excess_1w = ret_next - bench_ret on the return week
    where ret_next is the stock's next contiguous calendar-week return already on panel,
    and bench_ret is 0050's same return-week open→close adjusted return.

    Marked as adjusted_price_proxy until a full corporate-action cash ledger exists.
    """
    df = panel[["ticker", "week", "ret", "ret_next", "listing_date"]].copy()
    df["week"] = pd.to_datetime(df["week"])
    df["as_of"] = df["week"].map(as_of_from_feature_week)
    df["horizon"] = "1w"
    df["label_version"] = "forecast_v1_1w"
    df["label_definition"] = label_definition

    # Return week = feature week + 7D
    df["return_week"] = df["week"] + pd.Timedelta(days=7)
    bench = benchmark_weekly.rename(columns={"week": "return_week", "bench_ret": "y_bench_1w"})
    df = df.merge(
        bench[["return_week", "y_bench_1w", "n_trading_days", "first_date", "last_date"]],
        on="return_week",
        how="left",
        suffixes=("", "_bench"),
    )

    entries = []
    ends = []
    for w in df["week"]:
        entries.append(entry_at_for_feature_week(w, trading_days_sorted))
        ends.append(label_end_1w(w, trading_days_sorted))
    df["entry_at"] = entries
    df["label_end_at"] = ends
    df["label_available_at"] = df["label_end_at"].map(label_available_at)

    df["y_stock_1w"] = df["ret_next"]
    df["y_excess_1w"] = df["y_stock_1w"] - df["y_bench_1w"]
    df["y_outperform_1w"] = np.where(
        df["y_excess_1w"].isna(), np.nan, (df["y_excess_1w"] > 0).astype(float)
    )

    status = np.full(len(df), "ok", dtype=object)
    status[df["entry_at"].isna()] = "schedule_skip"
    status[df["y_stock_1w"].isna() & (status == "ok")] = "entry_unavailable_or_missing_stock"
    status[df["y_bench_1w"].isna() & (status == "ok")] = "benchmark_missing"
    status[df["y_excess_1w"].isna() & (status == "ok")] = "label_missing"
    df["label_status"] = status

    keep = [
        "ticker", "week", "as_of", "horizon", "label_version", "label_definition",
        "entry_at", "label_end_at", "label_available_at", "return_week",
        "y_stock_1w", "y_bench_1w", "y_excess_1w", "y_outperform_1w",
        "label_status", "n_trading_days",
    ]
    return df[keep].sort_values(["as_of", "ticker"]).reset_index(drop=True)


def load_or_build_benchmark_weekly(
    price_json: Path,
    exrights_csv: Path,
    out_path: Path | None = None,
) -> pd.DataFrame:
    daily = load_benchmark_prices(price_json, exrights_csv)
    weekly = weekly_benchmark_returns(daily)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_parquet(out_path, index=False)
        daily.to_parquet(out_path.with_name("benchmark_daily.parquet"), index=False)
    return weekly
