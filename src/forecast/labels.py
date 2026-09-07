"""1-week and 4-week excess labels vs benchmark (adjusted-price proxy)."""

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
    label_end_4w,
)


def _count_trading_sessions(
    entry_at: pd.Timestamp | pd.NaT,
    exit_at: pd.Timestamp | pd.NaT,
    trading_days_sorted: list[dt.date],
) -> float:
    """Count unique market sessions in [entry_at, exit_at] inclusive."""
    import bisect

    if pd.isna(entry_at) or pd.isna(exit_at):
        return np.nan
    start = pd.Timestamp(entry_at).date()
    end = pd.Timestamp(exit_at).date()
    lo = bisect.bisect_left(trading_days_sorted, start)
    hi = bisect.bisect_right(trading_days_sorted, end)
    return float(hi - lo)


def _period_adj_return_from_group(
    grp: pd.DataFrame,
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
) -> float | np.nan:
    if pd.isna(entry_at) or pd.isna(exit_at) or grp.empty:
        return np.nan
    entry = pd.Timestamp(entry_at).normalize()
    exit_ = pd.Timestamp(exit_at).normalize()
    sub = grp[(grp["date"] >= entry) & (grp["date"] <= exit_)].sort_values("date")
    if sub.empty:
        return np.nan
    entry_px = sub.iloc[0]["adj_open"]
    exit_px = sub.iloc[-1]["adj_close"]
    if pd.isna(entry_px) or pd.isna(exit_px) or entry_px <= 0:
        return np.nan
    return float(exit_px / entry_px - 1.0)


def _batch_period_returns(
    tickers: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    daily: pd.DataFrame,
    bench_ticker: str | None = None,
) -> pd.Series:
    """Vectorized-ish batch: dedupe (ticker, entry, exit) then map back."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    by_ticker = {t: g for t, g in daily.groupby("ticker", sort=False)}
    bench_grp = by_ticker.get(bench_ticker, pd.DataFrame()) if bench_ticker else None

    keys = pd.DataFrame({"ticker": tickers, "entry_at": entries, "label_end_at": exits})
    uniq = keys.drop_duplicates()
    ret_map: dict[tuple, float] = {}
    for row in uniq.itertuples(index=False):
        t, ent, ex = row.ticker, row.entry_at, row.label_end_at
        grp = bench_grp if bench_ticker and t == bench_ticker else by_ticker.get(t, pd.DataFrame())
        ret_map[(t, ent, ex)] = _period_adj_return_from_group(grp, ent, ex)

    return keys.apply(
        lambda r: ret_map.get((r["ticker"], r["entry_at"], r["label_end_at"]), np.nan),
        axis=1,
    )


def _build_horizon_labels(
    panel: pd.DataFrame,
    trading_days_sorted: list[dt.date],
    stock_daily: pd.DataFrame,
    bench_daily: pd.DataFrame,
    benchmark_weekly: pd.DataFrame,
    horizon: str,
    label_end_fn,
    label_version: str,
    label_definition: str,
) -> pd.DataFrame:
    df = panel[["ticker", "week", "ret_next", "listing_date"]].copy()
    df["week"] = pd.to_datetime(df["week"])
    df["as_of"] = df["week"].map(as_of_from_feature_week)
    df["horizon"] = horizon
    df["label_version"] = label_version
    df["label_definition"] = label_definition
    df["return_week"] = df["week"] + pd.Timedelta(days=7)

    entries, ends = [], []
    for w in df["week"]:
        entries.append(entry_at_for_feature_week(w, trading_days_sorted))
        ends.append(label_end_fn(w, trading_days_sorted))
    df["entry_at"] = entries
    df["label_end_at"] = ends
    df["label_available_at"] = df["label_end_at"].map(label_available_at)

    df["ticker"] = df["ticker"].astype(str)
    stock_s = _batch_period_returns(df["ticker"], df["entry_at"], df["label_end_at"], stock_daily)
    bench_s = _batch_period_returns(
        pd.Series(["0050"] * len(df)), df["entry_at"], df["label_end_at"], bench_daily, "0050",
    )
    y_stock = f"y_stock_{horizon}"
    y_bench = f"y_bench_{horizon}"
    y_excess = f"y_excess_{horizon}"
    y_out = f"y_outperform_{horizon}"
    df[y_stock] = stock_s.values
    df[y_bench] = bench_s.values
    df[y_excess] = df[y_stock] - df[y_bench]
    df[y_out] = np.where(df[y_excess].isna(), np.nan, (df[y_excess] > 0).astype(float))

    # Trading days in hold period: unique calendar sessions (not ticker-day rows)
    day_map: dict[tuple, float] = {}
    for ent, ex in zip(df["entry_at"], df["label_end_at"]):
        key = (ent, ex)
        if key not in day_map:
            day_map[key] = _count_trading_sessions(ent, ex, trading_days_sorted)
    df["n_trading_days"] = [day_map[(e, x)] for e, x in zip(df["entry_at"], df["label_end_at"])]

    status = np.full(len(df), "ok", dtype=object)
    status[df["entry_at"].isna()] = "schedule_skip"
    status[df[y_stock].isna() & (status == "ok")] = "entry_unavailable_or_missing_stock"
    status[df[y_bench].isna() & (status == "ok")] = "benchmark_missing"
    status[df[y_excess].isna() & (status == "ok")] = "label_missing"
    df["label_status"] = status

    keep = [
        "ticker", "week", "as_of", "horizon", "label_version", "label_definition",
        "entry_at", "label_end_at", "label_available_at", "return_week",
        y_stock, y_bench, y_excess, y_out, "label_status", "n_trading_days",
    ]
    return df[keep].sort_values(["as_of", "ticker"]).reset_index(drop=True)


def build_label_frame(
    panel: pd.DataFrame,
    trading_days_sorted: list[dt.date],
    benchmark_weekly: pd.DataFrame,
    stock_daily: pd.DataFrame | None = None,
    bench_daily: pd.DataFrame | None = None,
    label_definition: str = "adjusted_price_proxy",
) -> pd.DataFrame:
    """Build stacked 1w + 4w label rows.

    1w uses panel ret_next when available; 4w uses continuous hold open→close
    from daily adj prices (not ret_fwd4).
    """
    if stock_daily is not None and bench_daily is not None:
        stock_daily = stock_daily.copy()
        stock_daily["date"] = pd.to_datetime(stock_daily["date"])
        stock_daily["ticker"] = stock_daily["ticker"].astype(str)
        bench_daily = bench_daily.copy()
        bench_daily["date"] = pd.to_datetime(bench_daily["date"])
        bench_daily["ticker"] = bench_daily["ticker"].astype(str)

        lab1 = _build_horizon_labels(
            panel, trading_days_sorted, stock_daily, bench_daily, benchmark_weekly,
            horizon="1w", label_end_fn=label_end_1w,
            label_version="forecast_v1_1w", label_definition=label_definition,
        )
        # Prefer contiguous ret_next when it matches 1w window (already on panel)
        merge = panel[["ticker", "week", "ret_next"]].copy()
        merge["week"] = pd.to_datetime(merge["week"])
        lab1 = lab1.drop(columns=["y_stock_1w"], errors="ignore").merge(
            merge.rename(columns={"ret_next": "y_stock_1w"}),
            on=["ticker", "week"], how="left",
        )
        bench = benchmark_weekly.rename(columns={"week": "return_week", "bench_ret": "y_bench_1w"})
        lab1 = lab1.drop(columns=["y_bench_1w"], errors="ignore").merge(
            bench[["return_week", "y_bench_1w"]], on="return_week", how="left",
        )
        lab1["y_excess_1w"] = lab1["y_stock_1w"] - lab1["y_bench_1w"]
        lab1["y_outperform_1w"] = np.where(
            lab1["y_excess_1w"].isna(), np.nan, (lab1["y_excess_1w"] > 0).astype(float),
        )

        lab4 = _build_horizon_labels(
            panel, trading_days_sorted, stock_daily, bench_daily, benchmark_weekly,
            horizon="4w", label_end_fn=label_end_4w,
            label_version="forecast_v1_4w", label_definition=label_definition,
        )
        return pd.concat([lab1, lab4], ignore_index=True)

    # Fallback: 1w only from panel (P0/P1 compat)
    df = panel[["ticker", "week", "ret", "ret_next", "listing_date"]].copy()
    df["week"] = pd.to_datetime(df["week"])
    df["as_of"] = df["week"].map(as_of_from_feature_week)
    df["horizon"] = "1w"
    df["label_version"] = "forecast_v1_1w"
    df["label_definition"] = label_definition
    df["return_week"] = df["week"] + pd.Timedelta(days=7)
    bench = benchmark_weekly.rename(columns={"week": "return_week", "bench_ret": "y_bench_1w"})
    df = df.merge(
        bench[["return_week", "y_bench_1w", "n_trading_days"]],
        on="return_week", how="left",
    )
    entries, ends = [], []
    for w in df["week"]:
        entries.append(entry_at_for_feature_week(w, trading_days_sorted))
        ends.append(label_end_1w(w, trading_days_sorted))
    df["entry_at"] = entries
    df["label_end_at"] = ends
    df["label_available_at"] = df["label_end_at"].map(label_available_at)
    df["y_stock_1w"] = df["ret_next"]
    df["y_excess_1w"] = df["y_stock_1w"] - df["y_bench_1w"]
    df["y_outperform_1w"] = np.where(
        df["y_excess_1w"].isna(), np.nan, (df["y_excess_1w"] > 0).astype(float),
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = load_benchmark_prices(price_json, exrights_csv)
    weekly = weekly_benchmark_returns(daily)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_parquet(out_path, index=False)
        daily.to_parquet(out_path.with_name("benchmark_daily.parquet"), index=False)
    return weekly, daily
