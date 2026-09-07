"""Benchmark (0050) weekly returns for excess labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.sessions import week_of
from src.market.normalize import apply_adjustment


def load_benchmark_prices(price_json: Path, exrights_csv: Path | None = None) -> pd.DataFrame:
    """Load a single-ticker FinMind price JSON and apply ex-rights factors if available."""
    doc = json.loads(Path(price_json).read_text(encoding="utf-8"))
    rows = doc.get("data") or []
    if not rows:
        raise FileNotFoundError(f"benchmark price empty: {price_json}")
    df = pd.DataFrame(rows).rename(columns={
        "stock_id": "ticker",
        "Trading_Volume": "volume",
        "Trading_money": "value",
        "max": "high",
        "min": "low",
        "Trading_turnover": "n_transactions",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)
    df = df[["ticker", "date", "open", "high", "low", "close", "volume", "value"]].copy()

    factors = pd.DataFrame(columns=["ticker", "date", "factor"])
    if exrights_csv is not None and Path(exrights_csv).exists():
        ex = pd.read_csv(exrights_csv, dtype={"ticker": str}, parse_dates=["date"])
        ticker = str(df["ticker"].iloc[0])
        ex = ex[ex["ticker"] == ticker].copy()
        if not ex.empty and "factor" in ex.columns:
            ex = ex[(ex["factor"] > 0.3) & (ex["factor"] <= 1.05)]
            factors = ex[["ticker", "date", "factor"]]
    df = apply_adjustment(df, factors)
    return df.sort_values("date").reset_index(drop=True)


def weekly_benchmark_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Monday-open to Friday-close adjusted return by W-SUN week label of that return week.

    The week label here is the **return week** Sunday (same as panel `week` for `ret`),
    not the feature week.
    """
    df = daily.copy()
    df["week"] = df["date"].map(week_of)
    agg = (df.groupby("week", sort=True)
             .agg(
                 n_trading_days=("date", "size"),
                 open_adj=("adj_open", "first"),
                 close_adj=("adj_close", "last"),
                 first_date=("date", "min"),
                 last_date=("date", "max"),
             )
             .reset_index())
    agg["bench_ret"] = agg["close_adj"] / agg["open_adj"] - 1.0
    agg.loc[agg["open_adj"].isna() | agg["close_adj"].isna(), "bench_ret"] = np.nan
    return agg


def daily_vol(daily: pd.DataFrame, window: int = 20) -> pd.Series:
    """Trailing daily-return volatility indexed by date (shifted to be past-only when mapped)."""
    d = daily.sort_values("date").copy()
    d["daily_ret"] = d["adj_close"].pct_change(fill_method=None)
    return d.set_index("date")["daily_ret"].rolling(window, min_periods=max(5, window // 2)).std()
