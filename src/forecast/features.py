"""Forecast feature whitelist, engineering, and sklearn Pipeline helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.forecast.config import load_whitelist


FORBIDDEN_ALWAYS = {
    "ret_next", "non_inst_roi_next", "turnover_next",
    "y_excess_1w", "y_excess_4w", "y_outperform_1w", "y_stock_1w", "y_bench_1w",
}


def feature_lists(whitelist: dict[str, Any] | None = None) -> tuple[list[str], list[str], list[str]]:
    wl = whitelist or load_whitelist()
    a = list(wl["A"])
    b_extra = list(wl["B_extra"])
    b = list(dict.fromkeys(a + b_extra))
    forbidden = list(dict.fromkeys(list(wl.get("forbidden", [])) + list(FORBIDDEN_ALWAYS)))
    return a, b, forbidden


def feature_lists_full(whitelist: dict[str, Any] | None = None) -> dict[str, list[str]]:
    wl = whitelist or load_whitelist()
    a, b, forbidden = feature_lists(wl)
    c = list(dict.fromkeys(a + list(wl.get("C_extra", []))))
    d = list(dict.fromkeys(a + list(wl.get("D_extra", []))))
    e = list(dict.fromkeys(a + list(wl.get("D_extra", [])) + list(wl.get("B_extra", []))))
    return {"A": a, "B": b, "C": c, "D": d, "E": e, "forbidden": forbidden}


def columns_for_set(feature_set: str, whitelist: dict[str, Any] | None = None) -> list[str]:
    fl = feature_lists_full(whitelist)
    key = feature_set.upper()
    if key not in fl:
        raise KeyError(f"unknown feature set {feature_set}")
    return fl[key]


def assert_no_forbidden(columns: list[str], forbidden: list[str]) -> None:
    bad = sorted(set(columns) & set(forbidden))
    if bad:
        raise ValueError(f"forbidden columns in feature matrix: {bad}")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out


def engineer_forecast_features(panel: pd.DataFrame, bench_by_week: pd.DataFrame) -> pd.DataFrame:
    """Derive forecast A/B columns from research panel + benchmark weekly table.

    Warm-up mitigation: AbnAtt / initiation use panel values, but we set
    `history_weeks_available` from listing_age and mark `insufficient_history`
    when fewer than 8 post-listing weeks — AbnAtt then forced to NaN for forecast X.
    """
    df = panel.copy()
    df["week"] = pd.to_datetime(df["week"])
    g = df.sort_values(["ticker", "week"]).groupby("ticker", sort=False)

    # Momentum: contiguous cumulative-ish proxies from past weekly rets (known at as_of)
    # mom_k = product(1+ret) over past k weeks after shift(1), minus 1
    def cum_mom(s: pd.Series, k: int) -> pd.Series:
        r = s.shift(1)
        # rolling product via log1p
        return np.expm1(np.log1p(r).rolling(k, min_periods=max(2, k // 2)).sum())

    df["mom_4w"] = g["ret"].transform(lambda s: cum_mom(s, 4))
    df["mom_12w"] = g["ret"].transform(lambda s: cum_mom(s, 12))
    df["mom_26w"] = g["ret"].transform(lambda s: cum_mom(s, 26))

    # Liquidity / size from weekly aggregates already on panel
    df["log_market_cap"] = np.log(df["market_cap"].where(df["market_cap"] > 0))
    df["log_adv20"] = np.log(df["value"].where(df["value"] > 0))  # weekly value as ADV proxy
    # Relative volume vs past 4 weeks
    past_val = g["value"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    df["rel_volume_1w"] = _safe_div(df["value"], past_val)

    # Volatility proxies from weekly ret (panel lacks daily in this path)
    df["vol_20d"] = g["ret"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std())  # ~4 weeks
    df["vol_60d"] = g["ret"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=4).std())

    df["amihud_20d"] = g["amihud"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    df["zero_volume_share_20d"] = g["zero_volume_days"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    df["mean_turnover_20d"] = g["turnover"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean())

    # History / warm-up
    df["history_weeks_available"] = g.cumcount()  # weeks since first post-listing row in panel
    df["insufficient_history"] = (df["history_weeks_available"] < 8).astype(int)
    for col in ("abn_attention_all", "abn_attention_weekday", "abn_attention_weekend"):
        if col in df.columns:
            df.loc[df["insufficient_history"] == 1, col] = np.nan

    # Attention transforms
    for raw, out in [
        ("att_all", "log1p_att_all"),
        ("att_weekday", "log1p_att_weekday"),
        ("att_weekend", "log1p_att_weekend"),
        ("att_high_effort", "log1p_att_high_effort"),
        ("att_mid_effort", "log1p_att_mid_effort"),
        ("att_low_effort", "log1p_att_low_effort"),
    ]:
        df[out] = np.log1p(df[raw].astype(float)) if raw in df.columns else np.nan

    df["att_count_delta_1w"] = g["att_all"].transform(lambda s: s - s.shift(1))
    df["att_count_trend_4w"] = g["att_all"].transform(
        lambda s: s.shift(0) - s.shift(1).rolling(4, min_periods=2).mean())

    att_all = df["att_all"].astype(float).replace(0, np.nan)
    df["weekend_share"] = _safe_div(df["att_weekend"].astype(float), att_all)
    df["non_trading_share"] = _safe_div(df["att_non_trading"].astype(float), att_all)
    df["missing_attention_window"] = att_all.isna().astype(int)
    df.loc[df["att_all"] == 0, "weekend_share"] = np.nan
    df.loc[df["att_all"] == 0, "non_trading_share"] = np.nan
    df.loc[df["att_all"] == 0, "missing_attention_window"] = 1

    # Benchmark state known at feature week (same-week market state, not return week).
    b_sorted = bench_by_week.sort_values("week").copy()
    b_sorted["bench_ret_1w"] = b_sorted["bench_ret"]
    b_sorted["bench_vol_20d"] = (
        b_sorted["bench_ret"].shift(1).rolling(4, min_periods=2).std()
    )
    df = df.merge(
        b_sorted[["week", "bench_ret_1w", "bench_vol_20d"]],
        on="week",
        how="left",
    )

    df["missing_market_cap"] = df["market_cap"].isna().astype(int)
    df["missing_foreign_holding"] = df["foreign_holding_pct"].isna().astype(int)
    df["missing_amihud"] = df["amihud"].isna().astype(int)

    if "is_code_only_matched" in df.columns:
        df["is_code_only_matched"] = df["is_code_only_matched"].astype(int)
    else:
        df["is_code_only_matched"] = 0

    return df


def make_ridge_pipeline(feature_names: list[str], alpha: float, seed: int = 42) -> Pipeline:
    """Median impute + missing indicators are handled upstream; here: impute + scale + ridge."""
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    # ColumnTransformer with one block keeps feature order stable
    ct = ColumnTransformer([
        ("num", pre, list(range(len(feature_names)))),
    ], remainder="drop")
    return Pipeline([
        ("prep", ct),
        ("model", Ridge(alpha=alpha, random_state=seed)),
    ])


def matrix_from_frame(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    assert_no_forbidden(cols, list(FORBIDDEN_ALWAYS))
    return df[cols].to_numpy(dtype=float)
