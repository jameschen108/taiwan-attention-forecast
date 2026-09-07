"""Ridge A/B training and hyperparameter selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.forecast.evaluate import weekly_rank_ic
from src.forecast.features import feature_lists
from src.forecast.splits import filter_mature_fast, inner_forward_blocks


@dataclass
class FitResult:
    model_name: str
    feature_set: str
    alpha: float
    pipeline: Pipeline
    feature_names: list[str]
    fit_cutoff: pd.Timestamp
    n_train_rows: int
    n_train_weeks: int


def _pipeline(n_features: int, alpha: float, seed: int) -> Pipeline:
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    ct = ColumnTransformer([("num", pre, list(range(n_features)))], remainder="drop")
    return Pipeline([
        ("prep", ct),
        ("model", Ridge(alpha=alpha, random_state=seed)),
    ])


def _xy(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    use = df.dropna(subset=["y_excess_1w"]).copy()
    X = use[cols].to_numpy(dtype=float)
    y = use["y_excess_1w"].to_numpy(dtype=float)
    return X, y, use


def select_alpha_inner(
    train_pool: pd.DataFrame,
    feature_cols: list[str],
    alphas: list[float],
    seed: int = 42,
    n_blocks: int = 3,
    block_weeks: int = 26,
    min_train_weeks: int = 104,
) -> tuple[float, pd.DataFrame]:
    """Pick Ridge alpha by mean weekly Rank IC on inner forward blocks."""
    blocks = inner_forward_blocks(
        train_pool, n_blocks=n_blocks, block_weeks=block_weeks,
        min_train_weeks=min_train_weeks,
    )
    if not blocks:
        # Fall back: single holdout last 26 weeks if possible
        weeks = sorted(train_pool["as_of"].unique())
        if len(weeks) < min_train_weeks // 2:
            return alphas[len(alphas) // 2], pd.DataFrame()
        valid_start = weeks[-min(block_weeks, len(weeks) // 4)]
        blocks = [(valid_start, valid_start, weeks[-1] + pd.Timedelta(days=1))]

    records = []
    for alpha in alphas:
        ics = []
        for train_end, valid_start, valid_end in blocks:
            tr = filter_mature_fast(train_pool, fit_cutoff=valid_start)
            va = train_pool[
                (train_pool["as_of"] >= valid_start)
                & (train_pool["as_of"] < valid_end)
                & (train_pool["y_excess_1w"].notna())
                & (train_pool["label_status"] == "ok")
            ]
            if len(tr) < 100 or va["as_of"].nunique() < 5:
                continue
            Xtr, ytr, _ = _xy(tr, feature_cols)
            pipe = _pipeline(len(feature_cols), alpha, seed)
            pipe.fit(Xtr, ytr)
            Xva, _, va_df = _xy(va, feature_cols)
            pred = pipe.predict(Xva)
            scored = va_df[["as_of", "y_excess_1w"]].copy()
            scored["pred"] = pred
            ic = weekly_rank_ic(scored, min_names=10)
            if not ic.empty:
                ics.append(float(ic["ic"].mean()))
                records.append({
                    "alpha": alpha,
                    "valid_start": str(valid_start.date()),
                    "valid_end": str((valid_end - pd.Timedelta(days=1)).date()),
                    "mean_ic": float(ic["ic"].mean()),
                    "n_valid_weeks": int(ic["ic"].notna().sum()),
                })
        mean_ic = float(np.mean(ics)) if ics else -np.inf
        records.append({"alpha": alpha, "valid_start": "AGG", "valid_end": "AGG",
                        "mean_ic": mean_ic, "n_valid_weeks": len(ics)})

    rec = pd.DataFrame(records)
    agg = rec[rec["valid_start"] == "AGG"].sort_values(
        ["mean_ic", "alpha"], ascending=[False, False])
    if agg.empty or not np.isfinite(agg.iloc[0]["mean_ic"]):
        # Prefer stronger regularization on failure
        return max(alphas), rec
    return float(agg.iloc[0]["alpha"]), rec


def fit_ridge(
    train_df: pd.DataFrame,
    feature_set: str,
    alpha: float,
    fit_cutoff: pd.Timestamp,
    seed: int = 42,
    sparsity_tiers: list[str] | None = None,
) -> FitResult:
    a_cols, b_cols, _ = feature_lists()
    cols = a_cols if feature_set.upper() == "A" else b_cols
    tr = filter_mature_fast(train_df, fit_cutoff)
    if sparsity_tiers:
        tr = tr[tr["sparsity_tier"].isin(sparsity_tiers)]
    X, y, used = _xy(tr, cols)
    pipe = _pipeline(len(cols), alpha, seed)
    pipe.fit(X, y)
    return FitResult(
        model_name=f"ridge_{feature_set.lower()}_a{alpha}",
        feature_set=feature_set.upper(),
        alpha=alpha,
        pipeline=pipe,
        feature_names=cols,
        fit_cutoff=pd.Timestamp(fit_cutoff),
        n_train_rows=len(used),
        n_train_weeks=int(used["as_of"].nunique()),
    )


def predict_frame(fit: FitResult, df: pd.DataFrame) -> pd.Series:
    X = df[fit.feature_names].to_numpy(dtype=float)
    return pd.Series(fit.pipeline.predict(X), index=df.index, name="prediction")


def constant_predictor(train_df: pd.DataFrame, fit_cutoff: pd.Timestamp) -> float:
    tr = filter_mature_fast(train_df, fit_cutoff)
    if tr.empty:
        return 0.0
    return float(tr["y_excess_1w"].mean())
