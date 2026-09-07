"""Weekly Rank IC and A/B paired evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_rank_ic(
    df: pd.DataFrame,
    pred_col: str = "pred",
    y_col: str = "y_excess_1w",
    as_of_col: str = "as_of",
    min_names: int = 30,
) -> pd.DataFrame:
    """Per-as_of Spearman Rank IC; weeks with < min_names are NaN and flagged."""
    rows = []
    for as_of, g in df.groupby(as_of_col, sort=True):
        sub = g[[pred_col, y_col]].dropna()
        n = len(sub)
        if n < min_names:
            rows.append({"as_of": as_of, "ic": np.nan, "n": n, "sufficient": False})
            continue
        if float(sub[pred_col].std(ddof=0) or 0.0) == 0.0:
            rows.append({"as_of": as_of, "ic": np.nan, "n": n, "sufficient": False,
                         "reason": "constant_prediction"})
            continue
        ic = sub[pred_col].corr(sub[y_col], method="spearman")
        rows.append({"as_of": as_of, "ic": float(ic) if pd.notna(ic) else np.nan,
                     "n": n, "sufficient": True})
    return pd.DataFrame(rows)


def paired_delta_ic(
    ic_a: pd.DataFrame,
    ic_b: pd.DataFrame,
    min_names: int = 30,
) -> pd.DataFrame:
    m = ic_a.merge(ic_b, on="as_of", suffixes=("_a", "_b"))
    ok = m["sufficient_a"] & m["sufficient_b"]
    m["delta_ic"] = np.where(ok, m["ic_b"] - m["ic_a"], np.nan)
    m["sufficient"] = ok
    m["n"] = np.minimum(m["n_a"], m["n_b"])
    return m


def summarize_ics(weekly: pd.DataFrame, value_col: str = "ic") -> dict:
    s = weekly.loc[weekly.get("sufficient", True) == True, value_col].dropna()
    if s.empty:
        return {"mean": np.nan, "std": np.nan, "n_weeks": 0, "pos_share": np.nan}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "n_weeks": int(len(s)),
        "pos_share": float((s > 0).mean()),
    }


def non_overlapping_subsample(
    weekly: pd.DataFrame,
    value_col: str = "ic",
    step: int = 4,
) -> pd.DataFrame:
    """Take every ``step``-th sufficient week to reduce 4w label overlap."""
    w = weekly.loc[weekly.get("sufficient", True) == True].sort_values("as_of")
    return w.iloc[::step].copy()


def block_bootstrap_mean(
    values: pd.Series,
    block_weeks: int = 8,
    reps: int = 200,
    seed: int = 42,
) -> dict:
    """Block bootstrap CI for the mean of a weekly series (drops NaN)."""
    x = values.dropna().to_numpy()
    n = len(x)
    if n == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0, "block_weeks": block_weeks}
    rng = np.random.default_rng(seed)
    block = max(1, min(block_weeks, n))
    means = []
    n_blocks = int(np.ceil(n / block))
    for _ in range(reps):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means.append(sample.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "n": n, "block_weeks": block_weeks,
    }


def regression_error_metrics(
    df: pd.DataFrame,
    pred_col: str,
    y_col: str = "y_excess_1w",
) -> dict:
    sub = df[[pred_col, y_col]].dropna()
    if sub.empty:
        return {"mae": np.nan, "rmse": np.nan, "n": 0}
    err = sub[pred_col] - sub[y_col]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "n": int(len(sub)),
    }


def top_decile_metrics(
    df: pd.DataFrame,
    pred_col: str,
    y_col: str = "y_excess_1w",
    frac: float = 0.2,
) -> dict:
    rows = []
    for _, g in df.groupby("as_of", sort=True):
        sub = g[[pred_col, y_col]].dropna()
        if len(sub) < 5:
            continue
        k = max(1, int(len(sub) * frac))
        top = sub.nlargest(k, pred_col)
        rows.append({
            "mean_excess": float(top[y_col].mean()),
            "outperform_rate": float((top[y_col] > 0).mean()),
        })
    if not rows:
        return {"mean_excess_top": np.nan, "outperform_rate_top": np.nan, "n_weeks": 0}
    r = pd.DataFrame(rows)
    return {
        "mean_excess_top": float(r["mean_excess"].mean()),
        "outperform_rate_top": float(r["outperform_rate"].mean()),
        "n_weeks": int(len(r)),
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) == 0:
        return {"brier": np.nan, "roc_auc": np.nan, "balanced_accuracy": np.nan,
                "positive_rate": np.nan, "n": 0}
    pred_class = (y_prob >= 0.5).astype(int)
    out = {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    except ValueError:
        out["roc_auc"] = np.nan
    try:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred_class))
    except ValueError:
        out["balanced_accuracy"] = np.nan
    return out


def coverage_metrics(
    predictions: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    pred_col: str = "pred_m1",
) -> dict:
    n_pred = len(predictions)
    n_scored = len(scored)
    return {
        "prediction_rate": 1.0 if n_pred else 0.0,
        "scorable_rate": n_scored / n_pred if n_pred else np.nan,
        "n_predictions": n_pred,
        "n_scored": n_scored,
        "n_missing_label": int(predictions["label_status"].ne("ok").sum())
        if "label_status" in predictions.columns else np.nan,
    }


def stability_breakdown(
    df: pd.DataFrame,
    pred_col: str,
    y_col: str = "y_excess_1w",
    group_col: str = "sparsity_tier",
    min_names: int = 30,
) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for grp_val, g in df.groupby(group_col, dropna=False):
        ic = weekly_rank_ic(g.rename(columns={pred_col: "pred"}), min_names=min_names)
        s = summarize_ics(ic)
        rows.append({"group": grp_val, **s})
    return pd.DataFrame(rows)


def year_breakdown(weekly: pd.DataFrame, value_col: str = "delta_ic") -> pd.DataFrame:
    w = weekly.copy()
    w["year"] = pd.to_datetime(w["as_of"]).dt.year
    rows = []
    for year, g in w.groupby("year"):
        s = g.loc[g.get("sufficient", True) == True, value_col].dropna()
        rows.append({
            "year": int(year),
            "mean": float(s.mean()) if len(s) else np.nan,
            "n_weeks": int(len(s)),
            "pos_share": float((s > 0).mean()) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)
