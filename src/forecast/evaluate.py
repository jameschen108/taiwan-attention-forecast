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
    """Per-as_of Spearman Rank IC; weeks with < min_names are NaN and flagged.

    Constant predictions within a week yield undefined Spearman → ic=NaN,
    sufficient=False (so M0 constant baseline is not scored via Rank IC).
    """
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
    # Common sufficient weeks
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
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    block = max(1, min(block_weeks, n))
    means = []
    n_blocks = int(np.ceil(n / block))
    for _ in range(reps):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means.append(sample.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


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
