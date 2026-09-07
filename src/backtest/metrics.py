"""Portfolio performance metrics from NAV series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nav_metrics(nav: pd.DataFrame, cash_col: str = "nav") -> dict:
    """Compute total return, max drawdown, turnover from daily/weekly NAV."""
    s = nav[cash_col].astype(float)
    if len(s) < 2:
        return {"total_return": np.nan, "max_drawdown": np.nan, "n_periods": len(s)}
    total_return = float(s.iloc[-1] / s.iloc[0] - 1.0)
    peak = s.cummax()
    dd = (s - peak) / peak
    max_dd = float(dd.min())
    rets = s.pct_change().dropna()
    ann_vol = float(rets.std(ddof=1) * np.sqrt(52)) if len(rets) > 1 else np.nan
    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "ann_vol_weekly": ann_vol,
        "n_periods": int(len(s)),
        "final_nav": float(s.iloc[-1]),
        "initial_nav": float(s.iloc[0]),
    }


def compare_strategies(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append({"strategy": name, **m})
    return pd.DataFrame(rows)
