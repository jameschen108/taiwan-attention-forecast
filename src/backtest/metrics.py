"""Portfolio performance metrics from NAV series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nav_metrics(
    nav: pd.DataFrame,
    cash_col: str = "nav",
    initial_cash: float | None = None,
) -> dict:
    """Compute total return, max drawdown, turnover from daily/weekly NAV.

    When ``initial_cash`` is provided, total return is measured from that
    starting capital (spec §11.5), not from the first row of the NAV series.
    """
    s = nav[cash_col].astype(float)
    if s.empty:
        return {"total_return": np.nan, "max_drawdown": np.nan, "n_periods": 0}
    base = float(initial_cash) if initial_cash is not None else float(s.iloc[0])
    final = float(s.iloc[-1])
    total_return = float(final / base - 1.0) if base > 0 else np.nan
    peak = s.cummax()
    dd = (s - peak) / peak.replace(0, np.nan)
    max_dd = float(dd.min()) if len(s) else np.nan
    rets = s.pct_change().dropna()
    ann_vol = float(rets.std(ddof=1) * np.sqrt(52)) if len(rets) > 1 else np.nan
    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "ann_vol_weekly": ann_vol,
        "n_periods": int(len(s)),
        "final_nav": final,
        "initial_nav": base,
    }


def compare_strategies(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append({"strategy": name, **m})
    return pd.DataFrame(rows)
