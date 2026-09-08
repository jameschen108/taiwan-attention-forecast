"""Export train/validation/test fold assignments (P4)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.forecast.config import load_forecast_config, resolve_path
from src.forecast.splits import (
    as_of_list,
    filter_mature_fast,
    outer_year_plans,
    prediction_as_ofs_for_year,
)


def build_folds(
    xy: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    *,
    horizon: str = "1w",
) -> pd.DataFrame:
    """Materialize outer-year test folds and pre-cutoff train pools."""
    cfg = cfg or load_forecast_config()
    val = cfg["validation"]
    years = list(val["outer_test_years"])
    tiers = cfg["features"].get("train_sparsity_tiers")
    all_as_ofs = as_of_list(xy)
    rows: list[dict] = []

    for plan in outer_year_plans(years):
        year = plan.year
        train = filter_mature_fast(xy, plan.train_cutoff)
        if tiers and "sparsity_tier" in train.columns:
            train = train[train["sparsity_tier"].isin(tiers)]
        test_as_ofs = prediction_as_ofs_for_year(all_as_ofs, year)
        test = xy[xy["as_of"].isin(test_as_ofs)].copy()
        if tiers and "sparsity_tier" in test.columns:
            test = test[test["sparsity_tier"].isin(tiers)]

        fold_id = f"outer_{year}"
        for _, row in train.iterrows():
            rows.append({
                "fold_id": fold_id,
                "as_of": row["as_of"],
                "ticker": row["ticker"],
                "horizon": horizon,
                "split": "train",
                "exclude_reason": "",
                "outer_year": year,
            })
        for _, row in test.iterrows():
            reason = ""
            if row.get("label_status") != "ok":
                reason = str(row.get("label_status") or "label_not_ok")
            elif pd.isna(row.get("y_excess_1w")):
                reason = "label_immature_or_missing"
            rows.append({
                "fold_id": fold_id,
                "as_of": row["as_of"],
                "ticker": row["ticker"],
                "horizon": horizon,
                "split": "test",
                "exclude_reason": reason,
                "outer_year": year,
            })

    out = pd.DataFrame(rows)
    path = resolve_path(cfg, "output_dir") / "folds.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out
