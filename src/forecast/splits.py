"""Time-based train/validation/test splits by as_of week."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecast.time_contract import is_mature_label


@dataclass
class OuterYearPlan:
    year: int
    year_start: pd.Timestamp  # first as_of in outer test year
    train_cutoff: pd.Timestamp  # exclusive upper bound for initial train labels


def outer_year_plans(years: list[int]) -> list[OuterYearPlan]:
    plans = []
    for y in years:
        year_start = pd.Timestamp(f"{y}-01-01")
        # First Monday on/after Jan 1
        while year_start.weekday() != 0:
            year_start += pd.Timedelta(days=1)
        plans.append(OuterYearPlan(
            year=y,
            year_start=year_start,
            train_cutoff=year_start,
        ))
    return plans


def as_of_list(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(df["as_of"]).dropna().unique())


def filter_mature(df: pd.DataFrame, fit_cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep rows whose labels are known strictly before fit_cutoff."""
    mask = []
    for _, row in df[["label_end_at", "label_available_at"]].iterrows():
        if pd.isna(row["label_end_at"]) or pd.isna(row["label_available_at"]):
            mask.append(False)
        else:
            mask.append(is_mature_label(row["label_end_at"], row["label_available_at"], fit_cutoff))
    out = df.loc[np.asarray(mask)].copy()
    out = out[out["label_status"] == "ok"]
    out = out[out["y_excess_1w"].notna()]
    return out


def filter_mature_fast(df: pd.DataFrame, fit_cutoff: pd.Timestamp) -> pd.DataFrame:
    fit_cutoff = pd.Timestamp(fit_cutoff)
    out = df[
        (df["label_end_at"] < fit_cutoff)
        & (df["label_available_at"] < fit_cutoff)
        & (df["label_status"] == "ok")
        & (df["y_excess_1w"].notna())
        & (df["as_of"] < fit_cutoff)
    ].copy()
    return out


def inner_forward_blocks(
    train_df: pd.DataFrame,
    n_blocks: int = 3,
    block_weeks: int = 26,
    min_train_weeks: int = 104,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (train_end, valid_start, valid_end) cutoffs on as_of axis.

    Each validation block is `block_weeks` distinct as_of values; training uses
    as_of < valid_start with at least min_train_weeks distinct as_of.
    Blocks are the last n_blocks non-overlapping 26-week windows before outer year.
    """
    weeks = as_of_list(train_df)
    if len(weeks) < min_train_weeks + block_weeks:
        return []

    blocks: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    end_idx = len(weeks)
    for _ in range(n_blocks):
        valid_end_idx = end_idx
        valid_start_idx = end_idx - block_weeks
        if valid_start_idx < min_train_weeks:
            break
        train_end = weeks[valid_start_idx]  # exclusive in filter via as_of < valid_start
        valid_start = weeks[valid_start_idx]
        valid_end = weeks[valid_end_idx - 1] + pd.Timedelta(days=1)  # exclusive upper
        blocks.append((train_end, valid_start, valid_end))
        end_idx = valid_start_idx

    blocks.reverse()  # chronological
    return blocks


def prediction_as_ofs_for_year(
    all_as_ofs: list[pd.Timestamp],
    year: int,
) -> list[pd.Timestamp]:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    return [a for a in all_as_ofs if start <= a < end]


def refit_schedule(as_ofs: list[pd.Timestamp], every: int = 4) -> set[pd.Timestamp]:
    return {a for i, a in enumerate(as_ofs) if i % every == 0}
