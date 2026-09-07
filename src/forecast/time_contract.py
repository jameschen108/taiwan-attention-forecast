"""Time contract helpers for forecast as_of / label maturity."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.features.sessions import week_of


def as_of_from_feature_week(feature_week: pd.Timestamp) -> pd.Timestamp:
    """Information cutoff for feature week ending Sunday: next Monday 00:00.

    Spec: Asia/Taipei Monday 00:00 exclusive — only data strictly before as_of.
    Feature week W-SUN ends Sunday; as_of is the following Monday.
    """
    week = pd.Timestamp(feature_week).normalize()
    return week + pd.Timedelta(days=1)


def feature_week_from_as_of(as_of: pd.Timestamp) -> pd.Timestamp:
    """Inverse of as_of_from_feature_week for Monday cutoffs."""
    ts = pd.Timestamp(as_of).normalize()
    # Monday as_of -> previous Sunday feature week
    if ts.weekday() != 0:
        raise ValueError(f"as_of must be Monday 00:00 normalized; got weekday={ts.weekday()}")
    return ts - pd.Timedelta(days=1)


def entry_at_for_feature_week(
    feature_week: pd.Timestamp,
    trading_days_sorted: list[dt.date],
) -> pd.Timestamp | pd.NaT:
    """First market session in the next calendar week (return week)."""
    import bisect

    monday = (pd.Timestamp(feature_week) + pd.Timedelta(days=1)).date()
    sunday = (pd.Timestamp(feature_week) + pd.Timedelta(days=7)).date()
    idx = bisect.bisect_left(trading_days_sorted, monday)
    if idx >= len(trading_days_sorted):
        return pd.NaT
    day = trading_days_sorted[idx]
    if day > sunday:
        return pd.NaT  # schedule_skip: return week fully closed
    return pd.Timestamp(day)


def label_end_1w(
    feature_week: pd.Timestamp,
    trading_days_sorted: list[dt.date],
) -> pd.Timestamp | pd.NaT:
    """Last market session in the next calendar week."""
    import bisect

    monday = (pd.Timestamp(feature_week) + pd.Timedelta(days=1)).date()
    sunday = (pd.Timestamp(feature_week) + pd.Timedelta(days=7)).date()
    idx = bisect.bisect_right(trading_days_sorted, sunday) - 1
    if idx < 0:
        return pd.NaT
    day = trading_days_sorted[idx]
    if day < monday:
        return pd.NaT
    return pd.Timestamp(day)


def label_available_at(label_end: pd.Timestamp | pd.NaT) -> pd.Timestamp | pd.NaT:
    """Under retrospective_proxy: label usable after label_end day closes.

    Conservative: next calendar day 00:00 after label_end.
    """
    if pd.isna(label_end):
        return pd.NaT
    return pd.Timestamp(label_end).normalize() + pd.Timedelta(days=1)


def is_usable_observation(event_at: pd.Timestamp, available_at: pd.Timestamp,
                          as_of: pd.Timestamp) -> bool:
    """event_at < as_of AND available_at < as_of."""
    return (pd.Timestamp(event_at) < pd.Timestamp(as_of)
            and pd.Timestamp(available_at) < pd.Timestamp(as_of))


def is_mature_label(label_end_at: pd.Timestamp, label_available_at_: pd.Timestamp,
                    fit_cutoff: pd.Timestamp) -> bool:
    return (pd.Timestamp(label_end_at) < pd.Timestamp(fit_cutoff)
            and pd.Timestamp(label_available_at_) < pd.Timestamp(fit_cutoff))


def week_ends_on_or_before(as_of: pd.Timestamp) -> pd.Timestamp:
    """Latest W-SUN feature week fully known before as_of."""
    as_of = pd.Timestamp(as_of).normalize()
    # If as_of is Monday, feature week is previous Sunday.
    candidate = week_of(as_of - pd.Timedelta(days=1))
    if as_of_from_feature_week(candidate) > as_of:
        candidate = candidate - pd.Timedelta(days=7)
    return candidate
