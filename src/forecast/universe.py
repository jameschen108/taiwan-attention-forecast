"""Point-in-time universe snapshots for forecast pipeline (P4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.forecast.config import load_forecast_config, resolve_path


def _root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_root"])


def load_universe_base(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_forecast_config()
    path = _root(cfg) / "data" / "external" / "universe.csv"
    uni = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["listing_date", "delisting_date"])
    return uni


def load_venue_history(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_forecast_config()
    path = _root(cfg) / "data" / "external" / "market_venue.csv"
    ven = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["effective_from", "effective_to"])
    return ven


def members_at_as_of(
    uni: pd.DataFrame,
    ven: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    evidence_level: str = "ex_post_fixed",
) -> pd.DataFrame:
    """Return tickers active and tradable on as_of under fixed ex-post universe rules."""
    as_of = pd.Timestamp(as_of)
    rows = []
    for _, u in uni.iterrows():
        ticker = str(u["ticker"])
        listed = pd.Timestamp(u["listing_date"]) if pd.notna(u["listing_date"]) else pd.NaT
        delisted = pd.Timestamp(u["delisting_date"]) if pd.notna(u.get("delisting_date")) else pd.NaT
        if pd.notna(listed) and listed > as_of:
            status = "pre_listing"
        elif pd.notna(delisted) and delisted <= as_of:
            status = "delisted"
        else:
            status = "active"
        v = ven[ven["ticker"] == ticker]
        venue = u.get("venue", "TWSE")
        if not v.empty:
            ok = v[
                (v["effective_from"] <= as_of)
                & (v["effective_to"].isna() | (v["effective_to"] > as_of))
            ]
            if not ok.empty:
                venue = ok.iloc[-1]["venue"]
        rows.append({
            "ticker": ticker,
            "as_of": as_of,
            "status": status,
            "sector": u.get("sector"),
            "listing_date": listed,
            "delisting_date": delisted,
            "venue": venue,
            "evidence_level": evidence_level,
            "universe_mode": "existing_ex_post_universe",
        })
    return pd.DataFrame(rows)


def build_universe_snapshots(
    as_ofs: list[pd.Timestamp],
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = cfg or load_forecast_config()
    uni = load_universe_base(cfg)
    ven = load_venue_history(cfg)
    evidence = cfg.get("universe", {}).get("development_mode", "existing_ex_post_universe")
    frames = [members_at_as_of(uni, ven, a, evidence_level=evidence) for a in as_ofs]
    out = pd.concat(frames, ignore_index=True)
    out_path = resolve_path(cfg, "output_dir") / "universe_snapshots.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def active_tickers(snapshot: pd.DataFrame, as_of: pd.Timestamp) -> set[str]:
    sub = snapshot[(snapshot["as_of"] == pd.Timestamp(as_of)) & (snapshot["status"] == "active")]
    return set(sub["ticker"].astype(str))
