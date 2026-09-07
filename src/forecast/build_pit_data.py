"""Build forecast PIT daily + panel (no bfill, post-listing attention warm-up)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.features.build import build_panel_forecast
from src.forecast.config import load_forecast_config, resolve_path
from src.market.normalize import build_daily_panel, build_trading_calendar, load_prices


def build_pit_data(cfg: dict | None = None) -> dict:
    t0 = time.time()
    cfg = cfg or load_forecast_config()
    root = Path(cfg["_root"])
    data_cfg = cfg["data"]
    permit_bfill = bool(data_cfg.get("permit_backward_fill", False))

    forecast_dir = resolve_path(cfg, "output_dir")
    forecast_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = root / "audit" / "forecast"
    audit_dir.mkdir(parents=True, exist_ok=True)

    raw_root = root / "data" / "raw" / "finmind"
    interim = root / "data" / "interim"
    t86_dir = root / "data" / "twse" / "t86"

    pit_daily_path = forecast_dir / "market_daily_pit.parquet"
    pit_panel_path = forecast_dir / "panel_pit.parquet"

    # Reuse trading calendar from research interim (calendar facts, not PIT-sensitive)
    trading_days_path = resolve_path(cfg, "trading_days_path")
    if not trading_days_path.exists():
        prices = load_prices(raw_root)
        build_trading_calendar(prices, trading_days_path)

    build_daily_panel(
        raw_root,
        forecast_dir,
        audit_dir,
        t86_dir=t86_dir if t86_dir.exists() else None,
        exrights_csv=interim / "ex_rights.csv",
        shareholding_csv=interim / "shareholding.csv",
        reduction_csv=interim / "capital_reductions.csv",
        permit_backward_fill=permit_bfill,
    )
    # build_daily_panel writes market_daily.parquet under forecast_dir; rename
    built = forecast_dir / "market_daily.parquet"
    if built.exists():
        built.rename(pit_daily_path)

    build_panel_forecast(
        interim / "ptt_matches.parquet",
        pit_daily_path,
        root / "data" / "external" / "universe.csv",
        trading_days_path,
        root / "config" / "settings.yaml",
        pit_panel_path,
        audit_dir,
        post_listing_attention=True,
    )

    summary = {
        "pit_daily_path": str(pit_daily_path),
        "pit_panel_path": str(pit_panel_path),
        "permit_backward_fill": permit_bfill,
        "post_listing_attention": True,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (forecast_dir / "pit_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forecast PIT data path")
    parser.parse_args()
    build_pit_data()


if __name__ == "__main__":
    main()
