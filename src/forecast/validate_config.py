"""Forecast config validation and enforced settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any


UNSUPPORTED_KEYS = {
    "generation_deadline": "scheduling not implemented (P3)",
    "require_snapshot": "universe snapshots not implemented (P4)",
    "hyperparameter_refresh": "annual refresh enforced in code but not separately validated",
    "history_shortfall": "handled per-feature in engineer_forecast_features",
    "attention_lookback_weeks": "uses panel attention settings until wired",
    "sparsity_lookback_weeks": "uses panel attention settings until wired",
    "bulk_listing_max_tickers": "uses settings.yaml attention exclude_bulk_listing",
}


def validate_forecast_config(cfg: dict[str, Any]) -> list[str]:
    """Return list of warnings for declared-but-unimplemented settings."""
    warnings: list[str] = []
    data = cfg.get("data", {})
    if data.get("require_explicit_source_assumptions"):
        path = cfg.get("paths", {}).get("source_assumptions")
        if path and not (Path(cfg["_root"]) / path).exists():
            warnings.append(f"source_assumptions file missing: {path}")

    for key, reason in UNSUPPORTED_KEYS.items():
        if key in cfg.get("forecast", {}) or key in cfg.get("features", {}):
            warnings.append(f"{key}: {reason}")

    bt = cfg.get("backtest", {})
    if bt.get("max_drawdown_limit") is None:
        warnings.append("max_drawdown_limit unset")

    return warnings


def assert_data_paths(cfg: dict[str, Any]) -> None:
    """Fail fast when required PIT paths are missing."""
    root = Path(cfg["_root"])
    panel = root / cfg["data"]["panel_path"]
    daily = root / cfg["data"]["daily_path"]
    missing = [str(p) for p in (panel, daily) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Forecast PIT data missing. Run: python -m src.forecast.build_pit_data\n"
            + "\n".join(missing)
        )
