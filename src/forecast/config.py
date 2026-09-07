"""Load forecast configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORECAST_CFG = ROOT / "config" / "forecast.yaml"
DEFAULT_WHITELIST = ROOT / "config" / "forecast_feature_whitelist.yaml"


def load_forecast_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_FORECAST_CFG
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(cfg_path)
    cfg["_root"] = str(ROOT)
    return cfg


def load_whitelist(path: Path | None = None) -> dict[str, Any]:
    wl_path = Path(path) if path is not None else DEFAULT_WHITELIST
    return yaml.safe_load(wl_path.read_text(encoding="utf-8"))


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path under cfg['data'] or cfg['paths'] relative to project root."""
    root = Path(cfg.get("_root", ROOT))
    if key in cfg.get("data", {}):
        return root / cfg["data"][key]
    if key in cfg.get("paths", {}):
        return root / cfg["paths"][key]
    raise KeyError(key)
