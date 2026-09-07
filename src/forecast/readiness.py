"""Forecast readiness flags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_readiness(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_readiness(**kwargs: Any) -> dict[str, Any]:
    base = {
        "run_id": kwargs.get("run_id"),
        "data_ready": kwargs.get("data_ready", False),
        "model_ready": kwargs.get("model_ready", False),
        "eval_ready": kwargs.get("eval_ready", False),
        "trading_ready": False,
        "blockers": kwargs.get("blockers", []),
        "notes": kwargs.get("notes", []),
    }
    return base
