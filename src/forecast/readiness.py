"""Forecast readiness flags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_readiness(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assess_trading_readiness(
    backtest_metrics: dict[str, dict],
    *,
    max_drawdown_limit: float | None = 0.35,
    require_positive_return: bool = False,
) -> tuple[bool, list[str]]:
    """Apply spec §17.2 trading gate to backtest metric dicts."""
    blockers: list[str] = []
    for name, m in backtest_metrics.items():
        if "ridge" not in name and "benchmark" not in name:
            continue
        dd = m.get("max_drawdown")
        if max_drawdown_limit is not None and dd is not None and abs(dd) > max_drawdown_limit:
            blockers.append(f"{name}: max_drawdown {dd:.2%} exceeds limit {max_drawdown_limit:.0%}")
        if require_positive_return:
            tr = m.get("total_return")
            if tr is not None and tr <= 0:
                blockers.append(f"{name}: total_return {tr:.2%} not positive")
    return len(blockers) == 0, blockers


def default_readiness(**kwargs: Any) -> dict[str, Any]:
    trading_ready = kwargs.get("trading_ready", False)
    blockers = list(kwargs.get("blockers", []))
    return {
        "run_id": kwargs.get("run_id"),
        "data_ready": kwargs.get("data_ready", False),
        "model_ready": kwargs.get("model_ready", False),
        "eval_ready": kwargs.get("eval_ready", False),
        "trading_ready": trading_ready,
        "blockers": blockers,
        "notes": kwargs.get("notes", []),
    }
