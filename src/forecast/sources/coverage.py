"""Source availability and coverage reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.forecast.config import load_forecast_config, resolve_path


def write_source_coverage(rows: list[dict], cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_forecast_config()
    df = pd.DataFrame(rows)
    path = resolve_path(cfg, "output_dir") / "source_coverage.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not df.empty:
        prev = pd.read_parquet(path)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
            subset=["source", "as_of"], keep="last",
        )
    df.to_parquet(path, index=False)
    return df


def coverage_row(
    source: str,
    as_of: pd.Timestamp,
    *,
    available: bool,
    n_records: int = 0,
    notes: str = "",
) -> dict:
    return {
        "source": source,
        "as_of": pd.Timestamp(as_of),
        "available": available,
        "n_records": n_records,
        "notes": notes,
    }


def path_exists(root: Path, rel: str) -> bool:
    p = root / rel
    if p.is_file():
        return True
    if p.is_dir():
        return any(p.glob("*"))
    return False
