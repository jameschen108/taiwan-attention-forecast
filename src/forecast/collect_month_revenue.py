"""Fetch FinMind TaiwanStockMonthRevenue into data/raw/finmind/month_revenue/."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.market.collect_finmind import DATASETS, collect

MONTH_REVENUE = "TaiwanStockMonthRevenue"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect FinMind monthly revenue announcements")
    parser.add_argument("--universe", default="data/external/universe.csv")
    parser.add_argument("--out", default="data/raw/finmind")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    uni_path = root / args.universe
    with open(uni_path, encoding="utf-8") as fh:
        tickers = [row["ticker"].strip() for row in csv.DictReader(fh)]

    datasets = dict(DATASETS)
    datasets["month_revenue"] = MONTH_REVENUE
    # Patch module-level mapping for this run
    import src.market.collect_finmind as finmind_mod
    finmind_mod.DATASETS = datasets

    collect(tickers, root / args.out, kinds=["month_revenue"])


if __name__ == "__main__":
    main()
