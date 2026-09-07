"""Trading cost schedule and fee computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CostSchedule:
    fee_rate: float
    fee_discount: float
    tax_rate_sell: float
    slippage_bps: float
    min_fee_twd: float
    lot_size: int

    @property
    def effective_fee_rate(self) -> float:
        return self.fee_rate * self.fee_discount

    def buy_slippage_multiplier(self, stress: float = 1.0) -> float:
        return 1.0 + (self.slippage_bps / 10000.0) * stress

    def sell_slippage_multiplier(self, stress: float = 1.0) -> float:
        return 1.0 - (self.slippage_bps / 10000.0) * stress

    def buy_fee(self, gross_value: float) -> float:
        return max(gross_value * self.effective_fee_rate, self.min_fee_twd)

    def sell_fee(self, gross_value: float) -> float:
        commission = max(gross_value * self.effective_fee_rate, self.min_fee_twd)
        tax = gross_value * self.tax_rate_sell
        return commission + tax


def load_cost_schedule(path: Path, as_of: pd.Timestamp | None = None) -> CostSchedule:
    df = pd.read_csv(path, parse_dates=["effective_from"])
    df = df.sort_values("effective_from")
    if as_of is not None:
        row = df[df["effective_from"] <= pd.Timestamp(as_of)].iloc[-1]
    else:
        row = df.iloc[-1]
    return CostSchedule(
        fee_rate=float(row["fee_rate"]),
        fee_discount=float(row["fee_discount"]),
        tax_rate_sell=float(row["tax_rate_sell"]),
        slippage_bps=float(row["slippage_bps"]),
        min_fee_twd=float(row["min_fee_twd"]),
        lot_size=int(row["lot_size"]),
    )
