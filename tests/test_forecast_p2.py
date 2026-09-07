"""P2: 4w label continuity and backtest accounting tests."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostSchedule, load_cost_schedule
from src.backtest.engine import BacktestState, simulate_weekly_rank_strategy
from src.forecast.labels import _period_adj_return_from_group
from src.forecast.time_contract import label_end_1w, label_end_4w


TD = sorted([
    dt.date(2024, 5, 6), dt.date(2024, 5, 7), dt.date(2024, 5, 8),
    dt.date(2024, 5, 9), dt.date(2024, 5, 10),
    dt.date(2024, 5, 13), dt.date(2024, 5, 14), dt.date(2024, 5, 15),
    dt.date(2024, 5, 16), dt.date(2024, 5, 17),
    dt.date(2024, 5, 20), dt.date(2024, 5, 21), dt.date(2024, 5, 22),
    dt.date(2024, 5, 23), dt.date(2024, 5, 24),
    dt.date(2024, 5, 27), dt.date(2024, 5, 28), dt.date(2024, 5, 29),
    dt.date(2024, 5, 30), dt.date(2024, 5, 31),
])


class Test4wLabels:
    def test_4w_end_after_1w_end(self):
        week = pd.Timestamp("2024-05-05")  # Sunday
        e1 = label_end_1w(week, TD)
        e4 = label_end_4w(week, TD)
        assert e4 > e1

    def test_period_return_not_single_week_fwd(self):
        """4w hold spans more than one week of prices."""
        prices = pd.DataFrame({
            "ticker": ["1101"] * 10,
            "date": pd.date_range("2024-05-06", periods=10, freq="B"),
            "adj_open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "adj_close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        })
        r1 = _period_adj_return_from_group(prices, pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-10"))
        r4 = _period_adj_return_from_group(prices, pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-24"))
        assert r4 > r1


class TestBacktestAccounting:
    def test_cost_schedule_loads(self, tmp_path):
        p = tmp_path / "costs.csv"
        p.write_text("effective_from,fee_rate,fee_discount,tax_rate_sell,slippage_bps,min_fee_twd,lot_size\n"
                     "2020-01-01,0.001425,0.6,0.003,20,20,1000\n")
        cs = load_cost_schedule(p)
        assert cs.lot_size == 1000
        assert cs.effective_fee_rate == pytest.approx(0.001425 * 0.6)

    def test_buy_fee_respects_minimum(self):
        cs = CostSchedule(0.001425, 0.6, 0.003, 20, 20, 1000)
        assert cs.buy_fee(1000) == 20

    def test_nav_non_negative_cash_after_empty_week(self):
        state = BacktestState(cash=1_000_000)
        assert state.cash == 1_000_000
        assert state.positions == {}

    def test_simulate_records_initial_nav(self, tmp_path):
        preds = pd.DataFrame({
            "as_of": [pd.Timestamp("2024-05-06")],
            "ticker": ["1101"],
            "pred_m1": [0.5],
            "sparsity_tier": ["dense"],
        })
        daily = pd.DataFrame({
            "ticker": ["1101"] * 25,
            "date": pd.date_range("2024-04-01", periods=25, freq="B"),
            "adj_open": [10.0] * 25,
            "adj_close": [10.5] * 25,
            "value": [1e7] * 25,
        })
        labels = pd.DataFrame({
            "as_of": [pd.Timestamp("2024-05-06")], "ticker": ["1101"],
            "horizon": ["1w"], "entry_at": [pd.Timestamp("2024-05-06")],
            "label_end_at": [pd.Timestamp("2024-05-10")], "label_status": ["ok"],
        })
        cost = tmp_path / "c.csv"
        cost.write_text("effective_from,fee_rate,fee_discount,tax_rate_sell,slippage_bps,min_fee_twd,lot_size\n"
                        "2020-01-01,0.001425,0.6,0.003,20,20,1000\n")
        out = simulate_weekly_rank_strategy(preds, daily, labels, cost, sparsity_tiers=["dense"])
        assert out["metrics"]["initial_nav"] == pytest.approx(1_000_000)
        assert out["nav"].iloc[0]["nav"] == pytest.approx(1_000_000)
