"""Extended forecast correctness tests (spec §17.1)."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import ADV20_MIN_OBS, _adv20, simulate_weekly_rank_strategy
from src.backtest.metrics import nav_metrics
from src.features.attention import build_attention_panel, build_attention_panel_post_listing
from src.features.sessions import week_of
from src.forecast.labels import _count_trading_sessions
from src.market.normalize import build_daily_panel


TD = sorted([
    dt.date(2024, 5, 6), dt.date(2024, 5, 7), dt.date(2024, 5, 8),
    dt.date(2024, 5, 9), dt.date(2024, 5, 10),
    dt.date(2024, 5, 13), dt.date(2024, 5, 14),
])


class TestTradingDaysCount:
    def test_counts_unique_sessions_not_ticker_rows(self):
        n = _count_trading_sessions(
            pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-10"), TD)
        assert n == 5.0


class TestNavFromInitialCash:
    def test_total_return_uses_initial_cash_not_first_mark(self):
        nav = pd.DataFrame({
            "nav": [1_000_000, 992_000, 980_000],
            "as_of": [pd.NaT, pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-13")],
        })
        m = nav_metrics(nav, initial_cash=1_000_000)
        assert m["total_return"] == pytest.approx(-0.02)
        assert m["initial_nav"] == 1_000_000


class TestAdv20MinObs:
    def test_insufficient_history_rejected(self):
        daily = pd.DataFrame({
            "ticker": ["1101"] * 5,
            "date": pd.date_range("2024-05-01", periods=5, freq="B"),
            "value": [1e7] * 5,
        })
        adv, nobs = _adv20(daily, pd.Timestamp("2024-05-10"))
        assert pd.isna(adv["1101"])
        assert nobs["1101"] < ADV20_MIN_OBS


class TestNoBfillProduction:
    def test_build_daily_panel_respects_permit_backward_fill_false(self, tmp_path):
        raw = tmp_path / "raw"
        (raw / "price").mkdir(parents=True)
        rows = [
            {"date": "2024-01-02", "stock_id": "1101", "Trading_Volume": 1000,
             "Trading_money": 10000000, "open": 10, "max": 10, "min": 10, "close": 10,
             "Trading_turnover": 1},
            {"date": "2024-01-03", "stock_id": "1101", "Trading_Volume": 1000,
             "Trading_money": 10000000, "open": 10, "max": 10, "min": 10, "close": 11,
             "Trading_turnover": 1},
            {"date": "2024-01-04", "stock_id": "1101", "Trading_Volume": 1000,
             "Trading_money": 10000000, "open": 11, "max": 11, "min": 11, "close": 11,
             "Trading_turnover": 1},
        ]
        (raw / "price" / "1101.json").write_text(json.dumps({
            "dataset": "TaiwanStockPrice", "data_id": "1101", "data": rows,
        }), encoding="utf-8")
        (raw / "inst").mkdir(parents=True, exist_ok=True)
        (raw / "inst" / "1101.json").write_text(json.dumps({
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": "1101",
            "data": [
                {"date": "2024-01-02", "stock_id": "1101", "name": "Foreign_Investor",
                 "buy": 100, "sell": 50},
            ],
        }), encoding="utf-8")
        sh = tmp_path / "shareholding.csv"
        sh.write_text(
            "ticker,date,foreign_holding_pct,shares_outstanding\n"
            "1101,2024-01-04,5.0,1000\n"
        )
        out = tmp_path / "out"
        audit = tmp_path / "audit"
        daily = build_daily_panel(
            raw, out, audit,
            shareholding_csv=sh,
            permit_backward_fill=False,
        )
        sub = daily[daily["ticker"] == "1101"].sort_values("date")
        assert pd.isna(sub.iloc[0]["foreign_holding_pct"])
        assert pd.isna(sub.iloc[0]["shares_outstanding"])
        assert sub.iloc[2]["shares_outstanding"] == 1000.0


class TestPostListingAttention:
    def test_pre_listing_zeros_not_in_rolling_window(self):
        settings = {
            "attention": {
                "lookback_weeks": 8, "min_periods": 8, "transform": "log1p",
                "sparsity_lookback_weeks": 52, "exclude_bulk_listing": True,
            },
            "sparsity": {"dense_min_nonzero_weeks": 40, "silent_max_nonzero_weeks": 3},
        }
        weeks = pd.date_range("2024-01-07", periods=12, freq="7D")
        listing = pd.Timestamp("2024-02-18")
        matches = pd.DataFrame({
            "ticker": ["1101"], "week": [weeks[-1]],
            "window": ["weekday"], "session": ["intraday"], "effort": ["high_effort"],
        })
        listing_s = pd.Series({"1101": listing})
        pit = build_attention_panel_post_listing(matches, weeks, listing_s, settings)
        full = build_attention_panel(matches, weeks, ["1101"], settings).reset_index()
        full = full[full["week"] >= week_of(listing)]
        # Post-listing panel should not inherit 52w history from pre-listing zeros
        assert pit["att_nonzero_weeks_52"].max() <= full["att_nonzero_weeks_52"].max()


class TestListSelectionInvariant:
    def test_missing_future_label_does_not_change_ranking(self):
        """Top-K from predictions should not depend on other tickers' label gaps."""
        base = pd.DataFrame({
            "as_of": [pd.Timestamp("2024-05-06")] * 2,
            "ticker": ["1101", "1102"],
            "pred_m1": [0.3, 0.2],
            "sparsity_tier": ["dense", "dense"],
            "label_status": ["ok", "ok"],
            "entry_at": [pd.Timestamp("2024-05-06")] * 2,
            "label_end_at": [pd.Timestamp("2024-05-10")] * 2,
        })
        with_extra = pd.concat([base, pd.DataFrame([{
            "as_of": pd.Timestamp("2024-05-06"),
            "ticker": "1103",
            "pred_m1": 0.1,
            "sparsity_tier": "dense",
            "label_status": "entry_unavailable_or_missing_stock",
            "entry_at": pd.Timestamp("2024-05-06"),
            "label_end_at": pd.Timestamp("2024-05-10"),
        }])], ignore_index=True)

        def top2(df):
            ok = df[df["label_status"] == "ok"].sort_values("pred_m1", ascending=False)
            return list(ok.head(2)["ticker"])

        assert top2(base) == ["1101", "1102"]
        assert top2(with_extra) == ["1101", "1102"]
