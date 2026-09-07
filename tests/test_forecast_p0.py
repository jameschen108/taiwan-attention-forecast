"""P0 forecast time contract, labels, splits, inference guards."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features.sessions import attention_week, attention_week_calendar, week_of
from src.forecast.features import assert_no_forbidden, feature_lists, engineer_forecast_features
from src.forecast.labels import build_label_frame
from src.forecast.splits import filter_mature_fast, inner_forward_blocks
from src.forecast.time_contract import (
    as_of_from_feature_week,
    feature_week_from_as_of,
    is_mature_label,
    is_usable_observation,
)
from src.market.normalize import build_daily_panel


TD = [
    dt.date(2024, 5, 6), dt.date(2024, 5, 7), dt.date(2024, 5, 8),
    dt.date(2024, 5, 9), dt.date(2024, 5, 10), dt.date(2024, 5, 13),
]


class TestTimeContract:
    def test_as_of_is_monday_after_sunday_week(self):
        week = pd.Timestamp("2024-05-12")  # Sunday
        as_of = as_of_from_feature_week(week)
        assert as_of == pd.Timestamp("2024-05-13")
        assert as_of.weekday() == 0
        assert feature_week_from_as_of(as_of) == week

    def test_usable_observation_requires_available_before_as_of(self):
        as_of = pd.Timestamp("2024-05-13")
        assert is_usable_observation(
            pd.Timestamp("2024-05-12 10:00"),
            pd.Timestamp("2024-05-12 10:00"),
            as_of,
        )
        assert not is_usable_observation(
            pd.Timestamp("2024-05-12 10:00"),
            pd.Timestamp("2024-05-13 00:00"),
            as_of,
        )

    def test_mature_label_requires_end_and_available_before_fit(self):
        fit = pd.Timestamp("2024-05-20")
        assert is_mature_label(
            pd.Timestamp("2024-05-17"),
            pd.Timestamp("2024-05-18"),
            fit,
        )
        assert not is_mature_label(
            pd.Timestamp("2024-05-20"),
            pd.Timestamp("2024-05-21"),
            fit,
        )


class TestAttentionWeekForecast:
    def test_calendar_week_does_not_need_future_trading_day(self):
        # After last known trading day in TD, research attention_week is None
        late = pd.Timestamp("2024-05-14 20:00")
        assert attention_week(late, TD) is None
        assert attention_week_calendar(late) == week_of(late)


class TestWhitelist:
    def test_forbidden_not_in_a_or_b(self):
        a, b, forbidden = feature_lists()
        assert not (set(a) & set(forbidden))
        assert not (set(b) & set(forbidden))
        assert "ret_next" in forbidden
        assert "y_excess_1w" in forbidden

    def test_assert_no_forbidden_raises(self):
        with pytest.raises(ValueError):
            assert_no_forbidden(["ret", "ret_next"], ["ret_next"])


class TestLabels:
    def test_excess_equals_stock_minus_bench(self):
        panel = pd.DataFrame({
            "ticker": ["1101", "1101"],
            "week": [pd.Timestamp("2024-05-05"), pd.Timestamp("2024-05-12")],
            "ret": [0.01, 0.02],
            "ret_next": [0.02, np.nan],
            "listing_date": [pd.Timestamp("2010-01-01")] * 2,
        })
        bench = pd.DataFrame({
            "week": [pd.Timestamp("2024-05-12"), pd.Timestamp("2024-05-19")],
            "bench_ret": [0.005, 0.01],
            "n_trading_days": [5, 5],
            "first_date": [pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-13")],
            "last_date": [pd.Timestamp("2024-05-10"), pd.Timestamp("2024-05-17")],
        })
        labels = build_label_frame(panel, TD, bench)
        row = labels[labels["week"] == pd.Timestamp("2024-05-05")].iloc[0]
        assert row["y_stock_1w"] == pytest.approx(0.02)
        assert row["y_bench_1w"] == pytest.approx(0.005)
        assert row["y_excess_1w"] == pytest.approx(0.015)
        assert row["label_definition"] == "adjusted_price_proxy"


class TestSplits:
    def test_filter_mature_excludes_future_labels(self):
        df = pd.DataFrame({
            "as_of": [pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-13")],
            "label_end_at": [pd.Timestamp("2024-05-10"), pd.Timestamp("2024-05-17")],
            "label_available_at": [pd.Timestamp("2024-05-11"), pd.Timestamp("2024-05-18")],
            "label_status": ["ok", "ok"],
            "y_excess_1w": [0.01, 0.02],
        })
        cut = pd.Timestamp("2024-05-13")
        out = filter_mature_fast(df, cut)
        assert len(out) == 1
        assert out.iloc[0]["as_of"] == pd.Timestamp("2024-05-06")

    def test_inner_blocks_are_time_ordered(self):
        as_ofs = pd.date_range("2020-01-06", periods=200, freq="7D")
        df = pd.DataFrame({
            "as_of": np.repeat(as_ofs, 3),
            "label_end_at": np.repeat(as_ofs + pd.Timedelta(days=5), 3),
            "label_available_at": np.repeat(as_ofs + pd.Timedelta(days=6), 3),
            "label_status": "ok",
            "y_excess_1w": 0.0,
        })
        blocks = inner_forward_blocks(df, n_blocks=3, block_weeks=26, min_train_weeks=104)
        assert len(blocks) >= 1
        starts = [b[1] for b in blocks]
        assert starts == sorted(starts)


class TestInferenceWithoutLabel:
    def test_latest_week_can_have_features_without_y(self):
        # Synthetic mini panel
        weeks = pd.date_range("2024-01-07", periods=12, freq="7D")
        rows = []
        for t in ("1101", "1102"):
            for i, w in enumerate(weeks):
                rows.append({
                    "ticker": t,
                    "week": w,
                    "ret": 0.01,
                    "ret_lag1": 0.0,
                    "value": 1e7,
                    "market_cap": 1e10,
                    "amihud": 0.1,
                    "zero_volume_days": 0,
                    "turnover": 0.01,
                    "foreign_holding_pct": 10.0,
                    "listing_age_years": 20.0,
                    "att_all": float(i % 3),
                    "att_weekday": 1.0,
                    "att_weekend": 0.0,
                    "att_non_trading": 0.0,
                    "att_high_effort": 0.0,
                    "att_mid_effort": 0.0,
                    "att_low_effort": 0.0,
                    "abn_attention_all": 0.0,
                    "abn_attention_weekday": 0.0,
                    "abn_attention_weekend": 0.0,
                    "att_nonzero_weeks_52": 10.0,
                    "att_mean_level_52": 1.0,
                    "is_initiation": False,
                    "att_zero_base": 0,
                    "is_code_only_matched": 0,
                    "sparsity_tier": "sparse",
                    "listing_date": pd.Timestamp("2000-01-01"),
                    "sector": "水泥",
                })
        panel = pd.DataFrame(rows)
        bench = pd.DataFrame({
            "week": weeks,
            "bench_ret": 0.001,
        })
        feats = engineer_forecast_features(panel, bench)
        last = feats[feats["week"] == weeks[-1]]
        assert len(last) == 2
        assert "log1p_att_all" in last.columns
        # Labels for last week would be missing — features still exist
        assert last["log1p_att_all"].notna().all()


class TestNoBfill:
    def test_permit_backward_fill_false_keeps_leading_nan(self, tmp_path):
        raw = tmp_path / "raw"
        (raw / "price").mkdir(parents=True)
        import json
        rows = [
            {"date": "2024-01-02", "stock_id": "1101", "Trading_Volume": 1000,
             "Trading_money": 10000, "open": 10, "max": 10, "min": 10, "close": 10,
             "Trading_turnover": 1},
            {"date": "2024-01-03", "stock_id": "1101", "Trading_Volume": 1000,
             "Trading_money": 10000, "open": 10, "max": 10, "min": 10, "close": 11,
             "Trading_turnover": 1},
            {"date": "2024-01-04", "stock_id": "1101", "Trading_Volume": 1000,
             "Trading_money": 10000, "open": 11, "max": 11, "min": 11, "close": 11,
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
