"""P4 universe, folds, supplemental source, and registered gate tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecast.config import load_forecast_config
from src.forecast.evaluate import registered_gain_gate
from src.forecast.features import columns_for_set, feature_lists_full
from src.forecast.folds import build_folds
from src.forecast.run_p4 import P4_COMPARISONS
from src.forecast.sources.monthly_revenue import latest_revenue_known_before
from src.forecast.universe import members_at_as_of


@pytest.fixture
def cfg(tmp_path):
    cfg = load_forecast_config()
    cfg["_root"] = str(tmp_path)
    cfg["data"]["output_dir"] = "data/forecast"
    (tmp_path / "data" / "external").mkdir(parents=True)
    uni = pd.DataFrame([
        {"ticker": "1101", "name_short": "台泥", "sector": "水泥",
         "listing_date": "1962-02-09", "delisting_date": "", "venue": "TWSE"},
        {"ticker": "9999", "name_short": "新上市", "sector": "其他",
         "listing_date": "2024-06-01", "delisting_date": "", "venue": "TWSE"},
    ])
    uni.to_csv(tmp_path / "data" / "external" / "universe.csv", index=False)
    ven = pd.DataFrame([
        {"ticker": "1101", "venue": "TWSE", "effective_from": "1962-02-09", "effective_to": ""},
        {"ticker": "9999", "venue": "TWSE", "effective_from": "2024-06-01", "effective_to": ""},
    ])
    ven.to_csv(tmp_path / "data" / "external" / "market_venue.csv", index=False)
    return cfg


class TestUniverseSnapshot:
    def test_pre_listing_marked_inactive(self, cfg):
        uni = pd.read_csv(cfg["_root"] + "/data/external/universe.csv", dtype={"ticker": str},
                          parse_dates=["listing_date", "delisting_date"])
        ven = pd.read_csv(cfg["_root"] + "/data/external/market_venue.csv", dtype={"ticker": str},
                          parse_dates=["effective_from", "effective_to"])
        snap = members_at_as_of(uni, ven, pd.Timestamp("2024-05-06"))
        assert snap.loc[snap["ticker"] == "9999", "status"].iloc[0] == "pre_listing"
        assert snap.loc[snap["ticker"] == "1101", "status"].iloc[0] == "active"


class TestRevenuePIT:
    def test_announcement_after_as_of_excluded(self):
        revenue = pd.DataFrame([
            {"ticker": "1101", "announce_date": pd.Timestamp("2024-05-10"),
             "revenue": 1e9, "revenue_yoy": 0.1, "revenue_mom": 0.02},
            {"ticker": "1101", "announce_date": pd.Timestamp("2024-04-10"),
             "revenue": 9e8, "revenue_yoy": 0.05, "revenue_mom": -0.01},
        ])
        latest = latest_revenue_known_before(revenue, pd.Timestamp("2024-05-06"))
        assert latest.iloc[0]["revenue_yoy_latest"] == pytest.approx(0.05)


class TestFolds:
    def test_build_folds_has_train_and_test(self, cfg):
        weeks = pd.date_range("2023-01-02", periods=160, freq="7D")
        rows = []
        for w in weeks:
            rows.append({
                "as_of": w + pd.Timedelta(days=1),
                "week": w,
                "ticker": "1101",
                "y_excess_1w": 0.01,
                "label_status": "ok",
                "label_end_at": w + pd.Timedelta(days=5),
                "label_available_at": w + pd.Timedelta(days=5),
                "sparsity_tier": "dense",
            })
        xy = pd.DataFrame(rows)
        cfg["validation"]["outer_test_years"] = [2024]
        folds = build_folds(xy, cfg)
        assert "train" in set(folds["split"])
        assert "test" in set(folds["split"])


class TestRegisteredGainGate:
    def test_bootstrap_low_fail_blocks_gain(self):
        """D-like case: three gates pass, bootstrap lower bound fails."""
        gate = registered_gain_gate(
            {"mean": 0.0579},
            {"mean": 0.0060},
            {"ci_low": -0.0004, "ci_high": 0.0138},
            year_positive=4,
            n_years=5,
        )
        assert gate["gate_candidate_ic"] is True
        assert gate["gate_mean_delta"] is True
        assert gate["gate_boot_low"] is False
        assert gate["gate_years"] is True
        assert gate["gain"] is False


class TestFeatureSetE:
    def test_e_includes_revenue_and_ptt_without_forbidden(self):
        cols = columns_for_set("E")
        assert "revenue_yoy_latest" in cols
        assert "abn_attention_weekend" in cols
        forbidden = set(feature_lists_full()["forbidden"])
        assert not (set(cols) & forbidden)


class TestP4Comparisons:
    def test_e_uses_d_as_baseline(self):
        e_comp = [c for c in P4_COMPARISONS if c[0] == "E"]
        assert len(e_comp) == 1
        assert e_comp[0][1] == "D"
        assert e_comp[0][2] is False
