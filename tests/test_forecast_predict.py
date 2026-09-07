"""P3 prospective prediction tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecast.config import load_forecast_config
from src.forecast.predict import (
    append_predictions_immutable,
    predictions_path,
    record_skip,
    score_mature_predictions,
    skips_path,
)


@pytest.fixture
def cfg(tmp_path):
    cfg = load_forecast_config()
    cfg["_root"] = str(tmp_path)
    cfg["prospective"] = {
        "predictions_path": "data/forecast/prospective/predictions.parquet",
        "skips_path": "data/forecast/prospective/skips.csv",
        "scores_path": "data/forecast/prospective/scores.parquet",
        "model_store": "data/forecast/models",
        "baseline_model": "A",
        "min_operational_weeks": 12,
    }
    cfg["data"]["output_dir"] = "data/forecast"
    cfg["data"]["report_dir"] = "output/forecast"
    return cfg


class TestImmutableLedger:
    def test_append_twice_same_key_raises(self, cfg):
        rows = [{
            "run_id": "test_v1",
            "as_of": pd.Timestamp("2024-05-06"),
            "ticker": "1101",
            "horizon": "1w",
            "model_version": "test_v1",
            "pred_m1": 0.1,
        }]
        append_predictions_immutable(rows, cfg)
        with pytest.raises(ValueError, match="immutable"):
            append_predictions_immutable(rows, cfg)

    def test_skip_record_idempotent(self, cfg):
        record_skip(pd.Timestamp("2024-05-06"), "test_reason", cfg)
        record_skip(pd.Timestamp("2024-05-06"), "test_reason", cfg)
        skips = pd.read_csv(skips_path(cfg), parse_dates=["as_of"])
        assert len(skips) == 1


class TestScoreMature:
    def test_scores_do_not_modify_predictions(self, cfg, tmp_path):
        pred_rows = [{
            "run_id": "test_v1",
            "model_version": "test_v1",
            "horizon": "1w",
            "generated_at": "2024-05-06T00:00:00+00:00",
            "as_of": pd.Timestamp("2024-05-06"),
            "ticker": "1101",
            "week": pd.Timestamp("2024-04-28"),
            "pred_m1": 0.2,
            "pred_baseline": 0.2,
            "sparsity_tier": "dense",
        }]
        append_predictions_immutable(pred_rows, cfg)
        before = pd.read_parquet(predictions_path(cfg)).copy()

        out_dir = tmp_path / "data" / "forecast"
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = pd.DataFrame([{
            "ticker": "1101",
            "week": pd.Timestamp("2024-04-28"),
            "as_of": pd.Timestamp("2024-05-06"),
            "horizon": "1w",
            "y_excess_1w": 0.01,
            "y_outperform_1w": 1.0,
            "label_status": "ok",
            "label_end_at": pd.Timestamp("2024-05-10"),
            "label_available_at": pd.Timestamp("2024-05-10"),
        }])
        labels.to_parquet(out_dir / "labels.parquet", index=False)

        score_mature_predictions(cfg)
        after = pd.read_parquet(predictions_path(cfg))
        pd.testing.assert_frame_equal(before, after)
