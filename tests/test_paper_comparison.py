"""與原論文對照表的回歸測試（T13）。

對照表的價值完全取決於「兩邊的數字真的可比」。這裡鎖住三件事：
論文基準不被誤改、比值只在可比時才計算、推論標準的並列不被弄反。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.paper_comparison import PAPER, _stars, _t

TABLES = Path("output/tables")
pytestmark = pytest.mark.skipif(
    not (TABLES / "T13a_paper_comparison.csv").exists(),
    reason="需要已產出的 T13")


class TestPaperBenchmarks:
    """論文基準是從 PDF 逐項抄錄的，改動必須是刻意的。"""

    def test_key_benchmarks_unchanged(self):
        assert PAPER["ret_weekend_beta"] == 0.0068      # Table 3a 規格(4)
        assert PAPER["ret_weekend_se"] == 0.0021
        assert PAPER["ret_weekday_beta"] == 0.0001
        assert PAPER["corr_all_weekday"] == 0.9028      # Table 2
        assert PAPER["corr_all_weekend"] == 0.3981
        assert PAPER["n_obs_return"] == 255_059

    def test_paper_weekend_is_significant_weekday_is_not(self):
        """論文的核心對比：週末顯著、週間不顯著。抄錯數字會破壞這個關係。"""
        assert abs(_t(PAPER["ret_weekend_beta"], PAPER["ret_weekend_se"])) > 2.576
        assert abs(_t(PAPER["ret_weekday_beta"], PAPER["ret_weekday_se"])) < 1.645

    def test_paper_roi_pattern(self):
        """論文用來排除價格壓力的證據：週間 → 訂單失衡顯著，週末不顯著。"""
        assert abs(_t(PAPER["roi_weekday_beta"], PAPER["roi_weekday_se"])) > 2.576
        assert abs(_t(PAPER["roi_weekend_beta"], PAPER["roi_weekend_se"])) < 1.645

    def test_no_net_of_cost_benchmark(self):
        """論文未報告成本後報酬——這是本研究的實質差異之一，不得憑空補上。"""
        assert PAPER["portfolio_net_annual"] is None


class TestComparabilityGuard:
    """比值只能在建構相同、尺度相同、論文端顯著時出現。"""

    @pytest.fixture(scope="class")
    def comp(self):
        return pd.read_csv(TABLES / "T13a_paper_comparison.csv")

    def test_ratio_absent_when_paper_estimate_insignificant(self, comp):
        """論文週間係數 0.0001（t≈0.05），比值會爆到 53 倍，必須留空。"""
        row = comp[comp["item"].str.contains("H1 週間")].iloc[0]
        assert pd.isna(row["ratio_ours_to_paper"])

    def test_ratio_absent_for_different_constructs(self, comp):
        """訂單失衡與周轉率兩邊是不同測度，不得給比值。"""
        for kw in ("訂單失衡", "異常周轉率"):
            sub = comp[comp["item"].str.contains(kw)]
            assert sub["ratio_ours_to_paper"].isna().all(), kw
            assert sub["scale"].str.contains("不可直接比").all(), kw

    def test_ratio_present_for_comparable_items(self, comp):
        """相關係數與 H1 週末是同建構同尺度，應有比值。"""
        for kw in ("相關性 corr", "H1 週末"):
            sub = comp[comp["item"].str.contains(kw)]
            assert sub["ratio_ours_to_paper"].notna().any(), kw

    def test_every_row_cites_a_source(self, comp):
        assert comp["paper_source"].notna().all()
        assert (comp["paper_source"].str.len() > 0).all()


class TestInferenceSensitivity:
    @pytest.fixture(scope="class")
    def inf(self):
        return pd.read_csv(TABLES / "T13b_inference_sensitivity.csv")

    def test_both_inference_standards_present(self, inf):
        assert inf["inference"].nunique() == 2

    def test_point_estimates_identical_across_clustering(self, inf):
        """cluster 方式只影響標準誤，不影響係數。若係數也變了代表實作有誤。"""
        for w in inf["window"].unique():
            b = inf.loc[inf["window"] == w, "beta"].round(10).unique()
            assert len(b) == 1, f"{w} 的係數不應隨 cluster 方式改變"

    def test_two_way_clustering_is_more_conservative(self, inf):
        """雙重 cluster 的標準誤必須不小於個股 cluster，否則升級沒有意義。"""
        we = inf[inf["window"] == "週末"].set_index("inference")
        two = we.loc["本研究主規格：個股＋週雙重 cluster", "se"]
        one = we.loc["論文做法：僅 cluster 至個股", "se"]
        assert two >= one


def test_stars_thresholds():
    assert _stars(3.0) == "***"
    assert _stars(2.0) == "**"
    assert _stars(1.7) == "*"
    assert _stars(1.0) == ""
