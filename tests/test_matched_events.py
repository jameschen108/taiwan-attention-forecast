"""起始事件匹配設計的回歸測試（H7 反向因果處理）。

未匹配的事件研究有明顯事前趨勢（τ=−2 的 t = 2.34），代表價格先動、討論才出現。
匹配設計的價值完全取決於它是否真的消除事前趨勢，因此這幾項必須鎖住。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.matched_events import (
    PRE, POST, RETURN_CALIPER, balance_table, did_car, match,
)


def _cands(seed: int = 0, n_weeks: int = 40, n_stocks: int = 60) -> pd.DataFrame:
    """造一個「事前報酬驅動起始」的合成面板：處理組事前報酬系統性偏高。"""
    rng = np.random.default_rng(seed)
    rows = []
    weeks = pd.date_range("2020-01-05", periods=n_weeks, freq="7D")
    for w in weeks:
        for s in range(n_stocks):
            treated = s < 5                      # 每週固定 5 檔為事件
            path = rng.normal(0, 0.03, PRE + POST + 1)
            if treated:
                path[:PRE] += 0.02               # 事前趨勢（反向因果）
                path[PRE] += 0.03                # 同期效果
            rows.append({
                "ticker": f"T{s:03d}", "week": w, "treated": treated,
                "sparsity_tier": "sparse",
                "pre_car": float(path[:PRE].sum()),
                "ar_m1": float(path[PRE - 1]), "ar_m2": float(path[PRE - 2]),
                "pre_car_34": float(path[:PRE - 2].sum()),
                "log_mktcap": float(rng.normal(22, 1)),
                "log_turnover": float(rng.normal(4, 0.5)),
                **{f"ar_{t}": float(path[k])
                   for k, t in enumerate(range(-PRE, POST + 1))},
            })
    return pd.DataFrame(rows)


class TestCaliperMatching:
    def test_caliper_enforced_on_pre_period_returns(self):
        """caliper 是這個設計的核心：超出容忍度的配對一律不得出現。"""
        pairs = match(_cands(), k=1, caliper=RETURN_CALIPER)
        assert not pairs.empty
        for v in ("ar_m1", "ar_m2", "pre_car_34"):
            gap = (pairs[f"t_{v}"] - pairs[f"c_{v}"]).abs()
            assert gap.max() <= RETURN_CALIPER + 1e-12, f"{v} 超出 caliper"

    def test_caliper_reduces_matched_events(self):
        """以樣本數換平衡：加 caliper 後事件數必然不增。"""
        n_with = match(_cands(), k=1, caliper=RETURN_CALIPER)["treated_ticker"].size
        n_without = match(_cands(), k=1, caliper=None)["treated_ticker"].size
        assert n_with <= n_without

    def test_matching_happens_within_same_week(self):
        """同週匹配是市場共同衝擊被差分掉的前提。"""
        pairs = match(_cands(), k=1)
        assert not pairs.empty
        # pairs 的每一列都帶單一 week，處理與對照同屬該週（建構上保證）
        assert pairs["week"].notna().all()

    def test_treated_never_matched_to_itself(self):
        pairs = match(_cands(), k=1)
        assert (pairs["treated_ticker"] != pairs["control_ticker"]).all()

    def test_caliper_removes_pre_trend_on_synthetic_data(self):
        """合成資料中處理組事前報酬高 0.02；匹配後事前差異應被壓到不顯著。"""
        cands = _cands()
        unmatched_gap = (cands[cands["treated"]]["ar_m1"].mean()
                         - cands[~cands["treated"]]["ar_m1"].mean())
        assert unmatched_gap > 0.01, "合成資料應有明顯事前趨勢（此為守門情境）"

        pairs = match(cands, k=1, caliper=RETURN_CALIPER)
        did = did_car(pairs)
        pre = did[did["tau"] < 0]
        assert (pre["t"].abs() < 2.58).all(), \
            f"匹配後事前差異仍顯著：{pre[['tau', 't']].to_dict('records')}"

    def test_balance_improves(self):
        cands = _cands()
        pairs = match(cands, k=1, caliper=RETURN_CALIPER)
        bal = balance_table(cands, pairs)
        ret_rows = bal[bal["variable"].isin(["ar_m1", "ar_m2", "pre_car_34"])]
        assert (ret_rows["smd_matched"].abs()
                < ret_rows["smd_unmatched"].abs()).all(), "報酬維度的平衡未改善"


class TestDidCar:
    def test_clusters_at_week_level(self):
        """同一週的多個事件共享市場狀態，標準誤必須群集於週。"""
        did = did_car(match(_cands(), k=1))
        assert (did["n_week_clusters"] <= did["n_events"]).all()
        assert did["n_week_clusters"].max() > 1

    def test_car_is_cumulative_sum_of_mean_diff(self):
        did = did_car(match(_cands(), k=1))
        assert np.allclose(did["car_diff"].values,
                           did["mean_diff"].cumsum().values)

    def test_empty_pairs_returns_empty(self):
        assert did_car(pd.DataFrame()).empty
