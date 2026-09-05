"""核心運算邏輯的回歸測試（PRD §6.2 F4）。

「一旦寫錯，整份結果都會錯且不易察覺」——關注度指標、稀疏度分層、起始事件、
名稱碰撞消解、時段判定、訂單失衡、日曆對齊。測試先通過才實作下游。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features.attention import (
    abnormal_attention, sparsity_fields, sparsity_tier,
)
from src.features.imbalance import (
    non_inst_order_imbalance, weekly_non_inst_roi,
)
from src.features.sessions import (
    assign_calendar_window, assign_session_window, attention_week,
    next_trading_day, return_week, week_of, week_trading_day_count,
)
from src.universe.name_matching import Matcher, Variant, normalize


# ---------------------------------------------------------------------------
# 時段判定（PRD §3.5）
# ---------------------------------------------------------------------------

TD = {dt.date(2024, 5, 6), dt.date(2024, 5, 7), dt.date(2024, 5, 8),
      dt.date(2024, 5, 9), dt.date(2024, 5, 10), dt.date(2024, 5, 13)}
TD_SORTED = sorted(TD)


class TestSessions:
    def test_intraday_only_inside_market_hours_on_trading_day(self):
        assert assign_session_window(pd.Timestamp("2024-05-06 09:00"), TD) == "intraday"
        assert assign_session_window(pd.Timestamp("2024-05-06 13:29"), TD) == "intraday"
        # 13:30 收盤，邊界不含
        assert assign_session_window(pd.Timestamp("2024-05-06 13:30"), TD) == "non_trading"
        assert assign_session_window(pd.Timestamp("2024-05-06 08:59"), TD) == "non_trading"

    def test_holiday_weekday_is_non_trading_all_day(self):
        """週一為國定假日時，該日全天屬非交易時段——不可用星期幾判定。"""
        holiday = pd.Timestamp("2024-05-11 10:00")  # 不在 trading_days
        assert assign_session_window(holiday, TD) == "non_trading"

    def test_calendar_window(self):
        assert assign_calendar_window(pd.Timestamp("2024-05-11 10:00")) == "weekend"  # 六
        assert assign_calendar_window(pd.Timestamp("2024-05-12 10:00")) == "weekend"  # 日
        assert assign_calendar_window(pd.Timestamp("2024-05-10 10:00")) == "weekday"

    def test_long_holiday_maps_to_next_open_day_not_wrong_week(self):
        """連假期間的貼文歸屬到下一個開市日，不得落到錯的週。"""
        # 5/10(五) 收盤後 → 下一個開市日 5/13(一)
        assert next_trading_day(pd.Timestamp("2024-05-10 20:00"), TD_SORTED) == dt.date(2024, 5, 13)
        # 5/11(六) → 5/13(一)
        assert next_trading_day(pd.Timestamp("2024-05-11 09:00"), TD_SORTED) == dt.date(2024, 5, 13)

    def test_end_of_sample_returns_none_for_exclusion_list(self):
        """期末收盤後無法指派下一交易日者必須明確排除，不得杜撰交易日。"""
        assert next_trading_day(pd.Timestamp("2024-05-14 20:00"), TD_SORTED) is None
        assert attention_week(pd.Timestamp("2024-05-14 20:00"), TD_SORTED) is None

    def test_week_of_is_sunday_anchored(self):
        # 2024-05-06 是星期一，其 W-SUN 週結束於 2024-05-12（星期日）
        assert week_of(dt.date(2024, 5, 6)) == pd.Timestamp("2024-05-12")
        assert week_of(dt.date(2024, 5, 12)) == pd.Timestamp("2024-05-12")
        assert week_of(dt.date(2024, 5, 13)) == pd.Timestamp("2024-05-19")

    def test_weekend_attention_strictly_leads_return_week(self):
        """週六／日的關注度必須嚴格先於次週一至週五的報酬。"""
        saturday = pd.Timestamp("2024-05-11 15:00")
        fw = attention_week(saturday, TD_SORTED)
        assert fw == pd.Timestamp("2024-05-12")
        rw = return_week(fw)
        assert rw == pd.Timestamp("2024-05-19")
        # 報酬週的第一天（週一）在關注度時點之後
        monday = rw - pd.Timedelta(days=6)
        assert monday > saturday.normalize()

    def test_incomplete_week_detected(self):
        assert week_trading_day_count(TD, pd.Timestamp("2024-05-12")) == 5
        # 只有 5/13 開市的那一週
        assert week_trading_day_count(TD, pd.Timestamp("2024-05-19")) == 1


# ---------------------------------------------------------------------------
# 異常關注度與稀疏度（PRD §3.2、§3.2.1）
# ---------------------------------------------------------------------------

class TestAttention:
    def test_abn_uses_only_past_eight_weeks(self):
        counts = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 10])
        abn = abnormal_attention(counts, lookback=8, min_periods=8)
        assert abn.iloc[:8].isna().all(), "不足 8 週必須維持缺值"
        assert abn.iloc[8] == pytest.approx(np.log1p(10) - 0.0)

    def test_abn_zero_when_at_normal_level(self):
        counts = pd.Series([5] * 9)
        abn = abnormal_attention(counts)
        assert abn.iloc[8] == pytest.approx(0.0)

    def test_log1p_keeps_true_zero(self):
        """PTT 的零是真實的零，不可 mask（與 Google Trends 相反）。"""
        counts = pd.Series([4] * 8 + [0])
        abn = abnormal_attention(counts, transform="log1p")
        assert abn.iloc[8] == pytest.approx(-np.log1p(4))
        # 正值限定版本則視為缺值
        abn_pos = abnormal_attention(counts, transform="log_positive")
        assert pd.isna(abn_pos.iloc[8])

    def test_no_lookahead_in_baseline(self):
        """基準期不得含當期：改變當期值不應改變自己的基準。"""
        a = pd.Series([1] * 8 + [100] + [1])
        b = pd.Series([1] * 8 + [1] + [1])
        assert abnormal_attention(a).iloc[8] != abnormal_attention(b).iloc[8]
        # 但第 9 期的基準受第 8 期影響，這是正確的
        assert not np.isclose(abnormal_attention(a).iloc[9],
                              abnormal_attention(b).iloc[9])

    def test_initiation_requires_all_zero_lookback(self):
        counts = pd.Series([0] * 8 + [3, 5])
        sp = sparsity_fields(counts, lookback_weeks=52, abn_lookback=8)
        assert bool(sp["is_initiation"].iloc[8]) is True
        # 第 9 期前 8 週已有討論，不算起始
        assert bool(sp["is_initiation"].iloc[9]) is False

    def test_att_lookback_all_zero_flag(self):
        counts = pd.Series([0] * 8 + [0, 7])
        sp = sparsity_fields(counts, abn_lookback=8)
        assert bool(sp["att_lookback_all_zero"].iloc[8]) is True
        assert bool(sp["att_lookback_all_zero"].iloc[9]) is True
        assert bool(sp["is_initiation"].iloc[8]) is False  # 當期也是零

    def test_sparsity_tier_boundaries(self):
        s = pd.Series([0, 3, 4, 39, 40, 52], dtype=float)
        tiers = sparsity_tier(s, dense_min=40, silent_max=3)
        assert list(tiers) == ["silent", "silent", "sparse", "sparse", "dense", "dense"]

    def test_sparsity_uses_only_past(self):
        counts = pd.Series([0, 0, 5])
        sp = sparsity_fields(counts, lookback_weeks=52, abn_lookback=2)
        # 第 2 期的非零週數只數前兩週（皆為零）
        assert sp["att_nonzero_weeks_52"].iloc[2] == 0

    def test_silent_stock_not_confused_with_at_normal_level(self):
        """連續零的長尾股與剛好等於常態的台積電，AbnAtt 都是 0，必須靠旗標區分。"""
        silent = pd.Series([0] * 9)
        dense = pd.Series([50] * 9)
        assert abnormal_attention(silent).iloc[8] == pytest.approx(0.0)
        assert abnormal_attention(dense).iloc[8] == pytest.approx(0.0)
        assert bool(sparsity_fields(silent)["att_lookback_all_zero"].iloc[8]) is True
        assert bool(sparsity_fields(dense)["att_lookback_all_zero"].iloc[8]) is False


# ---------------------------------------------------------------------------
# 訂單失衡（PRD §3.7）
# ---------------------------------------------------------------------------

class TestImbalance:
    def test_scalar_formula(self):
        # 總量 1000，法人買 300、賣 100 → 非法人買 700、賣 900
        assert non_inst_order_imbalance(1000, 300, 100) == pytest.approx((700 - 900) / 1600)

    def test_balanced_is_zero(self):
        assert non_inst_order_imbalance(1000, 200, 200) == pytest.approx(0.0)

    def test_zero_denominator_is_nan_not_zero(self):
        assert np.isnan(non_inst_order_imbalance(0, 0, 0))

    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            vol = int(rng.integers(1, 10**7))
            buy = int(rng.integers(0, vol + 1))
            sell = int(rng.integers(0, vol + 1))
            val = non_inst_order_imbalance(vol, buy, sell)
            assert -1.0 <= val <= 1.0

    def test_net_vs_gross_mistake_is_detectably_different(self):
        """守門測試：用淨額取代買賣量會得到不同的值，避免 (a) 類量綱錯誤。"""
        correct = non_inst_order_imbalance(1000, 300, 100)
        wrong = (1000 - (300 - 100)) / 1000  # 常見誤寫
        assert not np.isclose(correct, wrong)

    def test_weekly_drops_thin_days_and_short_weeks(self):
        daily = pd.DataFrame({
            "week": ["w1"] * 5 + ["w2"] * 5,
            "volume": [50_000] * 5 + [1_000, 1_000, 1_000, 50_000, 50_000],
            "inst_buy": [10_000] * 10,
            "inst_sell": [5_000] * 10,
        })
        roi = weekly_non_inst_roi(daily, min_daily_volume=10_000, min_valid_days=3)
        assert not pd.isna(roi["w1"])
        # w2 只剩 2 個有效日 → 缺值
        assert pd.isna(roi["w2"])

    def test_weekly_rejects_inst_exceeding_volume(self):
        """法人量超過總成交量代表單位或來源錯誤，該日必須剔除而非產出假值。"""
        daily = pd.DataFrame({
            "week": ["w1"] * 4,
            "volume": [50_000] * 4,
            "inst_buy": [60_000, 10_000, 10_000, 10_000],  # 第一天量綱可疑
            "inst_sell": [5_000] * 4,
        })
        roi = weekly_non_inst_roi(daily, min_daily_volume=1, min_valid_days=3)
        assert not pd.isna(roi["w1"])
        # 若未剔除，週加總會被汙染
        contaminated = (50_000 * 4 - 90_000) - (50_000 * 4 - 20_000)
        assert not np.isclose(roi["w1"], contaminated / (2 * 200_000 - 110_000))


# ---------------------------------------------------------------------------
# 名稱碰撞消解（PRD §3.4）
# ---------------------------------------------------------------------------

def _matcher() -> Matcher:
    variants = [
        Variant("1216", "統一", "name_with_context", 2),
        Variant("2912", "統一超", "name", 3),
        Variant("1301", "台塑", "name_with_context", 2),
        Variant("6505", "台塑化", "name", 3),
        Variant("1414", "東和", "name_with_context", 2),
        Variant("2006", "東和鋼鐵", "name", 4),
        Variant("2002", "中鋼", "name_with_context", 2),
        Variant("2013", "中鋼構", "name", 3),
        Variant("2330", "台積電", "name", 3),
        Variant("2603", "長榮海運", "name", 4),
        Variant("2015", "豐興", "name_with_context", 2),
    ]
    return Matcher(variants=variants, code_only={"9937"},
                   valid_codes={v.ticker for v in variants} | {"9937"})


class TestNameMatching:
    def test_longest_match_first_prevents_prefix_theft(self):
        """「統一超」不得被「統一」吃掉前兩個字。"""
        m = _matcher()
        found = m.match("[標的] 統一超 2912 多", "股價便宜")
        assert "2912" in found
        assert "1216" not in found

    def test_cross_sector_prefix_collision(self):
        """東和(1414,紡織) vs 東和鋼鐵(2006,鋼鐵)——完全前綴且跨產業，最高風險。"""
        m = _matcher()
        found = m.match("[標的] 東和鋼鐵 多", "鋼價上漲 股價看好")
        assert "2006" in found and "1414" not in found

    def test_short_name_needs_context(self):
        m = _matcher()
        # 有股票語境詞 → 命中
        assert "1216" in m.match("[標的] 統一 多", "目標價 80 元")
        # 純食品討論、無股票語境 → 不命中
        assert "1216" not in m.match("今天買了統一麵包", "很好吃")

    def test_code_match_beats_name(self):
        m = _matcher()
        found = m.match("[標的] 2330 多", "")
        assert found["2330"] == "code"

    def test_generic_term_demotion_uses_code_only(self):
        """降級股停用簡稱比對，只用代號。"""
        m = _matcher()
        assert "9937" not in m.match("全國電子好便宜", "")
        assert "9937" in m.match("9937 這檔如何", "")

    def test_year_like_code_rejected_without_evidence(self):
        """2015 既是豐興的代號也是年份，無正面證據時不得採計。"""
        m = _matcher()
        assert "2015" not in m.match("回顧 2015 年的行情", "")
        assert "2015" not in m.match("從 2015 到 2020", "")

    def test_year_like_code_accepted_with_evidence(self):
        m = _matcher()
        assert "2015" in m.match("2015 這檔如何", "")
        assert "2015" in m.match("豐興 2015 財報", "營收成長")

    def test_price_context_code_rejected(self):
        m = _matcher()
        assert "2330" not in m.match("漲到 2330 點", "")

    def test_multiple_tickers_per_article(self):
        """一篇文可對應多檔（PRD §3.4 第 5 點）。"""
        m = _matcher()
        found = m.match("[標的] 台積電 2330 與 長榮海運 2603", "都看好")
        assert {"2330", "2603"} <= set(found)

    def test_normalization_handles_fullwidth(self):
        m = _matcher()
        assert "2330" in m.match("［標的］２３３０ 多", "這檔")

    def test_changrong_requires_full_name(self):
        """長榮大學／長榮航空不得算成長榮海運（PRD §3.4）。"""
        m = _matcher()
        assert "2603" not in m.match("長榮大學的學生", "")
        assert "2603" in m.match("長榮海運 財報", "")


def test_normalize_is_idempotent():
    s = "［標的］２３３０　台積電"
    assert normalize(normalize(s)) == normalize(s)
