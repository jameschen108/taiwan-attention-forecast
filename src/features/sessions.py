"""時段窗口指派與週對齊（PRD §3.5）。

不可用日曆上的週六日判定「非交易時段」——台灣春節、連假、颱風假頻繁，寫死
「週五 13:30 → 週一 09:00」會系統性低估非交易窗口。一律以實際交易日曆判定。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

OPEN_MIN = 9 * 60          # 09:00
CLOSE_MIN = 13 * 60 + 30   # 13:30


def assign_session_window(ts: pd.Timestamp, trading_days: set[dt.date]) -> str:
    """交易日的 09:00–13:30 為 intraday，其餘一律 non_trading。"""
    date = ts.date() if isinstance(ts, pd.Timestamp) else ts.date()
    minutes = ts.hour * 60 + ts.minute
    if date in trading_days and OPEN_MIN <= minutes < CLOSE_MIN:
        return "intraday"
    return "non_trading"


def assign_calendar_window(ts: pd.Timestamp) -> str:
    """日曆切法（PRD §3.2 的主規格，與原論文可比）：週六日為 weekend。"""
    return "weekend" if ts.weekday() >= 5 else "weekday"


def next_trading_day(ts: pd.Timestamp, trading_days_sorted: list[dt.date]) -> dt.date | None:
    """非交易時段的關注度歸屬到下一個開市日（PRD §3.5）。

    交易日盤中／盤後皆歸屬：盤中歸當日，收盤後歸下一個開市日。長假不得落到錯的週。
    """
    import bisect

    date = ts.date()
    minutes = ts.hour * 60 + ts.minute
    if date in set(trading_days_sorted) and minutes < CLOSE_MIN:
        return date
    idx = bisect.bisect_right(trading_days_sorted, date)
    if idx >= len(trading_days_sorted):
        return None  # 期末無法指派，列入排除清單（PRD §3.1 終點規則）
    return trading_days_sorted[idx]


def week_of(date: dt.date | pd.Timestamp) -> pd.Timestamp:
    """星期日錨定的週標籤（W-SUN）。

    使週六／日的關注度在時間上嚴格先於次週一至週五的報酬（PRD §3.1）。
    回傳該週的**結束日**（星期日），與 pandas `resample('W-SUN')` 一致。
    """
    ts = pd.Timestamp(date).normalize()
    # 週一=0 … 週日=6；W-SUN 的週結束於星期日
    offset = (6 - ts.weekday()) % 7
    return ts + pd.Timedelta(days=offset)


def attention_week(ts: pd.Timestamp, trading_days_sorted: list[dt.date]) -> pd.Timestamp | None:
    """關注度所屬的「特徵週」（研究管線）。

    以貼文時間所在的 W-SUN 週為準：週六／日的貼文落在以該週日結尾的週，其報酬期為
    下一個日曆週的週一至週五，因此嚴格領先。

    研究管線仍要求存在下一交易日；預測管線請用 `attention_week_calendar`。
    """
    if next_trading_day(ts, trading_days_sorted) is None:
        return None
    return week_of(ts)


def attention_week_calendar(ts: pd.Timestamp) -> pd.Timestamp:
    """預測管線：僅依日曆歸週，不依賴未來成交日是否已存在。

    週末貼文在尚無下週行情時仍可進入特徵週，供最新無標籤推論使用。
    """
    return week_of(ts)


def return_week(feature_week: pd.Timestamp) -> pd.Timestamp:
    """特徵週 w 對應的報酬週：下一個 W-SUN 週（週一開盤 → 週五收盤）。"""
    return pd.Timestamp(feature_week) + pd.Timedelta(days=7)


def build_trading_calendar(dates: list[dt.date]) -> dict:
    """由實際成交日建立交易日曆，並標記補班星期六（PRD §3.5、§5.10）。"""
    days = sorted(set(dates))
    makeup_saturdays = [d for d in days if d.weekday() == 5]
    return {
        "trading_days": set(days),
        "trading_days_sorted": days,
        "makeup_saturdays": makeup_saturdays,
    }


def week_trading_day_count(trading_days: set[dt.date], week_end: pd.Timestamp) -> int:
    """報酬週（週一至週五）的實際交易日數。不足 5 天即為不完整週（PRD §3.9）。"""
    monday = pd.Timestamp(week_end) - pd.Timedelta(days=6)
    return sum(
        1 for i in range(5)
        if (monday + pd.Timedelta(days=i)).date() in trading_days
    )
