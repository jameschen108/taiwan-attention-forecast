"""非三大法人訂單失衡（PRD §3.7）。

命名紀律：殘差包含未列入三大法人的大戶與其他機構，只能稱為「非三大法人訂單失衡」，
不得改寫為 retail order imbalance 或散戶淨買超（PRD §11）。

兩個必須避免的量綱錯誤：
  (a) 用「總成交 − 三大法人買賣超」把總額與淨額相減；
  (b) 用成交金額去減三大法人的股數。
三個輸入一律為**股數**。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def non_inst_order_imbalance(total_volume, inst_buy, inst_sell):
    """單日非三大法人訂單失衡。

    STOCK_DAY 的「成交股數」同時是總買量與總賣量；T86 的買進／賣出為股數。
    """
    r_buy = total_volume - inst_buy
    r_sell = total_volume - inst_sell
    denom = r_buy + r_sell
    if isinstance(denom, (int, float, np.floating, np.integer)):
        return (r_buy - r_sell) / denom if denom > 0 else np.nan
    out = (r_buy - r_sell) / denom.where(denom > 0)
    return out


def weekly_non_inst_roi(daily: pd.DataFrame, min_daily_volume: int = 10_000,
                        min_valid_days: int = 3) -> pd.Series:
    """週聚合。

    長尾個股常有單日成交量極小甚至為零，denom 極小時 ROI 會爆到 ±1 並產生假的
    極端值。規則：低於門檻的交易日排除；一週有效交易日不足 min_valid_days 則缺值。

    `daily` 須含欄位 volume / inst_buy / inst_sell / week，且皆為股數。
    """
    df = daily.copy()
    valid = (
        (df["volume"] >= min_daily_volume)
        & df["inst_buy"].notna()
        & df["inst_sell"].notna()
        # 法人買賣量不得超過總成交量（量綱錯誤的守門）
        & (df["inst_buy"] <= df["volume"])
        & (df["inst_sell"] <= df["volume"])
    )
    df = df[valid]
    if df.empty:
        return pd.Series(dtype=float)

    # 先加總週內股數再算失衡，避免小分母日的權重被放大
    agg = df.groupby("week").agg(
        volume=("volume", "sum"),
        inst_buy=("inst_buy", "sum"),
        inst_sell=("inst_sell", "sum"),
        n_valid_days=("volume", "size"),
    )
    roi = non_inst_order_imbalance(agg["volume"], agg["inst_buy"], agg["inst_sell"])
    return roi.where(agg["n_valid_days"] >= min_valid_days)


def abnormal_turnover(turnover: pd.Series, lookback: int = 8,
                      min_periods: int = 8) -> pd.Series:
    """異常周轉率：以過去 8 週均值為基準，與 AbnAtt 同一套定義。"""
    values = np.log1p(turnover.astype(float))
    baseline = values.shift(1).rolling(lookback, min_periods=min_periods).mean()
    return values - baseline
