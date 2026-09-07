"""異常關注度、稀疏度分層與起始事件（PRD §3.2、§3.2.1）。

PTT 的零是**真實的零**（該週確實沒人討論），不是 Google Trends 那種「低於回報門檻」
的遺漏值，因此用 log1p 而非 mask 後取 log。舊的正值限定指標保留為 robustness。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.sessions import week_of


def abnormal_attention(counts: pd.Series, lookback: int = 8,
                       min_periods: int = 8, transform: str = "log1p") -> pd.Series:
    """AbnAtt = f(Att_w) − mean(f(Att_{w-8..w-1}))。

    回顧窗**嚴格不含當期**，避免 look-ahead。不足 min_periods 維持缺值。
    """
    if transform == "log1p":
        values = np.log1p(counts.astype(float))
    elif transform == "log_positive":
        # 舊版正值限定指標（PRD §3.2 robustness）：零視為缺值
        values = np.log(counts.astype(float).where(counts > 0))
    else:
        raise ValueError(f"未知 transform: {transform}")
    baseline = values.shift(1).rolling(lookback, min_periods=min_periods).mean()
    return values - baseline


def sparsity_fields(counts: pd.Series, lookback_weeks: int = 52,
                    abn_lookback: int = 8) -> pd.DataFrame:
    """PRD §3.2.1 的稀疏度欄位。全部只用**過去**資訊。"""
    counts = counts.astype(float)
    nonzero = (counts > 0).astype(float)
    return pd.DataFrame({
        "att_nonzero_weeks_52": nonzero.shift(1)
            .rolling(lookback_weeks, min_periods=1).sum(),
        "att_mean_level_52": counts.shift(1)
            .rolling(lookback_weeks, min_periods=1).mean(),
        "is_initiation": (
            (counts > 0)
            & (counts.shift(1).rolling(abn_lookback, min_periods=abn_lookback).sum() == 0)
        ),
        "att_lookback_all_zero": (
            counts.shift(1).rolling(abn_lookback, min_periods=abn_lookback).sum() == 0
        ),
    }, index=counts.index)


def sparsity_tier(nonzero_weeks_52: pd.Series, dense_min: int = 40,
                  silent_max: int = 3) -> pd.Series:
    """dense（≥40）／sparse（4–39）／silent（≤3）。門檻寫死於 settings.yaml。"""
    tier = pd.Series(pd.NA, index=nonzero_weeks_52.index, dtype="object")
    valid = nonzero_weeks_52.notna()
    tier[valid & (nonzero_weeks_52 >= dense_min)] = "dense"
    tier[valid & (nonzero_weeks_52 <= silent_max)] = "silent"
    tier[valid & (nonzero_weeks_52 > silent_max) & (nonzero_weeks_52 < dense_min)] = "sparse"
    return tier


def build_attention_panel(matches: pd.DataFrame, weeks: pd.DatetimeIndex,
                          tickers: list[str], settings: dict) -> pd.DataFrame:
    """由 ticker-article 配對建出 ticker×week 的關注度面板。

    `matches` 須含 `ticker, week, window, effort`。未出現的 ticker-week 補零
    ——**但僅限於該股已上市的週**，上市前的補零由呼叫端剔除（PRD §3.1）。
    """
    att = settings["attention"]
    spa = settings["sparsity"]

    # 日曆切法（weekday/weekend，與原論文可比）與交易時段切法
    # （intraday/non_trading，台灣制度延伸）是**兩個不同欄位**，兩者都必須產出，
    # 不可互相取代（PRD §3.2）。
    calendar_windows = ["weekday", "weekend"]
    session_windows = ["intraday", "non_trading"]
    efforts = ["high_effort", "mid_effort", "low_effort"]

    idx = pd.MultiIndex.from_product([tickers, weeks], names=["ticker", "week"])
    panel = pd.DataFrame(index=idx)

    def counts_for(mask: pd.Series, label: str) -> None:
        sub = matches[mask]
        series = (sub.groupby(["ticker", "week"]).size()
                  .reindex(idx, fill_value=0).astype(float))
        panel[f"att_{label}"] = series

    counts_for(pd.Series(True, index=matches.index), "all")
    for w in calendar_windows:
        counts_for(matches["window"] == w, w)
    for w in session_windows:
        counts_for(matches["session"] == w, w)
    for e in efforts:
        counts_for(matches["effort"] == e, e)
        counts_for((matches["effort"] == e) & (matches["window"] == "weekend"),
                   f"{e}_weekend")
        counts_for((matches["effort"] == e) & (matches["window"] == "weekday"),
                   f"{e}_weekday")

    # 每檔獨立計算異常值與稀疏度
    out = []
    for ticker, grp in panel.groupby(level="ticker", sort=False):
        grp = grp.sort_index(level="week")
        block = grp.copy()
        for col in [c for c in grp.columns if c.startswith("att_")]:
            label = col[len("att_"):]
            block[f"abn_attention_{label}"] = abnormal_attention(
                grp[col], att["lookback_weeks"], att["min_periods"], att["transform"]
            ).values
            if label in ("all", "weekend", "weekday"):
                block[f"abn_attention_{label}_posonly"] = abnormal_attention(
                    grp[col], att["lookback_weeks"], att["min_periods"], "log_positive"
                ).values
        sp = sparsity_fields(grp["att_all"], att["sparsity_lookback_weeks"],
                             att["lookback_weeks"])
        for col in sp.columns:
            block[col] = sp[col].values
        block["is_initiation_weekend"] = (
            block["is_initiation"].values & (grp["att_weekend"].values > 0)
        )
        block["is_initiation_weekday"] = (
            block["is_initiation"].values & (grp["att_weekday"].values > 0)
        )
        out.append(block)

    panel = pd.concat(out)
    panel["sparsity_tier"] = sparsity_tier(
        panel["att_nonzero_weeks_52"],
        spa["dense_min_nonzero_weeks"], spa["silent_max_nonzero_weeks"],
    )
    # PRD §3.2.1 規則 2：回顧窗全零者另設虛擬變數吸收，不得當成 AbnAtt = 0
    panel["att_zero_base"] = panel["att_lookback_all_zero"].fillna(False).astype(int)
    return panel


def build_attention_panel_post_listing(
    matches: pd.DataFrame,
    weeks: pd.DatetimeIndex,
    listing_dates: pd.Series,
    settings: dict,
) -> pd.DataFrame:
    """Like build_attention_panel but rolling windows exclude pre-listing weeks.

    For each ticker, only weeks on/after listing week enter the grid and
    rolling AbnAtt / 52w sparsity (forecast spec §7.2 warm-up fix).
    """
    att = settings["attention"]
    spa = settings["sparsity"]
    calendar_windows = ["weekday", "weekend"]
    session_windows = ["intraday", "non_trading"]
    efforts = ["high_effort", "mid_effort", "low_effort"]

    listing_weeks = listing_dates.map(week_of)
    tickers = listing_dates.index.tolist()

    out_blocks = []
    for ticker in tickers:
        list_w = listing_weeks[ticker]
        ticker_weeks = weeks[weeks >= list_w]
        if len(ticker_weeks) == 0:
            continue
        idx = pd.MultiIndex.from_product([[ticker], ticker_weeks], names=["ticker", "week"])
        panel = pd.DataFrame(index=idx)

        def counts_for(mask: pd.Series, label: str) -> None:
            sub = matches[(matches["ticker"] == ticker) & mask]
            series = (sub.groupby("week").size()
                      .reindex(ticker_weeks, fill_value=0).astype(float))
            panel[f"att_{label}"] = series.values

        m_t = matches["ticker"] == ticker
        counts_for(m_t, "all")
        for w in calendar_windows:
            counts_for(m_t & (matches["window"] == w), w)
        for w in session_windows:
            counts_for(m_t & (matches["session"] == w), w)
        for e in efforts:
            counts_for(m_t & (matches["effort"] == e), e)
            counts_for(m_t & (matches["effort"] == e) & (matches["window"] == "weekend"),
                       f"{e}_weekend")
            counts_for(m_t & (matches["effort"] == e) & (matches["window"] == "weekday"),
                       f"{e}_weekday")

        grp = panel.sort_index(level="week")
        block = grp.copy()
        for col in [c for c in grp.columns if c.startswith("att_")]:
            label = col[len("att_"):]
            block[f"abn_attention_{label}"] = abnormal_attention(
                grp[col], att["lookback_weeks"], att["min_periods"], att["transform"]
            ).values
            if label in ("all", "weekend", "weekday"):
                block[f"abn_attention_{label}_posonly"] = abnormal_attention(
                    grp[col], att["lookback_weeks"], att["min_periods"], "log_positive"
                ).values
        sp = sparsity_fields(grp["att_all"], att["sparsity_lookback_weeks"],
                             att["lookback_weeks"])
        for col in sp.columns:
            block[col] = sp[col].values
        block["is_initiation_weekend"] = (
            block["is_initiation"].values & (grp["att_weekend"].values > 0)
        )
        block["is_initiation_weekday"] = (
            block["is_initiation"].values & (grp["att_weekday"].values > 0)
        )
        block["sparsity_tier"] = sparsity_tier(
            block["att_nonzero_weeks_52"],
            spa["dense_min_nonzero_weeks"], spa["silent_max_nonzero_weeks"],
        ).values
        block["att_zero_base"] = block["att_lookback_all_zero"].fillna(False).astype(int)
        out_blocks.append(block.reset_index())

    if not out_blocks:
        return pd.DataFrame()
    return pd.concat(out_blocks, ignore_index=False)
