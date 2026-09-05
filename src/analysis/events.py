"""關注度起始事件研究（H7）——PRD §5.4(b)。

「這是本專案相對 v1.0 在方法上最重要的新增：它把『稀疏』從缺點轉成乾淨的事件識別。」

對常態零討論的個股，連續型 AbnAtt 沒有意義（回顧窗全零時 AbnAtt 恆為 0），改用
「由零轉正」的事件設計，並以事件研究法報告 t−4 到 t+8 週的 CAR 路徑，含事前四週的
平行趨勢檢查（PRD §8.3）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.regressions import (
    BASE_CONTROLS, ModelResult, fit_twoway, standardize_within,
)


def initiation_regressions(panel: pd.DataFrame, controls: list[str] | None = None,
                           tiers: tuple[str, ...] = ("sparse", "silent"),
                           is_diagnostic: bool = True) -> list[ModelResult]:
    """PRD §5.4(b) 的事件迴歸。限定 sparse 與 silent 子樣本。"""
    controls = controls if controls is not None else BASE_CONTROLS
    sub = panel[panel["sparsity_tier"].isin(tiers)].copy()
    for col in ("is_initiation_weekend", "is_initiation_weekday", "is_initiation"):
        if col in sub.columns:
            sub[col] = sub[col].fillna(False).astype(float)

    d = standardize_within(sub, [c for c in controls if c != "att_zero_base"])
    out = [
        fit_twoway(d, "ret_next",
                   ["is_initiation_weekend", "is_initiation_weekday"],
                   controls, "T7-1 起始事件（sparse＋silent）", is_diagnostic),
        fit_twoway(d, "non_inst_roi_next",
                   ["is_initiation_weekend", "is_initiation_weekday"],
                   [*controls, "non_inst_roi_lag1"],
                   "T7-2 起始事件 → 非三大法人 ROI", is_diagnostic),
    ]
    for tier in tiers:
        dt = standardize_within(sub[sub["sparsity_tier"] == tier],
                                [c for c in controls if c != "att_zero_base"])
        out.append(fit_twoway(dt, "ret_next",
                              ["is_initiation_weekend", "is_initiation_weekday"],
                              controls, f"T7-3 起始事件（{tier}）", is_diagnostic))
    return out


def car_path(panel: pd.DataFrame, event_col: str = "is_initiation_weekend",
             pre: int = 4, post: int = 8,
             tiers: tuple[str, ...] = ("sparse", "silent")) -> pd.DataFrame:
    """t−4 到 t+8 週的累積異常報酬路徑（PRD §7.3 圖 F4、§8.3）。

    異常報酬 = 個股週報酬 − 同週全樣本等權平均（週效果的簡易對應）。事前四週用於
    平行趨勢檢查：若事件前已有顯著漂移，事件設計的識別假設不成立。
    """
    d = panel[panel["sparsity_tier"].isin(tiers)].copy()
    d = d.sort_values(["ticker", "week"])
    d["ar"] = d["ret"] - d.groupby("week")["ret"].transform("mean")

    # 每檔的週序列，用相對位移取事件窗
    frames = []
    for ticker, g in d.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        ev_idx = g.index[g[event_col].fillna(False).astype(bool)]
        for i in ev_idx:
            lo, hi = i - pre, i + post
            if lo < 0 or hi >= len(g):
                continue
            # 事件窗內須為連續週，否則對齊會錯
            span = g.loc[lo:hi, "week"]
            if (span.diff().dropna() != pd.Timedelta(days=7)).any():
                continue
            frames.append(pd.DataFrame({
                "ticker": ticker,
                "event_week": g.loc[i, "week"],
                "tau": range(-pre, post + 1),
                "ar": g.loc[lo:hi, "ar"].values,
            }))
    if not frames:
        return pd.DataFrame(columns=["tau", "mean_ar", "car", "t", "n_events"])

    ev = pd.concat(frames, ignore_index=True)
    agg = ev.groupby("tau")["ar"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "mean_ar", "count": "n_events"})
    agg["se"] = agg["std"] / np.sqrt(agg["n_events"])
    agg["t"] = agg["mean_ar"] / agg["se"].replace(0, np.nan)
    agg["car"] = agg["mean_ar"].cumsum()
    agg["car_from_event"] = np.where(
        agg["tau"] >= 0,
        agg["mean_ar"].where(agg["tau"] >= 0).fillna(0).cumsum(),
        np.nan,
    )

    pre_rows = agg[agg["tau"] < 0]
    agg.attrs["parallel_trend_ok"] = bool((pre_rows["t"].abs() < 1.96).all())
    agg.attrs["n_events_total"] = int(ev.groupby(["ticker", "event_week"]).ngroups)
    return agg


def initiation_feasibility(panel: pd.DataFrame) -> pd.DataFrame:
    """事件數與分布——先算清楚再解讀（與 §5.10 的可行性計算同精神）。"""
    rows = []
    for tier in ("dense", "sparse", "silent"):
        sub = panel[panel["sparsity_tier"] == tier]
        if sub.empty:
            continue
        rows.append({
            "sparsity_tier": tier,
            "n_ticker_weeks": len(sub),
            "n_tickers": sub["ticker"].nunique(),
            "n_initiation": int(sub["is_initiation"].fillna(False).sum()),
            "n_initiation_weekend": int(
                sub["is_initiation_weekend"].fillna(False).sum()),
            "n_initiation_weekday": int(
                sub["is_initiation_weekday"].fillna(False).sum()),
            "n_tickers_with_initiation": int(
                sub[sub["is_initiation"].fillna(False)]["ticker"].nunique()),
            "share_zero_attention_weeks": float((sub["att_all"] == 0).mean()),
        })
    return pd.DataFrame(rows)
