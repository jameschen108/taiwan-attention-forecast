"""補班日自然實驗（H4）——PRD §5.10。

台灣的補班星期六是外生的認知餘裕衝擊，原論文無等價設計。但**事件週數本身沒有增加**，
而週固定效果會吸收掉補班週的共同成分——識別完全來自「補班週內，高關注度股 vs. 低關
注度股」的差異。

因此 PRD 要求：**先計算可用事件週數、有效觀測數與檢定力再投入**，並且群集在週層級
——有效群集數等於事件週數，而非觀測數。標準誤必須反映這一點。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.regressions import (
    BASE_CONTROLS, ModelResult, fit_twoway, standardize_within,
)


def feasibility(panel: pd.DataFrame, trading_days: pd.DataFrame) -> pd.DataFrame:
    """makeup_feasibility.csv（PRD §5.10）。不足時不得硬跑。"""
    makeup_days = trading_days[trading_days["is_makeup_saturday"]]["date"]
    ev = panel[panel["is_makeup_saturday_week"]]
    usable = ev.dropna(subset=["abn_attention_weekend", "ret_next"])

    n_weeks = int(panel.loc[panel["is_makeup_saturday_week"], "week"].nunique())
    n_usable_weeks = int(usable["week"].nunique())

    # 群集數 = 事件週數。以雙尾 5%、常態近似估算可偵測的最小標準化效果量。
    # 有效自由度受群集數支配（Cameron-Gelbach-Miller 2008）。
    n_obs = len(usable)
    if n_usable_weeks >= 2 and n_obs > 0:
        # 保守：以群集數而非觀測數決定臨界值
        from scipy import stats as st
        crit = float(st.t.ppf(0.975, df=max(n_usable_weeks - 1, 1)))
        # 80% 檢定力所需的效果量（以群集標準誤為單位）
        mde_in_cluster_se = crit + 0.84
    else:
        crit, mde_in_cluster_se = np.nan, np.nan

    return pd.DataFrame([{
        "n_makeup_trading_saturdays": int(len(makeup_days)),
        "makeup_dates": ";".join(str(d.date()) for d in makeup_days),
        "n_makeup_weeks_in_panel": n_weeks,
        "n_makeup_weeks_usable": n_usable_weeks,
        "n_usable_obs": n_obs,
        "n_tickers_in_makeup_weeks": int(usable["ticker"].nunique()),
        "effective_clusters": n_usable_weeks,
        "t_critical_at_cluster_df": crit,
        "mde_in_cluster_se_units": mde_in_cluster_se,
        "verdict": _verdict(n_usable_weeks, n_obs),
        "last_makeup_saturday": (str(makeup_days.max().date())
                                 if len(makeup_days) else ""),
        "note": ("有效群集數等於事件週數而非觀測數；標準誤須以 wild cluster "
                 "bootstrap 或依事件週數調整的臨界值處理（PRD §5.10）"),
    }])


def _verdict(n_weeks: int, n_obs: int) -> str:
    if n_weeks == 0:
        return "不可行：面板中無補班週"
    if n_weeks < 5:
        return (f"檢定力嚴重不足（僅 {n_weeks} 個事件週）：結果只能定位為探索性訊號，"
                "不得作為結論（PRD §5.10 備案 c）")
    if n_weeks < 10:
        return (f"檢定力偏低（{n_weeks} 個事件週）：須以 wild cluster bootstrap 報告，"
                "並明確標示為探索性")
    return f"可行（{n_weeks} 個事件週、{n_obs:,} 個觀測）"


def interaction_model(panel: pd.DataFrame, controls: list[str] | None = None,
                      is_diagnostic: bool = True) -> ModelResult:
    """PRD §5.10 的交互項模型。預期資訊處理成立則 β₂ 顯著為負。"""
    controls = controls if controls is not None else BASE_CONTROLS
    d = standardize_within(panel, ["abn_attention_weekend",
                                   *[c for c in controls if c != "att_zero_base"]])
    d = d.copy()
    d["is_makeup"] = d["is_makeup_saturday_week"].astype(float)
    d["interaction"] = d["abn_attention_weekend"] * d["is_makeup"]
    # is_makeup 的主效果會被週固定效果完全吸收，只留交互項
    return fit_twoway(d, "ret_next", ["abn_attention_weekend", "interaction"],
                      controls, "T10 補班日交互項", is_diagnostic)


def wild_cluster_bootstrap(panel: pd.DataFrame, n_boot: int = 999,
                           seed: int = 20260905,
                           controls: list[str] | None = None) -> dict:
    """依事件週數做 wild cluster bootstrap（Cameron, Gelbach & Miller 2008）。

    群集在**週**層級——這是 PRD §5.10 明確要求的，因為有效群集數等於事件週數。
    """
    controls = controls if controls is not None else BASE_CONTROLS
    d = standardize_within(panel, ["abn_attention_weekend",
                                   *[c for c in controls if c != "att_zero_base"]])
    d = d.copy()
    d["interaction"] = (d["abn_attention_weekend"]
                        * d["is_makeup_saturday_week"].astype(float))
    cols = ["ticker", "week", "ret_next", "abn_attention_weekend", "interaction",
            *[c for c in controls if c in d.columns]]
    d = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if d["week"].nunique() < 10 or len(d) < 500:
        return {"status": "SKIPPED",
                "note": "群集數或觀測數不足，bootstrap 不具意義"}

    import statsmodels.api as sm

    y = d["ret_next"].values
    X = sm.add_constant(d[["abn_attention_weekend", "interaction"]]).values
    base = sm.OLS(y, X).fit()
    beta_hat = base.params[2]

    # 受限模型（H0: 交互項 = 0）的殘差，供 wild bootstrap 重抽
    X0 = sm.add_constant(d[["abn_attention_weekend"]]).values
    r0 = sm.OLS(y, X0).fit()
    resid0, fitted0 = r0.resid, r0.fittedvalues

    weeks = d["week"].values
    uniq = np.unique(weeks)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        # Rademacher 權重，逐**週**（群集）指派
        w = pd.Series(rng.choice([-1.0, 1.0], size=len(uniq)), index=uniq)
        y_star = fitted0 + resid0 * w.reindex(weeks).values
        stats[b] = sm.OLS(y_star, X).fit().params[2]

    p = float(np.mean(np.abs(stats) >= abs(beta_hat)))
    return {
        "status": "OK",
        "beta_interaction": float(beta_hat),
        "p_wild_cluster_bootstrap": p,
        "n_clusters": int(len(uniq)),
        "n_obs": int(len(d)),
        "n_boot": n_boot,
        "note": "群集於週層級；有效群集數等於事件週數（PRD §5.10）",
    }
