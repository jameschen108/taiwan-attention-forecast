"""跨截面異質性（H6）與持續性／反轉（H5）——PRD §5.5、§5.8。

**判讀必須聯立**：單獨的負交互項不能區分兩條管道——資訊處理與價格壓力都預測低覆蓋
股效果更強。決定性的證據是低覆蓋股的效果**是否在後續週反轉**。因此本模組把 §5.5 與
§5.8 合併輸出為同一張表（PRD §5.5 末段、§8.3）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.regressions import (
    BASE_CONTROLS, ModelResult, fit_twoway, standardize_within,
)

MODERATORS = [
    "log_att_mean_level_52",
    "log_market_cap",
    "turnover",
    "amihud",
    "foreign_holding_pct",
]


def interaction_models(panel: pd.DataFrame, controls: list[str] | None = None,
                       is_diagnostic: bool = True) -> list[ModelResult]:
    """PRD §5.5：AbnAtt_weekend × M。預期 β₂ 顯著為負（低覆蓋股效果更強）。"""
    controls = controls if controls is not None else BASE_CONTROLS
    out = []
    for mod in MODERATORS:
        if mod not in panel.columns:
            out.append(ModelResult(f"T6-{mod}", "SKIPPED",
                                   note=f"調節變數不存在：{mod}"))
            continue
        d = standardize_within(panel, ["abn_attention_weekend", mod,
                                       *[c for c in controls if c != "att_zero_base"]])
        d = d.copy()
        d["interaction"] = d["abn_attention_weekend"] * d[mod]
        extra = [c for c in controls if c != mod]
        out.append(fit_twoway(d, "ret_next",
                              ["abn_attention_weekend", "interaction", mod],
                              extra, f"T6-{mod}", is_diagnostic))
    return out


def tercile_models(panel: pd.DataFrame, moderator: str = "log_att_mean_level_52",
                   controls: list[str] | None = None,
                   is_diagnostic: bool = True) -> list[ModelResult]:
    """報告分組係數而非只有交互項——關係可能非線性（PRD §5.5 末段）。"""
    controls = controls if controls is not None else BASE_CONTROLS
    d = panel.dropna(subset=[moderator]).copy()
    if d.empty:
        return [ModelResult(f"T6-tercile-{moderator}", "SKIPPED", note="調節變數全缺")]
    try:
        d["tercile"] = d.groupby("week")[moderator].transform(
            lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop"))
    except ValueError:
        return [ModelResult(f"T6-tercile-{moderator}", "SKIPPED", note="無法分三組")]

    out = []
    for tercile in ("low", "mid", "high"):
        sub = standardize_within(
            d[d["tercile"] == tercile],
            ["abn_attention_weekend", *[c for c in controls if c != "att_zero_base"]])
        out.append(fit_twoway(sub, "ret_next", ["abn_attention_weekend"], controls,
                              f"T6-tercile-{moderator}:{tercile}", is_diagnostic))
    return out


def reversal_models(panel: pd.DataFrame, controls: list[str] | None = None,
                    horizons: range = range(2, 9),
                    by: str | None = None,
                    is_diagnostic: bool = True) -> list[ModelResult]:
    """PRD §5.8：報酬效果的後續 1–8 週反轉檢定。

    `by` 給定時（例如 sparsity_tier），**依分組分別報告**——這是 §5.5 判讀的必要
    輸入，不是附屬檢定。
    """
    controls = controls if controls is not None else BASE_CONTROLS
    groups: list[tuple[str, pd.DataFrame]] = (
        [("全樣本", panel)] if by is None
        else [(str(k), g) for k, g in panel.groupby(by, dropna=True)]
    )
    out = []
    for label, grp in groups:
        d = standardize_within(
            grp, ["abn_attention_weekend",
                  *[c for c in controls if c != "att_zero_base"]])
        out.append(fit_twoway(d, "ret_next", ["abn_attention_weekend"], controls,
                              f"T6R-{label}-h1", is_diagnostic))
        for h in horizons:
            col = f"ret_fwd{h}"
            if col not in d.columns:
                continue
            out.append(fit_twoway(d, col, ["abn_attention_weekend"], controls,
                                  f"T6R-{label}-h{h}", is_diagnostic))
    return out


def joint_reading(interaction: list[ModelResult],
                  reversal: list[ModelResult]) -> pd.DataFrame:
    """把交互項與反轉檢定合併為同一張表，並給出聯立判讀（PRD §2.3、§5.5）。"""
    rows = []
    for r in interaction:
        if r.status != "OK" or "interaction" not in r.params:
            continue
        beta, t = r.params["interaction"], r.tstats["interaction"]
        rows.append({
            "moderator": r.name.replace("T6-", ""),
            "beta_interaction": beta, "t_interaction": t,
            "p_interaction": r.pvalues["interaction"],
            "interaction_negative_sig": bool(beta < 0 and t < -1.96),
            "n_obs": r.n_obs,
        })
    inter = pd.DataFrame(rows)

    rev_rows = []
    for r in reversal:
        if r.status != "OK" or "abn_attention_weekend" not in r.params:
            continue
        label, _, horizon = r.name.replace("T6R-", "").rpartition("-h")
        rev_rows.append({
            "group": label, "horizon": int(horizon),
            "beta": r.params["abn_attention_weekend"],
            "t": r.tstats["abn_attention_weekend"],
            "n_obs": r.n_obs,
        })
    rev = pd.DataFrame(rev_rows)

    if rev.empty:
        return inter.assign(joint_reading="反轉檢定無結果，無法聯立判讀")

    # 每組是否出現顯著的後續反轉：h≥2 有顯著負係數
    verdicts = []
    for group, g in rev.groupby("group"):
        h1 = g[g["horizon"] == 1]
        later = g[g["horizon"] >= 2]
        has_effect = bool(not h1.empty and h1["t"].abs().iat[0] > 1.96)
        has_reversal = bool((later["t"] < -1.96).any())
        if not has_effect:
            reading = "無初始效果，反轉檢定不具判讀力"
        elif has_reversal:
            reading = "有效果且後續反轉 → 支持價格壓力管道"
        else:
            reading = "有效果且不反轉 → 支持資訊處理管道"
        verdicts.append({"group": group, "h1_significant": has_effect,
                         "reversal_detected": has_reversal,
                         "joint_reading": reading})
    return pd.DataFrame(verdicts).merge(
        inter, how="outer", left_on="group", right_on="moderator")


def fama_macbeth(panel: pd.DataFrame, xs: list[str],
                 y: str = "ret_next") -> pd.DataFrame:
    """PRD §5.5 的 Fama-MacBeth 對照版本。"""
    cols = ["week", y, *xs]
    d = panel[cols].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for week, g in d.groupby("week"):
        if len(g) < 20:
            continue
        X = sm.add_constant(g[xs])
        try:
            fit = sm.OLS(g[y], X).fit()
        except Exception:  # noqa: BLE001
            continue
        rows.append({"week": week, **fit.params.to_dict()})
    if not rows:
        return pd.DataFrame()
    coefs = pd.DataFrame(rows).set_index("week")
    out = []
    for col in coefs.columns:
        s = coefs[col].dropna()
        if len(s) < 10:
            continue
        # Newey-West 調整（lag 4）
        nw = sm.OLS(s, np.ones(len(s))).fit(cov_type="HAC",
                                            cov_kwds={"maxlags": 4})
        out.append({"term": col, "mean_coef": float(s.mean()),
                    "t_newey_west": float(nw.tvalues.iloc[0]),
                    "n_weeks": len(s)})
    return pd.DataFrame(out)
