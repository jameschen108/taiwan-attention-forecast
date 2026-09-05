"""產業維度（H8）——PRD §5.6。

28 個產業 × 約 10 檔的結構讓產業層級檢定成為可能，這是 v1.0 的 50 檔跨產業樣本
做不到的（每產業僅 1–3 檔）。
"""

from __future__ import annotations

import pandas as pd

from src.analysis.regressions import (
    BASE_CONTROLS, ModelResult, fit_twoway, standardize_within,
)


def spillover_models(panel: pd.DataFrame, controls: list[str] | None = None,
                     is_diagnostic: bool = True) -> list[ModelResult]:
    """1. 產業關注度外溢；2. 產業內相對關注度。"""
    controls = controls if controls is not None else BASE_CONTROLS
    d = standardize_within(panel, [
        "abn_attention_weekend", "sector_peer_abn_att_weekend",
        "abn_att_weekend_rel_sector",
        *[c for c in controls if c != "att_zero_base"],
    ])
    out = [
        fit_twoway(d, "ret_next",
                   ["abn_attention_weekend", "sector_peer_abn_att_weekend"],
                   controls, "T8-1 產業外溢", is_diagnostic),
        fit_twoway(d, "ret_next", ["abn_att_weekend_rel_sector"], controls,
                   "T8-2 產業內相對關注度", is_diagnostic),
    ]
    # 外溢是否反轉——判讀 H8 的必要輸入（PRD §2.3）
    for h in (2, 4):
        col = f"ret_fwd{h}"
        if col in d.columns:
            out.append(fit_twoway(
                d, col, ["abn_attention_weekend", "sector_peer_abn_att_weekend"],
                controls, f"T8-3 產業外溢 t+{h}（反轉檢定）", is_diagnostic))
    return out


def by_sector_coefficients(panel: pd.DataFrame, controls: list[str] | None = None,
                           min_obs: int = 500) -> pd.DataFrame:
    """3. 分產業估計 β₂ 並報告分布。**此為描述性，不宣稱因果**（PRD §5.6）。"""
    controls = controls if controls is not None else BASE_CONTROLS
    rows = []
    for sector, grp in panel.groupby("sector"):
        d = standardize_within(
            grp, ["abn_attention_weekend",
                  *[c for c in controls if c != "att_zero_base"]])
        res = fit_twoway(d, "ret_next", ["abn_attention_weekend"], controls,
                         f"sector:{sector}", is_diagnostic=True)
        rows.append({
            "sector": sector,
            "n_tickers": grp["ticker"].nunique(),
            "status": res.status,
            "n_obs": res.n_obs,
            "beta_weekend": res.params.get("abn_attention_weekend"),
            "t_weekend": res.tstats.get("abn_attention_weekend"),
            "p_weekend": res.pvalues.get("abn_attention_weekend"),
            "note": res.note,
        })
    out = pd.DataFrame(rows).sort_values("beta_weekend", ascending=False,
                                         na_position="last")
    out.attrs["caveat"] = "描述性，不宣稱因果（PRD §5.6 第 3 點）"
    return out


def sector_coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    """依產業與稀疏度分層的覆蓋表（PRD §7.3 表 T1）。"""
    g = panel.groupby("sector")
    out = g.agg(
        n_tickers=("ticker", "nunique"),
        n_ticker_weeks=("ticker", "size"),
        mean_att=("att_all", "mean"),
        median_att=("att_all", "median"),
        share_zero_weeks=("att_all", lambda s: float((s == 0).mean())),
    ).reset_index()
    tiers = (panel.groupby(["sector", "sparsity_tier"]).size()
             .unstack(fill_value=0).reset_index())
    return out.merge(tiers, on="sector", how="left").sort_values(
        "mean_att", ascending=False)
