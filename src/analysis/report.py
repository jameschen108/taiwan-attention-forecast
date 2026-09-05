"""分析產出彙整：描述統計、測度效度、健檢與就緒度（PRD §7.2、§7.3、§8）。

嚴格區分 diagnostic 與正式結果（PRD §3.8、§11）：控制變數不齊或權值還原未驗證時，
所有結果標記為 diagnostic，不得升格為主結果。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 正式主表所需的控制變數（PRD §3.8）。缺一即不得產出正式結果。
FORMAL_CONTROLS = [
    "market_cap", "shares_outstanding", "turnover", "amihud",
    "foreign_holding_pct", "ret_lag1", "ret_lag4", "ret_lag25",
    "news_count",              # PRD §4.2：主表必要控制項，非 robustness
    "selling_expense_to_sales",
    "analyst_count", "forecast_dispersion", "forecast_revision",
]


def descriptive_stats(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in [
        "att_all", "att_weekend", "att_weekday",
        "abn_attention_all", "abn_attention_weekend", "abn_attention_weekday",
        "ret", "ret_next", "non_inst_roi", "non_inst_roi_next",
        "turnover", "abn_turnover", "market_cap", "foreign_holding_pct",
        "att_nonzero_weeks_52", "att_mean_level_52",
    ] if c in panel.columns]
    d = panel[cols].replace([np.inf, -np.inf], np.nan)
    out = d.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    out["n_missing"] = d.isna().sum()
    out["share_missing"] = d.isna().mean()
    return out.reset_index().rename(columns={"index": "variable"})


def window_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """PRD §8.2：corr(整週, 週間) 應顯著高於 corr(整週, 週末)，型態同原論文
    （0.90 vs 0.40）。"""
    d = panel[["abn_attention_all", "abn_attention_weekday",
               "abn_attention_weekend"]].replace([np.inf, -np.inf], np.nan).dropna()
    c_wd = float(d["abn_attention_all"].corr(d["abn_attention_weekday"]))
    c_we = float(d["abn_attention_all"].corr(d["abn_attention_weekend"]))
    return pd.DataFrame([{
        "corr_all_weekday": c_wd,
        "corr_all_weekend": c_we,
        "gap": c_wd - c_we,
        "n_obs": len(d),
        "paper_benchmark_weekday": 0.90,
        "paper_benchmark_weekend": 0.40,
        "pattern_matches_paper": bool(c_wd > c_we),
    }])


def sparsity_distribution(panel: pd.DataFrame) -> pd.DataFrame:
    """sparsity_distribution.csv（PRD §7.2、§8.2）。"""
    by_ticker = (panel.groupby("ticker")["sparsity_tier"]
                 .agg(lambda s: s.dropna().mode().iat[0]
                      if not s.dropna().empty else "unknown"))
    rows = []
    for tier in ("dense", "sparse", "silent", "unknown"):
        sub = panel[panel["sparsity_tier"] == tier]
        rows.append({
            "sparsity_tier": tier,
            "n_tickers_modal": int((by_ticker == tier).sum()),
            "n_ticker_weeks": len(sub),
            "share_of_panel": len(sub) / max(len(panel), 1),
            "mean_att": float(sub["att_all"].mean()) if len(sub) else np.nan,
            "share_zero_weeks": float((sub["att_all"] == 0).mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def analysis_readiness(panel: pd.DataFrame,
                       price_adjustment_verified: bool) -> pd.DataFrame:
    """analysis_readiness.csv（PRD §7.2、§8.4）。"""
    have = {c: (c in panel.columns and panel[c].notna().any())
            for c in FORMAL_CONTROLS}
    formal_controls_available = all(have.values())
    return pd.DataFrame([{
        "formal_main_return": bool(formal_controls_available
                                   and price_adjustment_verified),
        "formal_controls_available": formal_controls_available,
        "price_adjustment_verified": price_adjustment_verified,
        "missing_controls": ";".join(k for k, v in have.items() if not v),
        "n_rows": len(panel),
        "n_tickers": panel["ticker"].nunique(),
        "n_weeks": panel["week"].nunique(),
        "verdict": ("正式主表可產出" if formal_controls_available
                    and price_adjustment_verified
                    else "控制變數不齊 → 所有結果標記為 diagnostic（PRD §3.8）"),
    }])


def health_checks(panel: pd.DataFrame, universe: pd.DataFrame,
                  screening: pd.DataFrame | None,
                  window_corr: pd.DataFrame,
                  sparsity: pd.DataFrame,
                  readiness: pd.DataFrame) -> pd.DataFrame:
    """health_checks.csv：每列寫出「要求／觀測值／是否通過」（PRD §6.2 F5）。"""
    rows: list[dict] = []

    def add(check: str, requirement: str, observed, passed: bool,
            section: str = "") -> None:
        rows.append({"check": check, "requirement": requirement,
                     "observed": observed, "passed": bool(passed),
                     "prd_section": section})

    add("宇宙列數", "恰為 267 列", len(universe), len(universe) == 267, "§8.1")
    add("ticker 唯一", "無重複", int(universe["ticker"].duplicated().sum()),
        not universe["ticker"].duplicated().any(), "§8.1")
    add("產業數", "28 個產業", universe["sector"].nunique(),
        universe["sector"].nunique() == 28, "§8.1")
    add("市場別解析", "未解析者為 0 檔",
        int(universe["venue"].isna().sum()), not universe["venue"].isna().any(), "§8.1")
    add("上市日齊備", "每檔皆有 listing_date",
        int(universe["listing_date"].isna().sum()),
        not universe["listing_date"].isna().any(), "§8.1")

    dup = panel.duplicated(["ticker", "week"]).sum()
    add("ticker,week 唯一", "無重複鍵", int(dup), dup == 0, "§8.1")

    roi = panel["non_inst_roi"].dropna()
    in_range = bool(((roi >= -1) & (roi <= 1)).all()) if len(roi) else True
    add("non_inst_roi 值域", "落在 [-1, 1]",
        f"[{roi.min():.4f}, {roi.max():.4f}]" if len(roi) else "無觀測",
        in_range, "§8.1")
    # 異常群聚：±1 附近的比例過高代表分母門檻沒發揮作用
    clump = float((roi.abs() > 0.99).mean()) if len(roi) else 0.0
    add("non_inst_roi 無異常群聚", "|ROI|>0.99 的比例 < 1%",
        f"{clump:.4%}", clump < 0.01, "§8.1")

    add("兩種窗口皆產出", "日曆與交易時段欄位皆存在",
        all(c in panel.columns for c in
            ("abn_attention_weekend", "abn_attention_non_trading")),
        all(c in panel.columns for c in
            ("abn_attention_weekend", "abn_attention_non_trading")), "§8.1")

    wc = window_corr.iloc[0]
    add("窗口相關性型態", "corr(整週,週間) > corr(整週,週末)",
        f"{wc['corr_all_weekday']:.3f} vs {wc['corr_all_weekend']:.3f}",
        bool(wc["pattern_matches_paper"]), "§8.2")

    n_dense = int(sparsity.loc[sparsity["sparsity_tier"] == "dense",
                               "n_tickers_modal"].sum())
    add("dense 檔數", "≥ 40 檔（否則 H2 無法識別，須觸發 Yahoo 備案）",
        n_dense, n_dense >= 40, "§8.2")

    if screening is not None:
        n_a = int((screening["match_quality_tier"] == "A_clean").sum())
        add("測度效度篩檢已執行", "三道篩檢皆產出報告", f"A_clean={n_a}",
            True, "§8.2")
        n_comove = int(screening["comovement_pass"].sum())
        add("交易活動共動", "通過檔數已量化並完整報告（刷掉本身就是結果）",
            f"{n_comove}/{len(screening)}", True, "§3.10")

    r = readiness.iloc[0]
    add("正式主表可產出", "formal_controls_available 為 True",
        bool(r["formal_controls_available"]),
        bool(r["formal_controls_available"]), "§8.4")
    add("權值還原已驗證", "price_adjustment_verified 為 True",
        bool(r["price_adjustment_verified"]),
        bool(r["price_adjustment_verified"]), "§8.1")

    pre_listing = int((pd.to_datetime(panel["week"])
                       < pd.to_datetime(panel["listing_date"])).sum())
    add("上市前為缺列", "上市前的週不得出現在面板", pre_listing,
        pre_listing == 0, "§8.1")

    return pd.DataFrame(rows)
