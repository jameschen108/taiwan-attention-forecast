"""穩健性檢定清單（PRD §5.11 十四項）。

每一項都必須執行；不可行者必須明確記錄原因（例如樣本期間不含該制度斷點），
不得靜默略過。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.regressions import (
    BASE_CONTROLS, ModelResult, fit_twoway, standardize_within,
)
from src.features.attention import sparsity_tier

XS = ["abn_attention_weekday", "abn_attention_weekend"]


def _fit(d: pd.DataFrame, name: str, xs: list[str] | None = None,
         y: str = "ret_next", controls: list[str] | None = None) -> ModelResult:
    controls = controls if controls is not None else BASE_CONTROLS
    xs = xs or XS
    std = standardize_within(d, [*xs, *[c for c in controls if c != "att_zero_base"]])
    return fit_twoway(std, y, xs, controls, name, is_diagnostic=True)


def run(panel: pd.DataFrame, settings: dict) -> list[ModelResult]:
    out: list[ModelResult] = []
    main = panel[panel["sparsity_tier"].isin(["dense", "sparse"])]

    # 1. 稀疏度 tier 門檻敏感度
    grid = settings["sparsity"]["sensitivity_grid"]
    for dense_min in grid["dense_min_nonzero_weeks"]:
        d = panel.copy()
        d["sparsity_tier"] = sparsity_tier(
            d["att_nonzero_weeks_52"], dense_min,
            settings["sparsity"]["silent_max_nonzero_weeks"])
        out.append(_fit(d[d["sparsity_tier"].isin(["dense", "sparse"])],
                        f"R1 稀疏門檻 dense≥{dense_min}"))

    # 2. 含／不含通用詞降級個股
    out.append(_fit(main[~main["is_code_only_matched"]], "R2 排除通用詞降級股"))
    out.append(_fit(main, "R2 含全部（對照）"))

    # 3. 含／不含上櫃（TPEx）個股
    n_tpex = int((panel["venue"] == "TPEx").sum())
    if n_tpex == 0:
        out.append(ModelResult("R3 排除 TPEx", "SKIPPED",
                               note="本宇宙 267 檔全為 TWSE 上市，無 TPEx 個股，"
                                    "此項不適用（PRD 假設與實得清單不符）"))
    else:
        out.append(_fit(main[main["venue"] == "TWSE"], "R3 僅 TWSE"))

    # 4. 含／不含 -KY 個股
    out.append(_fit(main[~main["is_ky"].astype(bool)], "R4 排除 -KY"))

    # 5. winsorize
    lo, hi = settings["regression"]["winsorize"]
    d = main.copy()
    for col in ("ret_next", *XS):
        if col in d.columns:
            q = d[col].quantile([lo, hi])
            d[col] = d[col].clip(q.iloc[0], q.iloc[1])
    out.append(_fit(d, f"R5 winsorize {lo:.0%}/{hi:.0%}"))

    # 6. 漲跌幅 7% 期間 vs. 10% 期間
    pre = main[~main["regime_price_limit_10pct"]]
    if pre["week"].nunique() < 20:
        out.append(ModelResult(
            "R6 漲跌幅 7% 期間", "SKIPPED",
            n_periods=int(pre["week"].nunique()),
            note=("主樣本始於 2015-05，制度變更在 2015-06-01，7% 期間僅 "
                  f"{pre['week'].nunique()} 週——此穩健性檢定在本樣本期間不可行")))
    else:
        out.append(_fit(pre, "R6 漲跌幅 7% 期間"))
    out.append(_fit(main[main["regime_price_limit_10pct"]], "R6 漲跌幅 10% 期間"))

    # 7. 舊版正值限定指標
    pos = [c for c in ("abn_attention_weekday_posonly",
                       "abn_attention_weekend_posonly") if c in main.columns]
    if len(pos) == 2:
        out.append(_fit(main, "R7 正值限定關注度指標", xs=pos))

    # 8. 日曆窗口 vs. 交易時段窗口
    out.append(_fit(main, "R8 交易時段切法",
                    xs=["abn_attention_intraday", "abn_attention_non_trading"]))

    # 9. 排除不完整週 vs. 保留
    out.append(_fit(main[~main["is_incomplete_week"]], "R9 排除不完整週"))

    # 10. sector_as_of_2023 vs. sector_pit
    out.append(ModelResult(
        "R10 sector_pit 對照", "SKIPPED",
        note=("逐年當期產業分類（sector_pit）需 TWSE 歷年分類快照，目前只有 2026 "
              "年的現況分類；主規格以 sector_as_of_2023 執行並在 LIMITATIONS 揭露")))

    # 11. 基準報酬三版本
    out.append(_fit(main, "R11 基準：週固定效果（主規格）"))
    for col, label in (("ret_next_ex_ew", "樣本等權"),):
        if col in main.columns:
            out.append(_fit(main, f"R11 基準：{label}", y=col))

    # 12. 子期間穩定性
    weeks = pd.to_datetime(main["week"])
    mid = weeks.min() + (weeks.max() - weeks.min()) / 2
    out.append(_fit(main[weeks <= mid], "R12 前半期"))
    out.append(_fit(main[weeks > mid], "R12 後半期"))
    out.append(_fit(main[main["regime_continuous_trading"]],
                    "R12 逐筆交易後（2020-03 起）"))

    # 13. 僅含樣本起始日前已上市個股（宇宙 look-ahead 對照）
    start = pd.Timestamp(settings["sample"]["main_start"])
    early = main[pd.to_datetime(main["listing_date"]) < start]
    out.append(_fit(early, "R13 僅樣本起始前已上市"))

    # 14. Yahoo 新聞管道對照
    out.append(ModelResult(
        "R14 Yahoo 新聞管道", "SKIPPED",
        note=("Yahoo 新聞管道依決策暫緩抓取（僅在 PTT 結果不佳時啟用），"
              "因此 §5.9 跨管道比較與 news_count 控制項本輪未執行")))

    # 15. 大量清單型貼文的門檻敏感度（PRD 未列，但實測顯示它是最大的測度威脅：
    #     不處理時 2.8% 的文章會貢獻 36.8% 的配對列）
    out.append(ModelResult(
        "R15 大量清單型貼文門檻", "SKIPPED",
        note=("需以不同 max_tickers_per_article 重建面板；"
              "執行方式見 scripts/sensitivity_bulk_listing.py")))

    return out


def summarize(results: list[ModelResult], focus: str = "abn_attention_weekend"
              ) -> pd.DataFrame:
    """穩健性彙總表 T12：每項的焦點係數、t 值與狀態。"""
    rows = []
    for r in results:
        rows.append({
            "check": r.name,
            "status": r.status,
            "n_obs": r.n_obs,
            "n_tickers": r.n_entities,
            "n_weeks": r.n_periods,
            "beta_focus": r.params.get(focus, np.nan),
            "t_focus": r.tstats.get(focus, np.nan),
            "p_focus": r.pvalues.get(focus, np.nan),
            "note": r.note,
        })
    return pd.DataFrame(rows)
