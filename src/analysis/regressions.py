"""主迴歸與機制檢定（PRD §5.1–§5.3）。

通則（§5.1）：
- 連續變數標準化**在稀疏度分層內進行**，避免長尾零值把大型股的變異壓扁。
- 標準誤同時 cluster 至個股與週（Petersen 2009）——267 檔同時暴露於相同的週別
  市場衝擊，橫斷面相依性是一階問題。
- 個股固定效果 ＋ 週固定效果為主規格，取代 v1.0 的「事前扣除等權市場報酬」。
- 樣本量或識別條件不足者寫入 model_status.csv 並跳過，不得勉強輸出係數。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

MIN_OBS = 500
MIN_ENTITIES = 20
MIN_PERIODS = 20


@dataclass
class ModelResult:
    name: str
    status: str
    n_obs: int = 0
    n_entities: int = 0
    n_periods: int = 0
    params: dict = field(default_factory=dict)
    tstats: dict = field(default_factory=dict)
    pvalues: dict = field(default_factory=dict)
    stderr: dict = field(default_factory=dict)
    rsquared: float = np.nan
    note: str = ""
    tier_composition: dict = field(default_factory=dict)
    is_diagnostic: bool = True


def standardize_within(df: pd.DataFrame, cols: list[str],
                       group: str = "sparsity_tier") -> pd.DataFrame:
    """在稀疏度分層內標準化（PRD §5.1 第 1 點）。"""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        g = out.groupby(group)[col]
        mu, sd = g.transform("mean"), g.transform("std")
        out[col] = (out[col] - mu) / sd.replace(0, np.nan)
    return out


def available_controls(df: pd.DataFrame, controls: list[str],
                       min_coverage: float = 0.5) -> tuple[list[str], list[str]]:
    """把控制變數分成「可用」與「不可用」。

    PRD §3.8 的缺值政策不可妥協：未取得的控制變數維持缺欄位，不得補零或補均值。
    正式主表要求欄位齊備才估計（F6）；**診斷輸出**則以可用控制估計，並把被剔除的
    欄位記入 note，讓讀者知道這組係數少了什麼。
    """
    usable, dropped = [], []
    for col in controls:
        if col in df.columns and df[col].notna().mean() >= min_coverage:
            usable.append(col)
        else:
            dropped.append(col)
    return usable, dropped


def _prepare(df: pd.DataFrame, y: str, xs: list[str],
             controls: list[str]) -> pd.DataFrame:
    cols = ["ticker", "week", y, *xs, *controls]
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return d


def fit_twoway(df: pd.DataFrame, y: str, xs: list[str], controls: list[str],
               name: str, is_diagnostic: bool = True,
               cluster_time: bool = True) -> ModelResult:
    """個股 ＋ 週雙向固定效果，個股與週雙重 cluster。

    診斷模式下，涵蓋率不足的控制變數會被剔除並記入 note；正式模式（is_diagnostic
    為 False）則要求控制變數齊備，否則直接中止（PRD §6.2 F6）。
    """
    controls, dropped = available_controls(df, controls)
    if dropped and not is_diagnostic:
        return ModelResult(name, "SKIPPED",
                           note=f"正式規格要求控制變數齊備，缺：{';'.join(dropped)}")
    d = _prepare(df, y, xs, controls)
    if d.empty:
        return ModelResult(name, "SKIPPED", note="無有效觀測")

    n_obs, n_ent, n_per = len(d), d["ticker"].nunique(), d["week"].nunique()
    if n_obs < MIN_OBS or n_ent < MIN_ENTITIES or n_per < MIN_PERIODS:
        return ModelResult(name, "SKIPPED", n_obs, n_ent, n_per,
                           note=f"樣本不足（門檻 obs≥{MIN_OBS}, 個股≥{MIN_ENTITIES}, "
                                f"週≥{MIN_PERIODS}）")

    tiers = {}
    if "sparsity_tier" in df.columns:
        sub = df.loc[d.index, "sparsity_tier"] if d.index.isin(df.index).all() else None
        if sub is not None:
            tiers = sub.value_counts().to_dict()

    panel = d.set_index(["ticker", "week"])
    exog = panel[[*xs, *[c for c in controls if c in panel.columns]]]
    # 常數項與零變異欄位會使雙向 FE 不可識別
    exog = exog.loc[:, exog.std() > 0]
    if exog.empty:
        return ModelResult(name, "SKIPPED", n_obs, n_ent, n_per, note="自變數無變異")

    try:
        mod = PanelOLS(panel[y], exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True, check_rank=False)
        res = mod.fit(cov_type="clustered", cluster_entity=True,
                      cluster_time=cluster_time)
    except Exception as exc:  # noqa: BLE001
        return ModelResult(name, "FAILED", n_obs, n_ent, n_per,
                           note=f"{type(exc).__name__}: {exc}"[:200])

    return ModelResult(
        name=name, status="OK", n_obs=int(res.nobs), n_entities=n_ent, n_periods=n_per,
        params={k: float(v) for k, v in res.params.items()},
        tstats={k: float(v) for k, v in res.tstats.items()},
        pvalues={k: float(v) for k, v in res.pvalues.items()},
        stderr={k: float(v) for k, v in res.std_errors.items()},
        rsquared=float(res.rsquared_within),
        note=("控制變數缺漏（診斷）：" + ";".join(dropped)) if dropped else "",
        tier_composition={str(k): int(v) for k, v in tiers.items()},
        is_diagnostic=is_diagnostic,
    )


def results_to_frame(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.status != "OK":
            rows.append({"model": r.name, "status": r.status, "term": "",
                         "coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                         "n_obs": r.n_obs, "n_entities": r.n_entities,
                         "n_periods": r.n_periods, "r2_within": np.nan,
                         "diagnostic": r.is_diagnostic, "note": r.note,
                         "tier_composition": ""})
            continue
        for term in r.params:
            rows.append({
                "model": r.name, "status": r.status, "term": term,
                "coef": r.params[term], "se": r.stderr[term],
                "t": r.tstats[term], "p": r.pvalues[term],
                "n_obs": r.n_obs, "n_entities": r.n_entities,
                "n_periods": r.n_periods, "r2_within": r.rsquared,
                "diagnostic": r.is_diagnostic, "note": r.note,
                "tier_composition": ";".join(
                    f"{k}={v}" for k, v in sorted(r.tier_composition.items())),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 規格
# ---------------------------------------------------------------------------

BASE_CONTROLS = [
    "att_zero_base", "ret_lag1", "ret_lag4", "ret_lag25",
    "log_market_cap", "turnover", "amihud", "foreign_holding_pct",
    "listing_age_years",
]


def add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["log_market_cap"] = np.log(d["market_cap"].where(d["market_cap"] > 0))
    d["log_att_mean_level_52"] = np.log1p(d["att_mean_level_52"])
    d["has_analyst_coverage"] = np.nan   # 授權資料未取得（PRD §3.8、§4.5）
    d["news_count"] = np.nan             # Yahoo 新聞管道暫緩（見 LIMITATIONS）
    return d


def main_regressions(panel: pd.DataFrame, controls: list[str] | None = None,
                     is_diagnostic: bool = True) -> list[ModelResult]:
    """PRD §5.2 的四個規格 ＋ 交易時段切法。"""
    controls = controls if controls is not None else BASE_CONTROLS
    d = standardize_within(panel, [
        "abn_attention_all", "abn_attention_weekday", "abn_attention_weekend",
        "abn_attention_intraday", "abn_attention_non_trading",
        *[c for c in controls if c != "att_zero_base"],
    ])
    specs = [
        ("T3-1 整週", ["abn_attention_all"]),
        ("T3-2 僅週間", ["abn_attention_weekday"]),
        ("T3-3 僅週末", ["abn_attention_weekend"]),
        ("T3-4 週間＋週末", ["abn_attention_weekday", "abn_attention_weekend"]),
        ("T3-5 交易時段切法", ["abn_attention_intraday", "abn_attention_non_trading"]),
    ]
    return [fit_twoway(d, "ret_next", xs, controls, name, is_diagnostic)
            for name, xs in specs]


def mechanism_regressions(panel: pd.DataFrame, controls: list[str] | None = None,
                          is_diagnostic: bool = True) -> list[ModelResult]:
    """PRD §5.3：以 non_inst_roi_next 與 turnover_next 取代應變數，各加落後應變數。"""
    controls = controls if controls is not None else BASE_CONTROLS
    d = standardize_within(panel, [
        "abn_attention_weekday", "abn_attention_weekend",
        *[c for c in controls if c != "att_zero_base"],
    ])
    xs = ["abn_attention_weekday", "abn_attention_weekend"]
    out = []
    out.append(fit_twoway(d, "non_inst_roi_next", xs,
                          [*controls, "non_inst_roi_lag1"],
                          "T5-1 非三大法人 ROI", is_diagnostic))
    out.append(fit_twoway(d, "turnover_next", xs,
                          [*controls, "turnover_lag1"],
                          "T5-2 異常周轉率", is_diagnostic))
    return out


def effort_regressions(panel_dense: pd.DataFrame, controls: list[str] | None = None,
                       is_diagnostic: bool = True) -> list[ModelResult]:
    """PRD §5.4(a)：認知投入分解。**限定於 dense 子樣本**。"""
    controls = controls if controls is not None else BASE_CONTROLS
    cols = [f"abn_attention_{e}_{w}" for e in
            ("high_effort", "mid_effort", "low_effort") for w in ("weekend", "weekday")]
    d = standardize_within(panel_dense,
                           [*cols, *[c for c in controls if c != "att_zero_base"]])
    out = []
    for effort in ("high_effort", "mid_effort", "low_effort"):
        xs = [f"abn_attention_{effort}_weekday", f"abn_attention_{effort}_weekend"]
        out.append(fit_twoway(d, "ret_next", xs, controls,
                              f"T4-{effort} 次週報酬（dense）", is_diagnostic))
    xs_all = [f"abn_attention_{e}_weekend" for e in
              ("high_effort", "mid_effort", "low_effort")]
    out.append(fit_twoway(d, "ret_next", xs_all, controls,
                          "T4-同時 三層週末關注度（dense）", is_diagnostic))
    return out
