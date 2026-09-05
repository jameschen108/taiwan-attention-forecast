"""投資組合排序（PRD §5.7，v2.0 升格為主結果之一）。

267 檔使每組約 53 檔，統計上遠比 v1.0 的每組 10 檔可信。

**呈現紀律**（PRD §5.7、§11）：
- 宇宙為 ex-post 選出，一律標示 ex-post universe，不得宣稱為可實作策略。
- 可排序週若不連續，「每週平均 × 52」只是機械年化，不可解讀為可投資績效。
- **不含成本的多空價差不得作為主要結論陳述。**
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def round_trip_cost(fee_rate: float, fee_discount: float, tax_rate: float,
                    slippage_bps: float) -> float:
    """一次完整多空換手的單邊成本估計。

    手續費買賣各一次（可打折）＋ 賣出證交稅 ＋ 買賣兩邊各一次滑價。
    """
    fees = fee_rate * fee_discount * 2
    slippage = (slippage_bps / 1e4) * 2
    return fees + tax_rate + slippage


def quantile_portfolios(panel: pd.DataFrame, signal: str = "abn_attention_weekend",
                        ret_col: str = "ret_next", n_q: int = 5,
                        weight: str = "equal",
                        tradability_filter: bool = False,
                        min_weekly_value: float = 5_000_000,
                        min_names_per_bucket: int = 5) -> pd.DataFrame:
    """依前週週末關注度分 n 等分，回傳每週各組報酬與多空價差。"""
    cols = ["ticker", "week", signal, ret_col, "market_cap", "value"]
    d = panel[[c for c in cols if c in panel.columns]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[signal, ret_col])

    if tradability_filter:
        # 剔除該週日均成交金額低於門檻者（PRD §5.7）
        d = d[d["value"].fillna(0) >= min_weekly_value]

    rows = []
    for week, g in d.groupby("week"):
        if len(g) < n_q * min_names_per_bucket:
            continue
        try:
            g = g.assign(q=pd.qcut(g[signal].rank(method="first"), n_q,
                                   labels=range(1, n_q + 1)))
        except ValueError:
            continue
        rec = {"week": week, "n_names": len(g)}
        for q, gq in g.groupby("q", observed=True):
            if weight == "value" and gq["market_cap"].notna().any():
                w = gq["market_cap"].fillna(0)
                r = float((gq[ret_col] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
            else:
                r = float(gq[ret_col].mean())
            rec[f"q{int(q)}"] = r
            rec[f"n_q{int(q)}"] = len(gq)
        if f"q{n_q}" in rec and "q1" in rec:
            rec["long_short"] = rec[f"q{n_q}"] - rec["q1"]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def summarize(port: pd.DataFrame, settings: dict, label: str,
              n_q: int = 5) -> dict:
    """含成本前／成本後的摘要。成本後為必要欄，不是附錄。"""
    if port.empty or "long_short" not in port.columns:
        return {"portfolio": label, "status": "SKIPPED",
                "note": "可排序週不足，無法建組"}

    ls = port["long_short"].dropna()
    cfg = settings["portfolio"]
    cost = round_trip_cost(cfg["fee_rate"], cfg["fee_discount"],
                           cfg["tax_rate"], cfg["slippage_bps"])
    # 多空兩腳每週各換手一次
    weekly_cost = cost * 2
    net = ls - weekly_cost

    def stats(s: pd.Series) -> tuple[float, float, float]:
        mean = float(s.mean())
        t = float(mean / (s.std(ddof=1) / np.sqrt(len(s)))) if s.std(ddof=1) > 0 else np.nan
        return mean, t, float(s.std(ddof=1))

    m_g, t_g, sd = stats(ls)
    m_n, t_n, _ = stats(net)

    weeks = pd.to_datetime(port["week"])
    contiguous = bool((weeks.diff().dropna() == pd.Timedelta(days=7)).all())

    return {
        "portfolio": label,
        "status": "OK",
        "n_weeks": len(ls),
        "weeks_contiguous": contiguous,
        "mean_ls_gross_weekly": m_g,
        "t_gross": t_g,
        "mean_ls_net_weekly": m_n,
        "t_net": t_n,
        "weekly_cost_assumed": weekly_cost,
        "sd_weekly": sd,
        "mechanical_annualized_gross": m_g * 52,
        "mechanical_annualized_net": m_n * 52,
        "universe": "ex-post universe",
        "caveat": ("ex-post universe 之機械年化多空價差；不可解讀為年化績效、"
                   "可投資報酬或策略績效（PRD §11）"),
    }


def run_all(panel: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PRD §8.3：同時報告成本前／成本後、篩選前／篩選後四個版本。"""
    cfg = settings["portfolio"]
    specs = [
        ("等權・未篩選", "equal", False),
        ("等權・可交易性篩選後", "equal", True),
        ("市值加權・未篩選", "value", False),
        ("市值加權・可交易性篩選後", "value", True),
    ]
    summaries, series = [], []
    for label, weight, filt in specs:
        port = quantile_portfolios(
            panel, n_q=cfg["n_quantiles"], weight=weight,
            tradability_filter=filt,
            min_weekly_value=cfg["tradability_min_daily_value_twd"] * 5)
        summaries.append(summarize(port, settings, label, cfg["n_quantiles"]))
        if not port.empty:
            series.append(port.assign(spec=label))

    # 依市值中位數雙重排序（PRD §5.7）
    if "market_cap" in panel.columns and panel["market_cap"].notna().any():
        med = panel.groupby("week")["market_cap"].transform("median")
        for side, mask in (("小型股", panel["market_cap"] <= med),
                           ("大型股", panel["market_cap"] > med)):
            port = quantile_portfolios(panel[mask], n_q=cfg["n_quantiles"])
            summaries.append(summarize(port, settings, f"雙重排序・{side}",
                                       cfg["n_quantiles"]))
            if not port.empty:
                series.append(port.assign(spec=f"雙重排序・{side}"))

    all_series = (pd.concat(series, ignore_index=True) if series
                  else pd.DataFrame())
    return pd.DataFrame(summaries), all_series
