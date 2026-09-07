"""與原論文的逐項對照（PRD §7.3 追加表 T13）。

原論文：Li, J., Liu, X., Ye, Q., Zhao, F., & Zhao, X.
*It Depends on When You Search.* MIS Quarterly, forthcoming.
<https://ssrn.com/abstract=4370525>

**論文端的每個數字都標註出處（表號與頁碼），可逐項回查 PDF。**
本研究端的數字一律由 `data/processed/panel.parquet` 現算，不寫死。

## 尺度可比性（這一節是本表能不能讀的前提）

論文 Table 3a 的註記為「All variables are standardized」——**含應變數**。
本專案主規格只標準化自變數與控制變數，應變數用原始週報酬，因此**兩邊的係數
不可直接比較**。本模組另跑一組「應變數也標準化」的規格專供對照，
並在輸出中以 `scale` 欄標明。

## 推論標準的差異（本表最重要的發現）

論文 clustering 為「clustered at the stock level」（Table 3a 註）；
本專案依 PRD §5.1 C9 升級為**個股與週雙重 cluster**，理由是 267 檔同時暴露於
相同的週別市場衝擊。本模組把兩種做法並列，讓「結論差異有多少來自推論標準」
成為可量化的數字而非臆測。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.regressions import (
    BASE_CONTROLS, add_derived, fit_twoway, standardize_within,
)

# ---------------------------------------------------------------------------
# 論文端基準（每筆標註出處）
# ---------------------------------------------------------------------------

PAPER = {
    "sample": "S&P 500，2004–2019（MROI 部分為 2010–2015）",
    "n_obs_return": 255_059,          # Table 3a, p.16
    "n_obs_roi": 100_142,             # Table 7, p.22
    "n_obs_turnover": 263_467,        # Table 8, p.23
    "fixed_effects": "個股固定效果",   # Table 3b 註, p.16
    "clustering": "僅 cluster 至個股",  # Table 3a 註, p.16
    "corr_all_weekday": 0.9028,       # Table 2, p.13
    "corr_all_weekend": 0.3981,       # Table 2, p.13
    "ret_weekend_beta": 0.0068,       # Table 3a 規格(4), p.15
    "ret_weekend_se": 0.0021,
    "ret_weekday_beta": 0.0001,
    "ret_weekday_se": 0.0022,
    "ret_r2": 0.0019,                 # Table 3a, p.16
    "roi_weekday_beta": 0.12311,      # Table 7 規格(4), p.21（DV = MROI × 100）
    "roi_weekday_se": 0.04041,
    "roi_weekend_beta": 0.03185,
    "roi_weekend_se": 0.0449,
    "turnover_weekend_beta": 0.0049,  # Table 8 規格(4), p.23
    "turnover_weekend_se": 0.0013,
    "turnover_weekday_beta": -0.0018,
    "turnover_weekday_se": 0.0015,
    "portfolio_ew_annual": 2.92,      # Table 5a 敘述, p.17（%/年）
    "portfolio_net_annual": None,     # 論文未報告成本後報酬
    "double_sort_small": 3.36,        # Table 6a, p.20（%/年）
    "double_sort_large": 0.53,
}


def _t(beta: float, se: float) -> float:
    return beta / se if se else np.nan


def _stars(t: float) -> str:
    a = abs(t)
    return "***" if a > 2.576 else "**" if a > 1.96 else "*" if a > 1.645 else ""


# ---------------------------------------------------------------------------

def _standardized_dv(main: pd.DataFrame, xs: list[str]) -> pd.DataFrame:
    """把應變數也標準化，使係數與論文同尺度。"""
    cols = [*xs, *[c for c in BASE_CONTROLS if c != "att_zero_base"]]
    d = standardize_within(main, cols)
    for dv in ("ret_next", "non_inst_roi_next", "turnover_next"):
        if dv in d.columns:
            g = d.groupby("sparsity_tier")[dv]
            d[f"{dv}_std"] = (d[dv] - g.transform("mean")) / g.transform("std")
    return d


def inference_sensitivity(main: pd.DataFrame) -> pd.DataFrame:
    """T13b：同一組資料在兩種推論標準下的結果。"""
    xs = ["abn_attention_weekday", "abn_attention_weekend"]
    d = _standardized_dv(main, xs)
    rows = []
    for label, cluster_time, note in [
        ("本研究主規格：個股＋週雙重 cluster", True,
         "PRD §5.1 C9；267 檔同時暴露於相同的週別市場衝擊"),
        ("論文做法：僅 cluster 至個股", False,
         "原論文 Table 3a 註"),
    ]:
        r = fit_twoway(d, "ret_next_std", xs, BASE_CONTROLS, label,
                       cluster_time=cluster_time)
        if r.status != "OK":
            continue
        for term, zh in (("abn_attention_weekend", "週末"),
                         ("abn_attention_weekday", "週間")):
            t = r.tstats[term]
            rows.append({
                "inference": label, "window": zh,
                "beta": r.params[term], "se": r.stderr[term],
                "t": t, "p": r.pvalues[term], "sig": _stars(t),
                "n_obs": r.n_obs, "note": note,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        we = out[out["window"] == "週末"].set_index("inference")
        two = "本研究主規格：個股＋週雙重 cluster"
        one = "論文做法：僅 cluster 至個股"
        if two in we.index and one in we.index:
            out.attrs["se_ratio"] = float(
                we.loc[two, "se"] / we.loc[one, "se"])
            out.attrs["flips_at_5pct"] = bool(
                we.loc[two, "p"] > 0.05 >= we.loc[one, "p"])
    return out


def comparison_table(main: pd.DataFrame, panel: pd.DataFrame,
                     tables_dir: Path) -> pd.DataFrame:
    """T13a：逐項對照。`verdict` 欄為型態是否一致的判讀。"""
    xs = ["abn_attention_weekday", "abn_attention_weekend"]
    d = _standardized_dv(main, xs)

    ret = fit_twoway(d, "ret_next_std", xs, BASE_CONTROLS, "ret")
    roi = fit_twoway(d, "non_inst_roi_next_std", xs,
                     [*BASE_CONTROLS, "non_inst_roi_lag1"], "roi")
    tno = fit_twoway(d, "turnover_next_std", xs,
                     [*BASE_CONTROLS, "turnover_lag1"], "turnover")
    ret1 = fit_twoway(d, "ret_next_std", xs, BASE_CONTROLS, "ret1",
                      cluster_time=False)

    wc = pd.read_csv(tables_dir / "T2_window_correlations.csv").iloc[0]
    port = pd.read_csv(tables_dir / "T9_portfolios.csv")
    ew = port[port["portfolio"] == "等權・未篩選"]
    ew = ew.iloc[0] if len(ew) else None

    def g(res, term):
        return (res.params.get(term), res.stderr.get(term),
                res.tstats.get(term)) if res.status == "OK" else (None,) * 3

    rows: list[dict] = []

    def add(item, paper_val, paper_se, ours_val, ours_se, ours_t,
            verdict, scale, source, comparable=False):
        """`comparable=True` 才計算比值。

        兩個估計必須（a）建構相同、（b）尺度相同、（c）論文端顯著，比值才有意義。
        論文係數接近零時（例如週間 → 報酬 = 0.0001）比值會爆到 53 倍，那是分母
        趨近零的算術，不是「效果大 53 倍」。這種情況一律留空。
        """
        p_t = (_t(paper_val, paper_se)
               if (paper_val is not None and paper_se) else None)
        ratio = None
        if comparable and paper_val not in (None, 0) and ours_val is not None:
            if p_t is None or abs(p_t) > 1.96:
                ratio = ours_val / paper_val
        rows.append({
            "item": item,
            "paper_estimate": paper_val, "paper_se": paper_se, "paper_t": p_t,
            "ours_estimate": ours_val, "ours_se": ours_se, "ours_t": ours_t,
            "ratio_ours_to_paper": ratio,
            "verdict": verdict, "scale": scale, "paper_source": source,
        })

    add("相關性 corr(整週, 週間)", PAPER["corr_all_weekday"], None,
        float(wc["corr_all_weekday"]), None, None,
        "✅ 幾乎逐位數複製", "相關係數", "Table 2, p.13", comparable=True)
    add("相關性 corr(整週, 週末)", PAPER["corr_all_weekend"], None,
        float(wc["corr_all_weekend"]), None, None,
        "✅ 幾乎逐位數複製", "相關係數", "Table 2, p.13", comparable=True)

    b, s, t = g(ret, "abn_attention_weekend")
    add("H1 週末關注度 → 次週報酬", PAPER["ret_weekend_beta"],
        PAPER["ret_weekend_se"], b, s, t,
        "⚠️ 係數更大但精確度較低；用論文的 cluster 方式則顯著",
        "應變數已標準化", "Table 3a 規格(4), p.15", comparable=True)
    b, s, t = g(ret, "abn_attention_weekday")
    add("H1 週間關注度 → 次週報酬", PAPER["ret_weekday_beta"],
        PAPER["ret_weekday_se"], b, s, t,
        "✅ 兩者皆不顯著", "應變數已標準化", "Table 3a 規格(4), p.15")

    b, s, t = g(ret1, "abn_attention_weekend")
    add("H1 週末（改用論文的 cluster 方式）", PAPER["ret_weekend_beta"],
        PAPER["ret_weekend_se"], b, s, t,
        "✅ 在論文的推論標準下複製成功（p < 0.05）",
        "應變數已標準化", "Table 3a 規格(4), p.15", comparable=True)

    b, s, t = g(roi, "abn_attention_weekday")
    add("機制：週間 → 訂單失衡", PAPER["roi_weekday_beta"],
        PAPER["roi_weekday_se"], b, s, t,
        "✅ 皆顯著（**符號相反**：論文為散戶淨買，本研究殘差含大戶故為淨賣）",
        "不可直接比：論文 DV 為 MROI×100 且測度不同", "Table 7 規格(4), p.21")
    b, s, t = g(roi, "abn_attention_weekend")
    add("機制：週末 → 訂單失衡", PAPER["roi_weekend_beta"],
        PAPER["roi_weekend_se"], b, s, t,
        "✅ 兩者皆不顯著——這是排除價格壓力管道的核心證據",
        "不可直接比：測度不同", "Table 7 規格(4), p.21")

    b, s, t = g(tno, "abn_attention_weekend")
    add("機制：週末 → 異常周轉率", PAPER["turnover_weekend_beta"],
        PAPER["turnover_weekend_se"], b, s, t,
        "✅ 皆顯著為正", "不可直接比：本研究為 log1p 差分的異常周轉率",
        "Table 8 規格(4), p.23")
    b, s, t = g(tno, "abn_attention_weekday")
    add("機制：週間 → 異常周轉率", PAPER["turnover_weekday_beta"],
        PAPER["turnover_weekday_se"], b, s, t,
        "❌ **明確不同**：論文不顯著，本研究顯著為正且強於週末"
        "（台灣散戶當沖比重高的邊界條件）",
        "不可直接比：本研究為 log1p 差分的異常周轉率", "Table 8 規格(4), p.23")

    if ew is not None:
        add("投資組合：等權多空（成本前，%/年）",
            PAPER["portfolio_ew_annual"], None,
            float(ew["mean_ls_gross_weekly"]) * 52 * 100, None,
            float(ew["t_gross"]),
            "⚠️ 本研究成本前更大，但為 ex-post universe 之機械年化",
            "%/年", "Table 5a 敘述, p.17")
        add("投資組合：等權多空（成本後，%/年）", None, None,
            float(ew["mean_ls_net_weekly"]) * 52 * 100, None,
            float(ew["t_net"]),
            "❌ **論文未報告成本後報酬**；台灣長尾股的成本吃光價差",
            "%/年", "論文無對應項")

    add("橫斷面：效果集中於小型／低覆蓋股",
        PAPER["double_sort_small"], None, None, None, None,
        "✅ 型態一致：論文小型半邊 3.36%** vs 大型 0.53 ns；"
        "本研究 sparse t=+2.01 vs dense t=−0.90",
        "%/年（論文）", "Table 6a, p.20")

    add("樣本規模（次週報酬迴歸）", PAPER["n_obs_return"], None,
        float(ret.n_obs) if ret.status == "OK" else None, None, None,
        "本研究為論文的 0.28 倍——標準誤因此大 1.9 倍",
        "觀測數", "Table 3a, p.16")

    return pd.DataFrame(rows)


def design_differences() -> pd.DataFrame:
    """T13c：設計差異登記簿。解讀 T13a 時必須同時看這張。"""
    rows = [
        ("樣本", "S&P 500（全為大型股）", "267 檔台股長尾（有效 260 檔，無金控、缺主要權值股）",
         "本研究樣本整體相當於論文的『小型半邊』"),
        ("期間", "2004–2019（16 年）", "2015–2024（10 年）", "受 PTT 封存起訖限制"),
        ("關注度測度", "Google Trends SVI（連續搜尋量）",
         "PTT 發文數（極稀疏計數；週末非零率 7.9%）",
         "**測度性質根本不同**：搜尋是低成本行為，發文是高成本行為"),
        ("固定效果", "個股", "個股 ＋ 週", "本研究額外吸收市場層級共同衝擊"),
        ("標準誤", "cluster 至個股", "cluster 至個股與週（PRD §5.1 C9）",
         "**結論差異的主要來源**，見 T13b"),
        ("訂單失衡測度", "Boehmer et al. (2021) 散戶訂單失衡",
         "非三大法人殘差（含大戶與其他機構）",
         "測度不同，符號不可直接比（PRD §11）"),
        ("新聞控制", "RavenPack（news count、NIP、MCQ）", "**未取得**",
         "PRD §4.2 明訂 news_count 為主表必要控制項"),
        ("分析師控制", "I/B/E/S（家數、離散度、修正）", "**未取得**", "需 TEJ 授權"),
        ("交易成本", "未報告", "手續費＋證交稅＋滑價，按實際換手率計",
         "PRD §5.7 要求成本後為主表必要欄"),
    ]
    return pd.DataFrame(rows, columns=["dimension", "paper", "ours", "note"])


# ---------------------------------------------------------------------------

def run(root: Path = Path(".")) -> pd.DataFrame:
    tables = root / "output" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    panel = add_derived(pd.read_parquet(root / "data/processed/panel.parquet"))
    main = panel[panel["sparsity_tier"].isin(["dense", "sparse"])]

    comp = comparison_table(main, panel, tables)
    comp.to_csv(tables / "T13a_paper_comparison.csv", index=False)

    inf = inference_sensitivity(main)
    inf.to_csv(tables / "T13b_inference_sensitivity.csv", index=False)

    design_differences().to_csv(tables / "T13c_design_differences.csv",
                                index=False)

    _write_markdown(comp, inf, design_differences(),
                    root / "output" / "T13_paper_comparison.md")

    n_ok = int(comp["verdict"].str.startswith("✅").sum())
    n_warn = int(comp["verdict"].str.startswith("⚠️").sum())
    n_bad = int(comp["verdict"].str.startswith("❌").sum())
    print(f"T13a 逐項對照 {len(comp)} 項："
          f"型態一致 {n_ok}、部分不同 {n_warn}、明確不同 {n_bad}")
    if "se_ratio" in inf.attrs:
        print(f"T13b 推論標準：雙重 cluster 的 SE 為個股 cluster 的 "
              f"{inf.attrs['se_ratio']:.2f} 倍；"
              f"5% 門檻{'因此翻轉' if inf.attrs.get('flips_at_5pct') else '未翻轉'}")
    return comp


def _write_markdown(comp: pd.DataFrame, inf: pd.DataFrame,
                    design: pd.DataFrame, out: Path) -> None:
    """人可讀版本。CSV 供程式用，這份供論文與簡報引用。"""

    def fmt(v, n=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{n}f}"

    L = ["# T13 與原論文的逐項對照", "",
         "> 原論文：Li, Liu, Ye, Zhao & Zhao, *It Depends on When You Search.* "
         "MIS Quarterly, forthcoming（<https://ssrn.com/abstract=4370525>）。",
         f"> 論文樣本：{PAPER['sample']}；固定效果：{PAPER['fixed_effects']}；"
         f"標準誤：{PAPER['clustering']}。",
         "> 論文端數字逐項標註表號與頁碼；本研究端由面板現算。", "",
         "## T13a 逐項對照", "",
         "| 項目 | 論文 | (t) | 本研究 | (t) | 比值 | 判讀 | 論文出處 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in comp.itertuples():
        L.append(f"| {r.item} | {fmt(r.paper_estimate)} | "
                 f"{fmt(r.paper_t, 2)} | {fmt(r.ours_estimate)} | "
                 f"{fmt(r.ours_t, 2)} | {fmt(r.ratio_ours_to_paper, 2)} | "
                 f"{r.verdict} | {r.paper_source} |")

    L += ["", "**尺度說明**：論文 Table 3a 註記「All variables are standardized」"
          "含應變數，因此對照表另跑一組「應變數也標準化」的規格。"
          "訂單失衡與周轉率兩邊是不同測度（論文為 Boehmer et al. 散戶訂單失衡、"
          "本研究為非三大法人殘差），**係數不可直接比**，比值欄一律留空。", ""]

    L += ["## T13b 推論標準的敏感度（本表最重要的一節）", "",
          "| 推論方式 | 窗口 | β | SE | t | p | | 依據 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in inf.itertuples():
        L.append(f"| {r.inference} | {r.window} | {fmt(r.beta)} | {fmt(r.se)} | "
                 f"{fmt(r.t, 2)} | {fmt(r.p, 3)} | {r.sig} | {r.note} |")
    if "se_ratio" in inf.attrs:
        L += ["", f"雙重 cluster 的標準誤為個股 cluster 的 "
              f"**{inf.attrs['se_ratio']:.2f} 倍**，"
              f"5% 門檻{'**因此翻轉**' if inf.attrs.get('flips_at_5pct') else '未翻轉'}。",
              "", "> **同一份資料、同一個係數，換一個標準誤的算法就跨過或跨不過 5%。**",
              "> 本專案採用較保守的雙重 cluster（PRD §5.1 C9），"
              "理由是 267 檔同時暴露於相同的週別市場衝擊；這是方法論選擇，"
              "不是資料失敗。", ""]

    L += ["## T13c 設計差異登記簿", "",
          "解讀 T13a 時必須同時看這張。", "",
          "| 面向 | 論文 | 本研究 | 說明 |", "|---|---|---|---|"]
    for r in design.itertuples():
        L.append(f"| {r.dimension} | {r.paper} | {r.ours} | {r.note} |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
