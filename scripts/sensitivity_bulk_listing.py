"""大量清單型貼文門檻的敏感度測試（穩健性 R15）。

不處理大量清單型貼文時，2.8% 的文章會貢獻 36.8% 的配對列，並在覆蓋熱圖上造出
2018–19 的假垂直帶。這個門檻是本專案最大的單一測度自由度，因此必須做敏感度測試
並完整報告——與稀疏度 tier 門檻同等對待（PRD §3.2.1 第 4 點的精神）。

對每個門檻重建關注度面板並重跑主迴歸，輸出 output/tables/R15_bulk_listing.csv。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.analysis.regressions import (
    BASE_CONTROLS, add_derived, fit_twoway, results_to_frame, standardize_within,
)
from src.features.attention import build_attention_panel
from src.features.sessions import week_of


def run(thresholds: list[int] | None = None) -> pd.DataFrame:
    settings = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    thresholds = thresholds or settings["attention"].get(
        "bulk_listing_sensitivity_grid", [5, 10, 15, 20, 30, 999])

    matches = pd.read_parquet("data/interim/ptt_matches.parquet")
    panel_full = pd.read_parquet("data/processed/panel.parquet")
    uni = pd.read_csv("data/external/universe.csv", dtype={"ticker": str})

    smp = settings["sample"]
    weeks = pd.date_range(week_of(pd.Timestamp(smp["main_start"])),
                          week_of(pd.Timestamp(smp["main_end"])), freq="7D")
    tickers = uni["ticker"].tolist()

    # 市場面與控制變數只取一次，逐門檻只換關注度欄位
    market_cols = [c for c in panel_full.columns
                   if not c.startswith(("att_", "abn_attention", "is_initiation"))
                   and c not in ("sparsity_tier",)]
    market = panel_full[market_cols].copy()

    results, rows = [], []
    for k in thresholds:
        sub = matches[matches["n_tickers_in_article"] <= k]
        att = build_attention_panel(sub, weeks, tickers, settings).reset_index()
        d = market.merge(att, on=["ticker", "week"], how="inner")
        d = add_derived(d)
        d = d[d["sparsity_tier"].isin(["dense", "sparse"])]
        std = standardize_within(d, ["abn_attention_weekday", "abn_attention_weekend",
                                     *[c for c in BASE_CONTROLS
                                       if c != "att_zero_base"]])
        label = f"R15 門檻 ≤{k} 檔/篇" + ("（不過濾）" if k >= 999 else "")
        res = fit_twoway(std, "ret_next",
                         ["abn_attention_weekday", "abn_attention_weekend"],
                         BASE_CONTROLS, label)
        results.append(res)
        rows.append({
            "threshold": k,
            "n_match_rows_kept": len(sub),
            "share_kept": len(sub) / len(matches),
            "n_dense_tickers": int(
                d[d["sparsity_tier"] == "dense"]["ticker"].nunique()),
            "status": res.status,
            "beta_weekend": res.params.get("abn_attention_weekend"),
            "t_weekend": res.tstats.get("abn_attention_weekend"),
            "beta_weekday": res.params.get("abn_attention_weekday"),
            "t_weekday": res.tstats.get("abn_attention_weekday"),
            "n_obs": res.n_obs,
        })
        print(f"  門檻 ≤{k}: 保留 {len(sub):,} 列（{len(sub)/len(matches):.1%}），"
              f"β_weekend={rows[-1]['beta_weekend']}, t={rows[-1]['t_weekend']}",
              flush=True)

    out = pd.DataFrame(rows)
    Path("output/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv("output/tables/R15_bulk_listing_sensitivity.csv", index=False)
    results_to_frame(results).to_csv(
        "output/tables/R15_bulk_listing_models.csv", index=False)
    return out


if __name__ == "__main__":
    print(run().to_string(index=False))
