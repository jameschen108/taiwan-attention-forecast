"""一條指令重建全部輸出（PRD §6.1、§6.2 F1）。

管線階段：
  宇宙與市場別建立 → raw 稽核（含既有快取驗證）→ interim 重建
    → 兩市場資料合併 → full panel 合併 → analysis-readiness 報告
    → 診斷分析包 → 正式分析包 → 稀疏度與異質性報告

結束時印出每階段的模型估計數與驗收門檻通過數。每個階段可用 --only 獨立重跑。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(".")
CONFIG = ROOT / "config"
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
AUDIT = ROOT / "audit"
TABLES = ROOT / "output" / "tables"

STAGES = ["universe", "audit", "market", "ptt", "panel", "screen",
          "analysis", "report"]


def _settings() -> dict:
    return yaml.safe_load((CONFIG / "settings.yaml").read_text(encoding="utf-8"))


def stage_universe() -> None:
    from src.universe.build import build
    build(DATA / "universe_267.csv", DATA / "external", CONFIG, AUDIT)


def stage_audit(skip_ptt: bool = False) -> None:
    from src.audit_data import audit_market, audit_ptt
    uni = pd.read_csv(DATA / "external" / "universe.csv", dtype={"ticker": str})
    results = {}
    if not skip_ptt:
        results.update({f"ptt_{k}": v for k, v in
                        audit_ptt(DATA / "pttweb", AUDIT).items()})
    results.update({f"market_{k}": v for k, v in
                    audit_market(DATA / "raw" / "finmind", uni, AUDIT).items()})
    pd.DataFrame(sorted(results.items()), columns=["check", "value"]).to_csv(
        AUDIT / "raw_audit_summary.csv", index=False)


def stage_market() -> None:
    from src.market.normalize import build_daily_panel, build_trading_calendar, load_prices
    prices = load_prices(DATA / "raw" / "finmind")
    build_trading_calendar(prices, INTERIM / "trading_days.csv")
    build_daily_panel(
        DATA / "raw" / "finmind", INTERIM, AUDIT,
        t86_dir=DATA / "twse" / "t86",
        exrights_csv=INTERIM / "ex_rights.csv",
        shareholding_csv=INTERIM / "shareholding.csv",
        reduction_csv=INTERIM / "capital_reductions.csv",
    )
    # 第三方驗證：以既有 Yahoo 快取獨立檢核權值還原（PRD §3.6、§8.1）
    yahoo_root = DATA / "raw" / "yahoo"
    if yahoo_root.exists():
        from src.market.validate_prices import validate
        uni = pd.read_csv(DATA / "external" / "universe.csv", dtype={"ticker": str})
        listing = pd.to_datetime(uni.set_index("ticker")["listing_date"])
        frames = [pd.read_csv(p, dtype={"ticker": str}, parse_dates=["date"])
                  [["ticker", "date"]]
                  for p in (INTERIM / "ex_rights.csv",
                            INTERIM / "capital_reductions.csv") if p.exists()]
        events = (pd.concat(frames, ignore_index=True) if frames
                  else pd.DataFrame(columns=["ticker", "date"]))
        daily = pd.read_parquet(INTERIM / "market_daily.parquet",
                                columns=["ticker", "date", "close", "adj_close"])
        validate(daily, yahoo_root, events, AUDIT, listing)


def stage_ptt() -> None:
    from src.ptt.transform import build_matches
    build_matches(DATA / "pttweb", CONFIG / "universe.yaml",
                  INTERIM / "trading_days.csv",
                  INTERIM / "ptt_matches.parquet", AUDIT)


def stage_panel() -> None:
    from src.features.build import build_panel
    build_panel(INTERIM / "ptt_matches.parquet", INTERIM / "market_daily.parquet",
                DATA / "external" / "universe.csv", INTERIM / "trading_days.csv",
                CONFIG / "settings.yaml", PROCESSED, AUDIT)


def stage_screen() -> None:
    from src.universe.screen import run
    run(INTERIM / "ptt_matches.parquet", PROCESSED / "panel.parquet",
        CONFIG / "universe.yaml", AUDIT)


def stage_analysis() -> dict:
    from src.analysis import (
        events, heterogeneity, makeup_days, matched_events, portfolios,
        regressions, robustness, sector,
    )
    settings = _settings()
    TABLES.mkdir(parents=True, exist_ok=True)

    panel = regressions.add_derived(pd.read_parquet(PROCESSED / "panel.parquet"))
    # 主迴歸樣本限定 dense + sparse（PRD §3.2.1 規則 1）
    main = panel[panel["sparsity_tier"].isin(["dense", "sparse"])]
    dense = panel[panel["sparsity_tier"] == "dense"]
    counts: dict[str, int] = {}

    def dump(results, name: str) -> None:
        df = regressions.results_to_frame(results)
        df.to_csv(TABLES / f"{name}.csv", index=False)
        counts[name] = int((df["status"] == "OK").sum() and df["model"].nunique())
        ok = df.loc[df["status"] == "OK", "model"].nunique()
        skipped = df.loc[df["status"] != "OK", "model"].nunique()
        counts[name] = ok
        print(f"  {name}: 估計 {ok} 個規格，跳過 {skipped} 個")

    print("[T3] 主迴歸")
    dump(regressions.main_regressions(main), "T3_main_regressions")
    print("[T5] 機制檢定")
    dump(regressions.mechanism_regressions(main), "T5_mechanism")
    print("[T4] 認知投入分解（dense）")
    dump(regressions.effort_regressions(dense), "T4_effort")

    print("[T6] 異質性 ＋ 反轉（聯立）")
    inter = heterogeneity.interaction_models(main)
    terc = heterogeneity.tercile_models(main)
    rev_tier = heterogeneity.reversal_models(main, by="sparsity_tier")
    rev_all = heterogeneity.reversal_models(main)
    dump(inter + terc + rev_tier + rev_all, "T6_heterogeneity_reversal")
    heterogeneity.joint_reading(inter, rev_tier + rev_all).to_csv(
        TABLES / "T6_joint_reading.csv", index=False)
    fm = heterogeneity.fama_macbeth(
        main, ["abn_attention_weekend", "abn_attention_weekday"])
    if not fm.empty:
        fm.to_csv(TABLES / "T6_fama_macbeth.csv", index=False)

    print("[T7] 起始事件")
    events.initiation_feasibility(panel).to_csv(
        TABLES / "T7_initiation_feasibility.csv", index=False)
    dump(events.initiation_regressions(panel), "T7_initiation")
    car = events.car_path(panel)
    car.to_csv(TABLES / "T7_car_path.csv", index=False)
    if not car.empty:
        print(f"  CAR 路徑：{car.attrs.get('n_events_total', 0)} 個事件，"
              f"平行趨勢 {'通過' if car.attrs.get('parallel_trend_ok') else '未通過'}")

    print("[T7M] 起始事件・匹配對照（反向因果）")
    for col, tag in (("is_initiation", "all"),
                     ("is_initiation_weekend", "weekend")):
        res = matched_events.run(panel, col)
        if res.matched.empty:
            print(f"  [{tag}] {res.verdict}")
            continue
        res.unmatched.to_csv(TABLES / f"T7M_unmatched_{tag}.csv", index=False)
        res.matched.to_csv(TABLES / f"T7M_matched_{tag}.csv", index=False)
        res.balance.to_csv(TABLES / f"T7M_balance_{tag}.csv", index=False)
        res.clean.to_csv(TABLES / f"T7M_clean_{tag}.csv", index=False)
        if not res.propensity.empty:
            res.propensity.to_csv(TABLES / f"T7M_propensity_{tag}.csv", index=False)
        counts[f"T7M_{tag}"] = 1
        print(f"  [{tag}] 事件 {int(res.matched['n_events'].max())}、"
              f"週群集 {int(res.matched['n_week_clusters'].max())}；{res.verdict}")

    print("[T8] 產業")
    dump(sector.spillover_models(main), "T8_sector_spillover")
    sector.by_sector_coefficients(main).to_csv(
        TABLES / "T8_by_sector.csv", index=False)
    sector.sector_coverage_table(panel).to_csv(
        TABLES / "T1_sector_coverage.csv", index=False)

    print("[T9] 投資組合")
    summ, series = portfolios.run_all(main, settings)
    summ.to_csv(TABLES / "T9_portfolios.csv", index=False)
    if not series.empty:
        series.to_csv(TABLES / "T9_portfolio_weekly.csv", index=False)
    counts["T9_portfolios"] = int((summ["status"] == "OK").sum())

    print("[T10] 補班日")
    cal = pd.read_csv(INTERIM / "trading_days.csv", parse_dates=["date"])
    feas = makeup_days.feasibility(panel, cal)
    feas.to_csv(TABLES / "T10_makeup_feasibility.csv", index=False)
    print(f"  {feas.iloc[0]['verdict']}")
    mk = [makeup_days.interaction_model(main)]
    dump(mk, "T10_makeup")
    boot = makeup_days.wild_cluster_bootstrap(main)
    pd.DataFrame([boot]).to_csv(TABLES / "T10_makeup_bootstrap.csv", index=False)

    print("[T13] 與原論文對照")
    from src.analysis import paper_comparison
    comp = paper_comparison.run(ROOT)
    counts["T13_paper_comparison"] = len(comp)

    print("[T12] 穩健性")
    rob = robustness.run(panel, settings)
    robustness.summarize(rob).to_csv(TABLES / "T12_robustness.csv", index=False)
    counts["T12_robustness"] = int(sum(r.status == "OK" for r in rob))
    print(f"  穩健性 {counts['T12_robustness']}/{len(rob)} 項完成估計")

    return counts


def stage_report() -> pd.DataFrame:
    from src import audit_integrity
    from src.analysis import regressions, report

    # checksum 需讀取 1.4 GB，改由 `python3 -m src.audit_integrity` 單獨執行；
    # 已產出的 checksum 報告會被 health_checks 讀入。
    audit_integrity.title_parsing_accuracy(DATA / "pttweb", AUDIT)
    audit_integrity.lead_lag_verification(
        PROCESSED / "panel.parquet", AUDIT)
    TABLES.mkdir(parents=True, exist_ok=True)
    panel = regressions.add_derived(pd.read_parquet(PROCESSED / "panel.parquet"))
    uni = pd.read_csv(DATA / "external" / "universe.csv", dtype={"ticker": str})

    report.descriptive_stats(panel).to_csv(
        TABLES / "T1_descriptive_stats.csv", index=False)
    wc = report.window_correlations(panel)
    wc.to_csv(TABLES / "T2_window_correlations.csv", index=False)
    sp = report.sparsity_distribution(panel)
    sp.to_csv(AUDIT / "sparsity_distribution.csv", index=False)

    # 權值還原是否「已驗證」由**第三方對照**決定，不是自我宣告（PRD §3.6）
    val_path = AUDIT / "price_validation_yahoo.csv"
    verified = False
    if val_path.exists():
        val = pd.read_csv(val_path)
        checked = val[val["status"].isin(["PASS", "FAIL"])]
        if len(checked):
            verified = bool((checked["status"] == "PASS").mean() >= 0.95)

    readiness = report.analysis_readiness(panel, verified)
    readiness.to_csv(PROCESSED / "analysis_readiness.csv", index=False)

    screening = None
    sp_path = AUDIT / "keyword_screening_report.csv"
    if sp_path.exists():
        screening = pd.read_csv(sp_path, dtype={"ticker": str})

    hc = report.health_checks(panel, uni, screening, wc, sp, readiness)
    hc.to_csv(AUDIT / "health_checks.csv", index=False)

    from src.analysis import coverage_reports, figures
    made = figures.run(panel, TABLES, ROOT / "output" / "figures")
    print(f"圖：{', '.join(made)}")
    coverage_reports.run(ROOT)
    return hc


def main() -> None:
    parser = argparse.ArgumentParser(description="重建全部輸出")
    parser.add_argument("--only", nargs="*", choices=STAGES, default=None,
                        help="只跑指定階段（除錯用，PRD §6.2 F2）")
    parser.add_argument("--skip-ptt-audit", action="store_true",
                        help="跳過 PTT 封存的逐檔稽核（耗時約 2 分鐘）")
    args = parser.parse_args()

    stages = args.only or STAGES
    counts: dict[str, int] = {}
    t0 = time.time()

    for stage in STAGES:
        if stage not in stages:
            continue
        print(f"\n=== {stage} ===")
        started = time.time()
        if stage == "universe":
            stage_universe()
        elif stage == "audit":
            stage_audit(args.skip_ptt_audit)
        elif stage == "market":
            stage_market()
        elif stage == "ptt":
            stage_ptt()
        elif stage == "panel":
            stage_panel()
        elif stage == "screen":
            stage_screen()
        elif stage == "analysis":
            counts.update(stage_analysis())
        elif stage == "report":
            hc = stage_report()
            n_pass = int(hc["passed"].sum())
            print(f"\n驗收門檻：{n_pass}/{len(hc)} 通過")
            for _, r in hc[~hc["passed"]].iterrows():
                print(f"  ✗ {r['check']}（{r['prd_section']}）："
                      f"要求 {r['requirement']}，實得 {r['observed']}")
        print(f"--- {stage} 完成 {time.time() - started:.1f}s")

    if counts:
        print("\n=== 各表估計規格數 ===")
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v}")
    print(f"\n總耗時 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
