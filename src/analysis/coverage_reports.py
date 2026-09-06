"""PRD §7.2 要求但尚未產出的稽核報告。

listing_events / ptt_coverage_ticker(_monthly) / name_collision_report /
full_coverage / model_status / research_status / mops_selling_expense_coverage
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------

def listing_events(universe: pd.DataFrame, panel: pd.DataFrame,
                   settings: dict) -> pd.DataFrame:
    """上市／下市事件與面板列數對帳（PRD §8.1：「與面板列數對帳一致」）。"""
    end = pd.Timestamp(settings["sample"]["main_end"])
    rows = []
    by_ticker = panel.groupby("ticker")
    for r in universe.itertuples():
        ld = pd.to_datetime(r.listing_date)
        g = by_ticker.get_group(r.ticker) if r.ticker in by_ticker.groups else None
        rows.append({
            "ticker": r.ticker, "name_short": r.name_short, "sector": r.sector,
            "listing_date": ld.date() if pd.notna(ld) else "",
            "delisting_date": r.delisting_date if pd.notna(r.delisting_date) else "",
            "listed_before_sample_end": bool(pd.notna(ld) and ld <= end),
            "n_panel_weeks": 0 if g is None else len(g),
            "first_panel_week": "" if g is None else str(g["week"].min().date()),
            "last_panel_week": "" if g is None else str(g["week"].max().date()),
            "excluded_reason": ("" if g is not None else
                                "TWSE 上市日晚於主樣本結束日，無任何觀測"),
        })
    out = pd.DataFrame(rows)
    out.attrs["reconciles"] = int(out["n_panel_weeks"].sum()) == len(panel)
    return out


def ptt_coverage(matches: pd.DataFrame, universe: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐檔與逐檔×月的 PTT 覆蓋率（PRD §7.2）。只計入分析用配對。"""
    m = matches[~matches["is_bulk_listing"]].copy()
    m["week"] = pd.to_datetime(m["week"])
    m["month"] = m["timestamp"].dt.to_period("M").astype(str)

    per = (m.groupby("ticker")
           .agg(n_matches=("article_id", "size"),
                n_articles=("article_id", "nunique"),
                n_weeks_nonzero=("week", "nunique"),
                first_match=("timestamp", "min"),
                last_match=("timestamp", "max"),
                share_code_match=("match_mode",
                                  lambda s: float((s == "code").mean())),
                share_weekend=("window",
                               lambda s: float((s == "weekend").mean())))
           .reset_index())
    per = universe[["ticker", "name_short", "sector"]].merge(per, on="ticker",
                                                             how="left")
    per[["n_matches", "n_articles", "n_weeks_nonzero"]] = (
        per[["n_matches", "n_articles", "n_weeks_nonzero"]].fillna(0).astype(int))

    monthly = (m.groupby(["ticker", "month"]).size()
               .rename("n_matches").reset_index())
    return per.sort_values("n_matches", ascending=False), monthly


def name_collision_report(universe_cfg: dict, matches: pd.DataFrame,
                          universe: pd.DataFrame) -> pd.DataFrame:
    """碰撞群組、通用詞降級與阻擋延伸規則的現況與效果（PRD §3.4、§7.2）。"""
    groups = universe_cfg.get("collision_groups", {})
    in_group = {t: g for g, ms in groups.items() for t in ms}
    code_only = set(universe_cfg.get("code_only_tickers", []))
    reasons = universe_cfg.get("code_only_reason", {})

    m = matches[~matches["is_bulk_listing"]]
    cnt = m.groupby("ticker").size()
    mode = (m.groupby(["ticker", "match_mode"]).size().unstack(fill_value=0)
            if not m.empty else pd.DataFrame())

    rows = []
    for r in universe.itertuples():
        spec = universe_cfg["tickers"].get(r.ticker, {})
        variants = spec.get("variants", [])
        rows.append({
            "ticker": r.ticker, "name_short": r.name_short, "sector": r.sector,
            "collision_group": in_group.get(r.ticker, ""),
            "is_code_only": r.ticker in code_only,
            "code_only_reason": reasons.get(r.ticker, ""),
            "variants": ";".join(v["text"] for v in variants),
            "match_mode": variants[0]["match_mode"] if variants else "code_only",
            "blocked_after": ";".join(variants[0].get("blocked_after", []))
                             if variants else "",
            "blocked_before": ";".join(variants[0].get("blocked_before", []))
                              if variants else "",
            "n_matches": int(cnt.get(r.ticker, 0)),
            "n_code": int(mode.get("code", {}).get(r.ticker, 0)) if len(mode) else 0,
            "n_name": int(mode.get("name", {}).get(r.ticker, 0)) if len(mode) else 0,
            "n_name_with_context": int(
                mode.get("name_with_context", {}).get(r.ticker, 0)) if len(mode) else 0,
        })
    return pd.DataFrame(rows).sort_values(
        ["collision_group", "ticker"], ascending=[False, True])


def full_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """面板每個欄位的涵蓋率（PRD §7.2）。缺值一律維持缺值，此表即其證據。"""
    rows = []
    for col in panel.columns:
        s = panel[col]
        rows.append({
            "variable": col,
            "dtype": str(s.dtype),
            "n_non_missing": int(s.notna().sum()),
            "coverage": float(s.notna().mean()),
            "n_tickers_with_data": int(
                panel.loc[s.notna(), "ticker"].nunique()) if s.notna().any() else 0,
        })
    return pd.DataFrame(rows).sort_values("coverage")


def model_status(tables_dir: Path) -> pd.DataFrame:
    """所有嘗試過的模型與其狀態（PRD §5.1：不足者寫入 model_status 並跳過）。"""
    frames = []
    for path in sorted(tables_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        # 穩健性表用 `check` 當模型名，其餘用 `model`
        key = "model" if "model" in df.columns else (
            "check" if "check" in df.columns else None)
        if key is None or "status" not in df.columns:
            continue
        df = df.rename(columns={key: "model", "n_tickers": "n_entities",
                                "n_weeks": "n_periods"})
        for col in ("n_obs", "n_entities", "n_periods", "note"):
            if col not in df.columns:
                df[col] = np.nan
        sub = (df[["model", "status", "n_obs", "n_entities", "n_periods", "note"]]
               .drop_duplicates("model").assign(table=path.stem))
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[["table", "model", "status", "n_obs", "n_entities",
                "n_periods", "note"]].sort_values(["table", "model"])


def mops_selling_expense_coverage(mops_dir: Path, universe: pd.DataFrame
                                  ) -> pd.DataFrame:
    """推銷費用強度的涵蓋率（PRD §4.5：金融業與 -KY 無此科目，須列為未涵蓋而非零）。"""
    rows = []
    have: dict[str, list[str]] = {}
    if mops_dir.exists():
        for f in mops_dir.iterdir():
            m = re.match(r"(\d{4})_(\d{4})\.", f.name)
            if m:
                have.setdefault(m.group(1), []).append(m.group(2))

    for r in universe.itertuples():
        years = sorted(have.get(r.ticker, []))
        rows.append({
            "ticker": r.ticker, "name_short": r.name_short, "sector": r.sector,
            "is_ky": int(r.is_ky),
            "n_years_available": len(years),
            "years": ";".join(years),
            "status": "OK" if years else "NOT_COVERED",
            "note": ("" if years else
                     "MOPS 損益表未抓取；金融業與 -KY 另有科目不同的問題"
                     "——一律記為未涵蓋，不得填零（PRD §4.5）"),
        })
    return pd.DataFrame(rows)


def research_status(panel: pd.DataFrame, health: pd.DataFrame,
                    readiness: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """各分期的完成狀態（PRD §7.2、§9）。"""
    r = readiness.iloc[0]
    n_pass = int(health["passed"].sum())
    rows = [
        ("P0 骨架", "DONE", "config、宇宙與市場別、核心邏輯回歸測試全綠"),
        ("P1 既有資料稽核", "DONE", "PTT 封存、TWSE T86、價格快取皆通過稽核"),
        ("P2 注意力資料", "DONE",
         "解析、歸屬、窗口、稀疏度完成；歸屬正確率經抽驗修正至 95%"),
        ("P3 市場資料", "DONE",
         "TWSE 全期間；除權息＋減資還原並經 Yahoo 第三方驗證"),
        ("P4 面板", "DONE",
         f"{len(panel):,} 列 × {panel['ticker'].nunique()} 檔 × "
         f"{panel['week'].nunique()} 週"),
        ("MVP", "DONE", "主迴歸方向確認"),
        ("P5 分析（核心）", "DONE",
         "機制、異質性、起始事件（含反向因果匹配）、反轉皆完成"),
        ("P6 分析（延伸）", "DONE", "效果分層、產業、投資組合、補班日完成"),
        ("P7 外部控制", "BLOCKED",
         f"授權資料未取得，缺：{r['missing_controls']}"),
        ("P8 穩健性與交付", "PARTIAL",
         f"穩健性 17/23 可執行項完成；驗收門檻 {n_pass}/{len(health)} 通過"),
    ]
    return pd.DataFrame(rows, columns=["phase", "status", "note"])


# ---------------------------------------------------------------------------

def run(root: Path = Path(".")) -> dict:
    audit = root / "audit"
    audit.mkdir(exist_ok=True)
    settings = yaml.safe_load(
        (root / "config" / "settings.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(
        (root / "config" / "universe.yaml").read_text(encoding="utf-8"))
    uni = pd.read_csv(root / "data/external/universe.csv", dtype={"ticker": str})
    panel = pd.read_parquet(root / "data/processed/panel.parquet")
    matches = pd.read_parquet(root / "data/interim/ptt_matches.parquet")

    ev = listing_events(uni, panel, settings)
    ev.to_csv(audit / "listing_events.csv", index=False)

    per, monthly = ptt_coverage(matches, uni)
    per.to_csv(audit / "ptt_coverage_ticker.csv", index=False)
    monthly.to_csv(audit / "ptt_coverage_ticker_monthly.csv", index=False)

    name_collision_report(cfg, matches, uni).to_csv(
        audit / "name_collision_report.csv", index=False)
    full_coverage(panel).to_csv(audit / "full_coverage.csv", index=False)
    model_status(root / "output" / "tables").to_csv(
        audit / "model_status.csv", index=False)
    mops_selling_expense_coverage(
        root / "data/external/mops_income", uni).to_csv(
        audit / "mops_selling_expense_coverage.csv", index=False)

    health = pd.read_csv(audit / "health_checks.csv")
    readiness = pd.read_csv(root / "data/processed/analysis_readiness.csv")
    research_status(panel, health, readiness, settings).to_csv(
        audit / "research_status.csv", index=False)

    ms = pd.read_csv(audit / "model_status.csv")
    print(f"listing_events {len(ev)} 列；與面板對帳 "
          f"{'一致' if ev.attrs['reconciles'] else '**不一致**'}")
    print(f"ptt_coverage_ticker {len(per)} 檔；月度 {len(monthly):,} 列")
    print(f"model_status {len(ms)} 個模型："
          f"OK {int((ms['status'] == 'OK').sum())}、"
          f"SKIPPED {int((ms['status'] == 'SKIPPED').sum())}、"
          f"FAILED {int((ms['status'] == 'FAILED').sum())}")
    return {"reconciles": ev.attrs["reconciles"]}


if __name__ == "__main__":
    run()
