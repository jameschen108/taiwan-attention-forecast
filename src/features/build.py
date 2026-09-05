"""建立 ticker×week 面板（PRD §3.1、§3.6、§4 資料契約）。

非平衡且**必須明確非平衡**：上市前的週維持缺列，不得補零關注度、不得補零報酬。
所有領先與落後項依 Sunday-anchored 日曆對齊；完全休市的週保留為一列明確的無報酬觀測。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.attention import build_attention_panel
from src.features.imbalance import abnormal_turnover, weekly_non_inst_roi
from src.features.sessions import week_of


def _weekly_market(daily: pd.DataFrame, trading_days: set,
                   settings: dict) -> pd.DataFrame:
    """日資料 → 週資料：週一開盤至週五收盤報酬、周轉率、ROI、流動性。"""
    imb = settings["imbalance"]
    df = daily.copy()
    df["week"] = df["date"].map(lambda d: week_of(d))

    agg = df.groupby(["ticker", "week"]).agg(
        n_trading_days=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        open_adj=("adj_open", "first"),
        close_adj=("adj_close", "last"),
        volume=("volume", "sum"),
        value=("value", "sum"),
        turnover=("turnover", "sum"),
        shares_outstanding=("shares_outstanding", "last"),
        market_cap=("market_cap", "last"),
        foreign_holding_pct=("foreign_holding_pct", "last"),
        amihud=("amihud", "mean"),
        zero_volume_days=("volume", lambda s: int((s == 0).sum())),
    ).reset_index()

    # 週報酬：週一開盤 → 週五收盤（皆為還原後價格）
    agg["ret"] = agg["close_adj"] / agg["open_adj"] - 1.0
    agg.loc[agg["open_adj"].isna() | agg["close_adj"].isna(), "ret"] = np.nan

    # 非三大法人訂單失衡（PRD §3.7）
    roi_parts = []
    for ticker, grp in df.groupby("ticker", sort=False):
        roi = weekly_non_inst_roi(
            grp[["week", "volume", "inst_buy", "inst_sell"]],
            imb["min_daily_volume_shares"], imb["min_valid_days_per_week"],
        )
        if not roi.empty:
            roi_parts.append(pd.DataFrame({"ticker": ticker, "week": roi.index,
                                           "non_inst_roi": roi.values}))
    if roi_parts:
        agg = agg.merge(pd.concat(roi_parts, ignore_index=True),
                        on=["ticker", "week"], how="left")
    else:
        agg["non_inst_roi"] = np.nan
    return agg


def build_panel(matches_path: Path, daily_path: Path, universe_path: Path,
                trading_days_path: Path, settings_path: Path,
                out_dir: Path, audit_dir: Path) -> pd.DataFrame:
    settings = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    smp = settings["sample"]

    uni = pd.read_csv(universe_path, dtype={"ticker": str})
    uni["listing_date"] = pd.to_datetime(uni["listing_date"])
    matches = pd.read_parquet(matches_path)
    daily = pd.read_parquet(daily_path)
    cal = pd.read_csv(trading_days_path, parse_dates=["date"])
    trading_days = {d.date() for d in cal["date"]}

    # --- 週軸 ---
    start, end = pd.Timestamp(smp["main_start"]), pd.Timestamp(smp["main_end"])
    weeks = pd.date_range(week_of(start), week_of(end), freq="7D")

    tickers = uni["ticker"].tolist()
    att = build_attention_panel(matches, weeks, tickers, settings)
    att = att.reset_index()

    market = _weekly_market(daily, trading_days, settings)

    panel = att.merge(market, on=["ticker", "week"], how="left")
    panel = panel.merge(
        uni[["ticker", "name_short", "sector", "listing_date", "venue", "is_ky"]],
        on="ticker", how="left")

    # --- 非平衡：上市前的週維持缺列（PRD §3.1 第 1 點）---
    n_before = len(panel)
    panel = panel[panel["week"] >= panel["listing_date"].map(week_of)].copy()
    n_dropped = n_before - len(panel)

    # --- 交易日曆屬性 ---
    week_days = (pd.Series(sorted(trading_days))
                 .map(lambda d: week_of(d)).value_counts().sort_index())
    panel["week_n_trading_days"] = panel["week"].map(week_days).fillna(0).astype(int)
    panel["is_incomplete_week"] = (
        panel["week_n_trading_days"] < settings["returns"]["min_trading_days_per_week"])
    makeup = set(cal.loc[cal["is_makeup_saturday"], "date"].map(lambda d: week_of(d)))
    panel["is_makeup_saturday_week"] = panel["week"].isin(makeup)

    # --- 領先項：次週報酬與機制變數（PRD §3.6）---
    panel = panel.sort_values(["ticker", "week"]).reset_index(drop=True)
    g = panel.groupby("ticker", sort=False)
    for src, dst in [("ret", "ret_next"), ("non_inst_roi", "non_inst_roi_next")]:
        # 只在「下一列剛好是下一個日曆週」時才取值，休市週不得跳過（PRD §3.6）
        nxt_week = g["week"].shift(-1)
        ok = nxt_week == panel["week"] + pd.Timedelta(days=7)
        panel[dst] = g[src].shift(-1).where(ok)

    panel["abn_turnover"] = g["turnover"].transform(lambda s: abnormal_turnover(s))
    nxt_week = g["week"].shift(-1)
    ok = nxt_week == panel["week"] + pd.Timedelta(days=7)
    panel["turnover_next"] = g["abn_turnover"].shift(-1).where(ok)

    # --- 落後項：動能控制（PRD §3.8）---
    for lag, name in [(1, "ret_lag1"), (4, "ret_lag4"), (25, "ret_lag25")]:
        if lag == 1:
            panel[name] = g["ret"].shift(1)
        else:
            panel[name] = g["ret"].transform(
                lambda s, k=lag: s.shift(1).rolling(k, min_periods=max(2, k // 2)).mean())
    panel["non_inst_roi_lag1"] = g["non_inst_roi"].shift(1)
    panel["turnover_lag1"] = g["abn_turnover"].shift(1)

    # --- 反轉檢定用的 t+2..t+8 報酬（PRD §5.8）---
    for h in range(2, 9):
        nxt = g["week"].shift(-h)
        ok_h = nxt == panel["week"] + pd.Timedelta(days=7 * h)
        panel[f"ret_fwd{h}"] = g["ret"].shift(-h).where(ok_h)

    # --- 制度斷點（PRD §3.9）---
    inst_cfg = settings["institutions"]
    panel["regime_price_limit_10pct"] = panel["week"] >= pd.Timestamp(
        inst_cfg["price_limit_change"])
    panel["regime_continuous_trading"] = panel["week"] >= pd.Timestamp(
        inst_cfg["continuous_trading"])
    panel["regime_odd_lot"] = panel["week"] >= pd.Timestamp(inst_cfg["odd_lot_trading"])
    panel["listing_age_years"] = (
        (panel["week"] - panel["listing_date"]).dt.days / 365.25)

    # --- 降級標記（PRD §3.4 末段）---
    cfg = yaml.safe_load(Path("config/universe.yaml").read_text(encoding="utf-8"))
    code_only = set(cfg.get("code_only_tickers", []))
    panel["is_code_only_matched"] = panel["ticker"].isin(code_only)

    # --- 產業內外溢（PRD §5.6）---
    sec = panel.groupby(["sector", "week"])["abn_attention_weekend"]
    panel["sector_abn_att_weekend_sum"] = sec.transform("sum")
    panel["sector_n"] = sec.transform("count")
    panel["sector_peer_abn_att_weekend"] = (
        (panel["sector_abn_att_weekend_sum"] - panel["abn_attention_weekend"].fillna(0))
        / (panel["sector_n"] - 1).replace(0, np.nan))
    panel["abn_att_weekend_rel_sector"] = (
        panel["abn_attention_weekend"]
        - panel.groupby(["sector", "week"])["abn_attention_weekend"].transform("mean"))

    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    dense = panel[panel["sparsity_tier"] == "dense"]
    dense.to_parquet(out_dir / "panel_dense.parquet", index=False)

    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "n_rows": len(panel),
        "n_tickers": panel["ticker"].nunique(),
        "n_weeks": panel["week"].nunique(),
        "rows_dropped_pre_listing": n_dropped,
        "first_week": str(panel["week"].min().date()),
        "last_week": str(panel["week"].max().date()),
        "n_dense_rows": int((panel["sparsity_tier"] == "dense").sum()),
        "n_sparse_rows": int((panel["sparsity_tier"] == "sparse").sum()),
        "n_silent_rows": int((panel["sparsity_tier"] == "silent").sum()),
        "n_with_ret_next": int(panel["ret_next"].notna().sum()),
        "n_with_roi_next": int(panel["non_inst_roi_next"].notna().sum()),
    }]).to_csv(audit_dir / "panel_summary.csv", index=False)

    print(f"panel {len(panel):,} 列 × {panel['ticker'].nunique()} 檔 × "
          f"{panel['week'].nunique()} 週；上市前剔除 {n_dropped:,} 列")
    return panel


def main() -> None:
    build_panel(
        Path("data/interim/ptt_matches.parquet"),
        Path("data/interim/market_daily.parquet"),
        Path("data/external/universe.csv"),
        Path("data/interim/trading_days.csv"),
        Path("config/settings.yaml"),
        Path("data/processed"),
        Path("audit"),
    )


if __name__ == "__main__":
    main()
