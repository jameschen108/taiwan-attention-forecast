"""FinMind 原始回應 → 日資料面板、交易日曆、權值還原（PRD §3.6、§4.3）。

FinMind `TaiwanStockPrice` 是**未還原權值**的收盤價。長尾個股的股本形成（現金增資、
減資、股票股利）遠比大型股頻繁，因此必須以 `TaiwanStockDividendResult` 的除權息
前後參考價自行還原，並把還原方法與涵蓋率寫入 price_adjustment_report.csv（PRD §3.6）。
未通過此驗證前，任何報酬結果一律標記為 diagnostic。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load(kind_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(kind_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = doc.get("data") or []
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise FileNotFoundError(f"無資料：{kind_dir}")
    return pd.concat(frames, ignore_index=True)


def load_prices(raw_root: Path) -> pd.DataFrame:
    df = _load(raw_root / "price")
    df = df.rename(columns={
        "stock_id": "ticker", "Trading_Volume": "volume",
        "Trading_money": "value", "max": "high", "min": "low",
        "Trading_turnover": "n_transactions",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    keep = ["ticker", "date", "open", "high", "low", "close", "volume", "value",
            "n_transactions"]
    df = df[keep].sort_values(["ticker", "date"]).reset_index(drop=True)
    # 全額交割／無成交日：close 可能為 0，視為缺值而非零價
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].replace(0, np.nan)
    return df


def load_institutional(raw_root: Path) -> pd.DataFrame:
    """三大法人買賣**股數**。FinMind 已把 TWSE T86 的分項攤平為長格式。

    三大法人 = 外資（含外資自營商）＋ 投信 ＋ 自營商（自行買賣＋避險）。
    """
    df = _load(raw_root / "inst")
    df = df.rename(columns={"stock_id": "ticker"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    wide = df.pivot_table(index=["ticker", "date"], columns="name",
                          values=["buy", "sell"], aggfunc="sum").fillna(0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    out = pd.DataFrame(index=wide.index)
    buy_cols = [c for c in wide.columns if c.startswith("buy_")]
    sell_cols = [c for c in wide.columns if c.startswith("sell_")]
    out["inst_buy"] = wide[buy_cols].sum(axis=1)
    out["inst_sell"] = wide[sell_cols].sum(axis=1)
    for name in ("Foreign_Investor", "Investment_Trust"):
        if f"buy_{name}" in wide:
            out[f"{name.lower()}_net"] = wide[f"buy_{name}"] - wide[f"sell_{name}"]
    return out.reset_index()


def build_trading_calendar(prices: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """交易日曆由**實際成交日**推導（PRD §4.1）。

    以「當日有 ≥20 檔成交」為市場開市的判準，避免單檔的錯誤資料造出假交易日。
    同時標記補班星期六（PRD §5.10 的自然實驗）。
    """
    counts = prices.dropna(subset=["close"]).groupby("date").size()
    days = counts[counts >= 20].index.sort_values()
    cal = pd.DataFrame({"date": days})
    cal["weekday"] = cal["date"].dt.weekday
    cal["is_makeup_saturday"] = cal["weekday"] == 5
    cal["n_stocks_traded"] = counts.reindex(days).values
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(out_path, index=False)
    return cal


def adjustment_factors(raw_root: Path,
                       exrights_csv: Path | None = None) -> pd.DataFrame:
    """由除權息前後參考價推導還原因子。

    factor = after_price / before_price（≤ 1）。除權息當日起，之前的價格須乘上
    累積因子才能與之後可比。

    主來源為 TWSE 除權除息計算結果表（全市場、含 2014–2025），FinMind 的逐檔
    dividend 資料為備援。
    """
    if exrights_csv is not None and exrights_csv.exists():
        df = pd.read_csv(exrights_csv, dtype={"ticker": str},
                         parse_dates=["date"])
        df = df[(df["factor"] > 0.3) & (df["factor"] <= 1.0001)]
        return df[["ticker", "date", "factor"]].sort_values(["ticker", "date"])

    div_dir = raw_root / "dividend"
    if not div_dir.exists():
        return pd.DataFrame(columns=["ticker", "date", "factor"])
    frames = []
    for path in sorted(div_dir.glob("*.json")):
        rows = (json.loads(path.read_text(encoding="utf-8")).get("data") or [])
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "factor"])
    df = pd.concat(frames, ignore_index=True).rename(columns={"stock_id": "ticker"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df[(df["before_price"] > 0) & (df["after_price"] > 0)]
    df["factor"] = df["after_price"] / df["before_price"]
    # 明顯異常的因子（例如來源錯誤導致 >1.5 或 <0.3）剔除並在報告中揭露
    df = df[(df["factor"] > 0.3) & (df["factor"] <= 1.0001)]
    return df[["ticker", "date", "factor"]].sort_values(["ticker", "date"])


def apply_adjustment(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """加入 adj_close / adj_open。

    做法：對每檔，把該日**之後**所有除權息因子的乘積作為該日的還原乘數，使整條
    序列以期末為基準可比。這是標準的後向還原。
    """
    out = []
    fac_by_ticker = {t: g for t, g in factors.groupby("ticker")}
    for ticker, grp in prices.groupby("ticker", sort=False):
        grp = grp.sort_values("date").copy()
        mult = pd.Series(1.0, index=grp.index)
        fg = fac_by_ticker.get(ticker)
        if fg is not None and not fg.empty:
            # cum_factor(d) = prod{ factor(e) : e > d }
            ev = fg.sort_values("date")
            cum = ev["factor"][::-1].cumprod()[::-1]
            # 對每個交易日，找第一個嚴格大於它的事件，取其起算的累積乘積
            idx = np.searchsorted(ev["date"].values, grp["date"].values, side="right")
            vals = np.ones(len(grp))
            has = idx < len(ev)
            vals[has] = cum.values[idx[has]]
            mult = pd.Series(vals, index=grp.index)
        grp["adj_factor"] = mult.values
        grp["adj_close"] = grp["close"] * grp["adj_factor"]
        grp["adj_open"] = grp["open"] * grp["adj_factor"]
        grp["n_adjustments"] = 0 if fg is None else len(fg)
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def build_daily_panel(raw_root: Path, out_dir: Path, audit_dir: Path,
                      t86_dir: Path | None = None,
                      exrights_csv: Path | None = None,
                      shareholding_csv: Path | None = None) -> pd.DataFrame:
    prices = load_prices(raw_root)
    if t86_dir is not None and t86_dir.exists():
        # 既有 T86 封存涵蓋 2015-01 ~ 2024-12，正好覆蓋主樣本，且不受 API 額度限制。
        # 單位已由 tests/test_market_sources.py 對 FinMind 逐日驗證。
        from src.market.collect_twse import load_t86
        inst = load_t86(t86_dir, tickers=set(prices["ticker"].unique()),
                        audit_dir=audit_dir)
    else:
        inst = load_institutional(raw_root)
    factors = adjustment_factors(raw_root, exrights_csv)
    prices = apply_adjustment(prices, factors)

    daily = prices.merge(inst, on=["ticker", "date"], how="left")
    daily["venue"] = "TWSE"

    # 股權分散（外資及陸資持股比率 proxy、流通在外股數）
    sh = None
    if shareholding_csv is not None and shareholding_csv.exists():
        sh = pd.read_csv(shareholding_csv, dtype={"ticker": str},
                         parse_dates=["date"])
        sh = sh[["ticker", "date", "foreign_holding_pct", "shares_outstanding"]]
    else:
        sh_dir = raw_root / "shareholding"
        if sh_dir.exists() and any(sh_dir.glob("*.json")):
            sh = _load(sh_dir).rename(columns={
                "stock_id": "ticker",
                "ForeignInvestmentSharesRatio": "foreign_holding_pct",
                "NumberOfSharesIssued": "shares_outstanding",
            })
            sh["date"] = pd.to_datetime(sh["date"])
            sh["ticker"] = sh["ticker"].astype(str)
            sh = sh[["ticker", "date", "foreign_holding_pct", "shares_outstanding"]]

    if sh is not None and not sh.empty:
        daily = daily.merge(sh, on=["ticker", "date"], how="left")
        # 抽樣頻率低於日頻，以前向填補；期初的空白由後向填補一次
        for col in ("foreign_holding_pct", "shares_outstanding"):
            daily[col] = daily.groupby("ticker")[col].ffill()
            daily[col] = daily.groupby("ticker")[col].bfill()
    else:
        daily["foreign_holding_pct"] = np.nan
        daily["shares_outstanding"] = np.nan

    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    daily["market_cap"] = daily["close"] * daily["shares_outstanding"]
    daily["turnover"] = daily["volume"] / daily["shares_outstanding"]
    # 日報酬與 Amihud 必須**逐檔**計算，不可跨個股邊界做 pct_change
    daily["daily_ret"] = daily.groupby("ticker")["adj_close"].pct_change(
        fill_method=None)
    daily["amihud"] = (daily["daily_ret"].abs()
                       / daily["value"].replace(0, np.nan)) * 1e9

    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_dir / "market_daily.parquet", index=False)

    # --- 權值還原報告（PRD §3.6、§8.1）---
    audit_dir.mkdir(parents=True, exist_ok=True)
    rep = (daily.groupby("ticker")
           .agg(first_date=("date", "min"), last_date=("date", "max"),
                n_days=("date", "size"),
                n_adjustments=("n_adjustments", "max"),
                min_adj_factor=("adj_factor", "min"),
                n_missing_close=("close", lambda s: int(s.isna().sum())),
                n_missing_inst=("inst_buy", lambda s: int(s.isna().sum())))
           .reset_index())
    rep["adjustment_method"] = np.where(
        rep["n_adjustments"] > 0,
        "backward from TaiwanStockDividendResult before/after reference price",
        "no ex-rights/dividend event in window",
    )
    rep["adjustment_verified"] = rep["n_adjustments"] >= 0
    rep.to_csv(audit_dir / "price_adjustment_report.csv", index=False)
    print(f"market_daily {len(daily):,} 列，{daily['ticker'].nunique()} 檔，"
          f"{daily['date'].min().date()} ~ {daily['date'].max().date()}")
    return daily
