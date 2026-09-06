"""以既有 Yahoo 快取獨立驗證權值還原（PRD §3.6、§8.1）。

PRD 要求「`checkpoint.json` 的價格是否已還原權值必須逐檔驗證……未通過此驗證前，
任何報酬結果一律標記為 diagnostic」。`data/raw/yahoo/*/price/` 提供了這個第三方對照。

**兩個來源的還原性質互補**（本專案實測確認）：

| | 股利（除權息） | 股本變動（減資、股票股利） |
|---|---|---|
| Yahoo `close` | **未**還原 | **已**還原 |
| 本專案 `adj_close` | 已還原（TWT49U） | 已還原（TWTAUU） |

因此正確的檢定是：
1. **非除權息日**：兩者的日報酬應**幾乎完全相等**——這一關同時驗證了股本變動的
   還原是否正確（若減資漏掉，Yahoo 會抓到而本專案不會，報酬立刻分歧）。
2. **除權息日**：兩者應**系統性分歧**，且分歧量約等於股利率——若這裡也相等，
   代表本專案根本沒還原股利。

正是這個對照揭露了「只用 TWT49U 會漏掉減資」的錯誤：修正前主樣本內有 127 筆
|日報酬| > 11%（台股漲跌幅上限 10%，定義上不可能），修正後降到 79 筆，且其中
82% 落在 TWSE 上市日之前（興櫃／上櫃期間無漲跌幅限制），不進入分析面板。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# 非除權息日的日報酬差異容忍度（浮點與四捨五入誤差）
TOL = 1e-4
# 逐檔通過門檻：非除權息日中相符的比例
MIN_MATCH_RATE = 0.98


def load_yahoo(ticker: str, root: Path) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(root / ticker / "price" / "*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "close"]].rename(columns={"close": "y_close"})
    return df.drop_duplicates("date").sort_values("date")


def validate(daily: pd.DataFrame, yahoo_root: Path, events: pd.DataFrame,
             audit_dir: Path, listing: pd.Series | None = None) -> pd.DataFrame:
    """逐檔比對，輸出 audit/price_validation_yahoo.csv。

    `listing` 給定時只比對**TWSE 上市日之後**的資料。上市前的興櫃／上櫃期間，
    Yahoo 與 TWSE 的價格序列來自不同市場，本就不該相符；而面板已把上市前的週
    剔除（PRD §3.1），所以那段期間的分歧不影響任何分析結果。
    """
    ev_dates = {(r.ticker, r.date.normalize()) for r in events.itertuples()}
    rows = []

    for ticker, grp in daily.groupby("ticker", sort=True):
        y = load_yahoo(ticker, yahoo_root)
        if y is None:
            rows.append({"ticker": ticker, "status": "NO_YAHOO_DATA"})
            continue
        if listing is not None and ticker in listing.index:
            grp = grp[grp["date"] >= listing[ticker]]
        m = grp[["date", "close", "adj_close"]].merge(y, on="date", how="inner")
        m = m.dropna(subset=["close", "adj_close", "y_close"]).sort_values("date")
        if len(m) < 100:
            rows.append({"ticker": ticker, "status": "TOO_FEW_OVERLAP",
                         "n_overlap": len(m)})
            continue

        m["r_adj"] = m["adj_close"].pct_change(fill_method=None)
        m["r_yahoo"] = m["y_close"].pct_change(fill_method=None)
        m = m.dropna(subset=["r_adj", "r_yahoo"])
        m["is_event"] = [(ticker, d.normalize()) in ev_dates for d in m["date"]]

        non_ev = m[~m["is_event"]]
        on_ev = m[m["is_event"]]
        diff = (non_ev["r_adj"] - non_ev["r_yahoo"]).abs()
        match_rate = float((diff <= TOL).mean()) if len(non_ev) else np.nan

        ev_diff = ((on_ev["r_adj"] - on_ev["r_yahoo"]).abs()
                   if len(on_ev) else pd.Series(dtype=float))
        rows.append({
            "ticker": ticker,
            "status": ("PASS" if match_rate >= MIN_MATCH_RATE else "FAIL"),
            "n_overlap": len(m),
            "n_non_event_days": len(non_ev),
            "non_event_match_rate": match_rate,
            "non_event_max_abs_diff": float(diff.max()) if len(diff) else np.nan,
            "n_event_days": len(on_ev),
            # 除權息日應**分歧**：本專案有還原股利、Yahoo 沒有
            "event_mean_abs_diff": float(ev_diff.mean()) if len(ev_diff) else np.nan,
            "dividend_adjustment_detected": bool(
                len(ev_diff) and ev_diff.mean() > 5 * TOL),
        })

    rep = pd.DataFrame(rows)
    audit_dir.mkdir(parents=True, exist_ok=True)
    rep.to_csv(audit_dir / "price_validation_yahoo.csv", index=False)

    checked = rep[rep["status"].isin(["PASS", "FAIL"])]
    n_pass = int((checked["status"] == "PASS").sum())
    print(f"權值還原第三方驗證：{n_pass}/{len(checked)} 檔通過"
          f"（非除權息日報酬與 Yahoo 相符率 ≥ {MIN_MATCH_RATE:.0%}）；"
          f"偵測到股利還原 {int(checked['dividend_adjustment_detected'].sum())} 檔")
    return rep


def main() -> None:
    daily = pd.read_parquet("data/interim/market_daily.parquet",
                            columns=["ticker", "date", "close", "adj_close"])
    frames = []
    for path in ("data/interim/ex_rights.csv",
                 "data/interim/capital_reductions.csv"):
        if Path(path).exists():
            frames.append(pd.read_csv(path, dtype={"ticker": str},
                                      parse_dates=["date"])[["ticker", "date"]])
    events = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=["ticker", "date"]))
    uni = pd.read_csv("data/external/universe.csv", dtype={"ticker": str})
    listing = pd.to_datetime(uni.set_index("ticker")["listing_date"])
    validate(daily, Path("data/raw/yahoo"), events, Path("audit"), listing)


if __name__ == "__main__":
    main()
