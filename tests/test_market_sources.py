"""兩個來源的單位與欄位對應回歸測試（PRD §6.2 F4、R9）。

「量綱錯誤（金額 vs. 股數、股 vs. 仟股、總額 vs. 淨額）→ 最關鍵的機制變數失效。」
T86 有兩個欄位版本，必須都能正確解析，且與獨立來源（FinMind）逐日對得上。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.market.collect_twse import parse_t86_file

T86_DIR = Path("data/twse/t86")
FINMIND_SAMPLE = Path("data/raw/finmind_inst_sample")

pytestmark = pytest.mark.skipif(not T86_DIR.exists(), reason="需要 T86 封存")


def _finmind_daily(ticker: str) -> pd.DataFrame:
    doc = json.loads((FINMIND_SAMPLE / f"{ticker}.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(doc["data"])
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="name", values=["buy", "sell"],
                          aggfunc="sum").fillna(0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return pd.DataFrame({
        "inst_buy": wide[[c for c in wide if c.startswith("buy_")]].sum(axis=1),
        "inst_sell": wide[[c for c in wide if c.startswith("sell_")]].sum(axis=1),
    })


class TestT86Schemas:
    def test_both_schema_versions_parse(self):
        """16 欄（2015–2017）與 19 欄（2017–2024）兩版都要解析成功。"""
        old = parse_t86_file(T86_DIR / "t86_20150105.json")
        new = parse_t86_file(T86_DIR / "t86_20200302.json")
        assert not old.empty and not new.empty
        for df in (old, new):
            assert (df["inst_buy"] >= 0).all()
            assert (df["inst_sell"] >= 0).all()

    def test_only_four_digit_tickers_kept(self):
        """ETF（006205）、權證等非四碼證券必須排除。"""
        df = parse_t86_file(T86_DIR / "t86_20200302.json")
        assert df["ticker"].str.fullmatch(r"\d{4}").all()

    def test_buy_minus_sell_equals_reported_net(self):
        """守門：分項買賣加總必須等於申報的三大法人買賣超，否則欄位對應錯了。"""
        for name in ("t86_20150105.json", "t86_20200302.json", "t86_20241231.json"):
            path = T86_DIR / name
            if not path.exists():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            idx = {f: i for i, f in enumerate(doc["fields"])}
            net_i = idx["三大法人買賣超股數"]
            df = parse_t86_file(path)
            reported = {}
            for r in doc["data"]:
                t = str(r[0]).strip()
                if len(t) == 4 and t.isdigit():
                    reported[t] = float(str(r[net_i]).replace(",", "") or 0)
            merged = df.assign(reported=df["ticker"].map(reported))
            diff = (merged["inst_buy"] - merged["inst_sell"] - merged["reported"]).abs()
            assert diff.max() < 1.0, f"{name}: 最大差 {diff.max()}"


@pytest.mark.skipif(not FINMIND_SAMPLE.exists() or
                    not list(FINMIND_SAMPLE.glob("*.json")),
                    reason="需要 FinMind 交叉驗證樣本")
class TestCrossSourceUnits:
    def test_t86_matches_finmind_shares_exactly(self):
        """兩個獨立來源的股數必須逐日相等——這是單位正確的最強證據。"""
        tickers = sorted(p.stem for p in FINMIND_SAMPLE.glob("*.json"))[:5]
        sample_dates = ["t86_20200302.json", "t86_20210105.json", "t86_20180103.json"]
        checked = 0
        for name in sample_dates:
            path = T86_DIR / name
            if not path.exists():
                continue
            t86 = parse_t86_file(path).set_index("ticker")
            date = pd.Timestamp(name[4:12])
            for ticker in tickers:
                if ticker not in t86.index:
                    continue
                fm = _finmind_daily(ticker)
                if date not in fm.index:
                    continue
                assert t86.loc[ticker, "inst_buy"] == pytest.approx(
                    fm.loc[date, "inst_buy"], rel=1e-9), f"{ticker}@{date} 買進不符"
                assert t86.loc[ticker, "inst_sell"] == pytest.approx(
                    fm.loc[date, "inst_sell"], rel=1e-9), f"{ticker}@{date} 賣出不符"
                checked += 1
        assert checked >= 5, f"實際比對筆數過少（{checked}）"

    def test_not_off_by_thousand(self):
        """守門：若任一來源以仟股計，比值會是 1000。"""
        ticker = sorted(p.stem for p in FINMIND_SAMPLE.glob("*.json"))[0]
        t86 = parse_t86_file(T86_DIR / "t86_20200302.json").set_index("ticker")
        if ticker not in t86.index:
            pytest.skip("該檔不在此日 T86")
        fm = _finmind_daily(ticker)
        date = pd.Timestamp("2020-03-02")
        if date not in fm.index or fm.loc[date, "inst_buy"] == 0:
            pytest.skip("無可比對值")
        ratio = t86.loc[ticker, "inst_buy"] / fm.loc[date, "inst_buy"]
        assert 0.999 < ratio < 1.001, f"單位比值 {ratio}（疑為股 vs. 仟股）"


# ---------------------------------------------------------------------------
# 股本變動還原（減資）——由 Yahoo 交叉比對揭露的錯誤，回歸測試鎖定
# ---------------------------------------------------------------------------

from src.market.normalize import apply_adjustment  # noqa: E402


class TestCapitalReductionAdjustment:
    """只用除權息表還原會漏掉減資，在減資日產生完全虛假的極端報酬。

    長榮 2603 於 2022-09-19 減資，未還原時單日「上漲」109%——台股漲跌幅上限為
    10%，這在定義上不可能是價格變動。
    """

    def _prices(self):
        dates = pd.to_datetime(["2022-09-15", "2022-09-16", "2022-09-19",
                                "2022-09-20"])
        # 減資前後：股本減半使股價機械性跳升
        return pd.DataFrame({
            "ticker": "2603", "date": dates,
            "open": [81.0, 80.5, 168.0, 170.0],
            "close": [81.0, 80.8, 169.0, 172.5],
        })

    def test_without_reduction_factor_produces_impossible_return(self):
        empty = pd.DataFrame(columns=["ticker", "date", "factor"])
        out = apply_adjustment(self._prices(), empty)
        ret = out["adj_close"].pct_change().iloc[2]
        assert ret > 1.0, "未還原時應出現 >100% 的虛假報酬（此為守門情境）"

    def test_reduction_factor_restores_plausible_return(self):
        factors = pd.DataFrame({
            "ticker": ["2603"], "date": [pd.Timestamp("2022-09-19")],
            "factor": [2.09],   # 恢復買賣參考價 / 停止買賣前收盤價
        })
        out = apply_adjustment(self._prices(), factors)
        ret = out["adj_close"].pct_change().iloc[2]
        assert abs(ret) <= 0.10 + 1e-9, f"還原後應落在漲跌幅上限內，實得 {ret:.4f}"

    def test_factor_direction_is_opposite_to_exrights(self):
        """除權息因子 < 1（股價下調），減資因子 > 1（股價上調）。方向寫反會加倍錯誤。

        少數現金增資（認購價高於市價）的除權參考價會微幅上調，因此檢定的是
        「絕大多數 < 1 且無極端值」，而非全部 < 1。
        """
        ex_path = Path("data/interim/ex_rights.csv")
        rd_path = Path("data/interim/capital_reductions.csv")
        if not ex_path.exists() or not rd_path.exists():
            pytest.skip("需要已收集的事件表")
        ex = pd.read_csv(ex_path)
        rd = pd.read_csv(rd_path)
        assert (ex["factor"] < 1.0).mean() > 0.99, "除權息因子應絕大多數 < 1"
        assert ex["factor"].max() < 1.05, "除權息因子不應出現大幅 > 1"
        assert (rd["factor"] > 1.0).mean() > 0.9, "減資因子應絕大多數 > 1"

    def test_pipeline_factors_merge_both_sources(self):
        """管線實際使用的因子表必須同時含兩類事件。"""
        from src.market.normalize import adjustment_factors
        ex_path = Path("data/interim/ex_rights.csv")
        rd_path = Path("data/interim/capital_reductions.csv")
        if not ex_path.exists() or not rd_path.exists():
            pytest.skip("需要已收集的事件表")
        f = adjustment_factors(Path("data/raw/finmind"), ex_path, rd_path)
        assert (f["factor"] < 1.0).any(), "缺除權息因子"
        assert (f["factor"] > 1.0).any(), "缺減資因子"
        assert not f.duplicated(["ticker", "date"]).any(), "同日事件應合併相乘"


class TestAdjustedReturnsWithinPriceLimit:
    """整體守門：面板中不應留下大量超過漲跌幅上限的日報酬。"""

    def test_few_impossible_daily_returns_after_listing(self):
        path = Path("data/interim/market_daily.parquet")
        if not path.exists():
            pytest.skip("需要已建置的日資料")
        uni = pd.read_csv("data/external/universe.csv", dtype={"ticker": str})
        listing = pd.to_datetime(uni.set_index("ticker")["listing_date"])
        d = pd.read_parquet(path, columns=["ticker", "date", "adj_close"])
        d = d.sort_values(["ticker", "date"])
        d["r"] = d.groupby("ticker")["adj_close"].pct_change(fill_method=None)
        d["listing"] = d["ticker"].map(listing)
        # 只看 TWSE 上市之後（興櫃／上櫃期間無漲跌幅限制）
        post = d[(d["date"] >= d["listing"]) & d["r"].notna()]
        rate = float((post["r"].abs() > 0.11).mean())
        assert rate < 0.0005, f"上市後超過漲跌幅上限的日報酬比例 {rate:.4%} 過高"
