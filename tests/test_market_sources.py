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
