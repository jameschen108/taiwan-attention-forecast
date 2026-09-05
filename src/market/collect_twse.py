"""TWSE T86 三大法人買賣（PRD §3.7、§4.1、§4.3 第 3 點）。

既有封存 data/twse/t86/ 涵蓋 2015-01 ~ 2024-12，正好覆蓋主樣本期間。

**欄位歷年有改版**，實測有兩個版本：
  2015-01 ~ 2017-12：16 欄，外資為單一欄「外資買進股數」
  2017-12 ~ 2024-12：19 欄，外資拆為「外陸資(不含外資自營商)」與「外資自營商」
因此一律以**欄位名稱**查找，不得用位置索引。任一版本判定錯誤，整篇機制檢定作廢（R9）。

三大法人 = 外資（含外資自營商）＋ 投信 ＋ 自營商（自行買賣 ＋ 避險）。
所有數量單位皆為**股數**（PRD §3.7）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# 各版本的「買進股數」欄位名稱；缺席者視為 0（該版本不存在此分類）
BUY_FIELDS = (
    "外陸資買進股數(不含外資自營商)",
    "外資買進股數",
    "外資自營商買進股數",
    "投信買進股數",
    "自營商買進股數(自行買賣)",
    "自營商買進股數(避險)",
)
SELL_FIELDS = (
    "外陸資賣出股數(不含外資自營商)",
    "外資賣出股數",
    "外資自營商賣出股數",
    "投信賣出股數",
    "自營商賣出股數(自行買賣)",
    "自營商賣出股數(避險)",
)
NET_FIELD = "三大法人買賣超股數"

_NUM = re.compile(r"[^\d\-]")


def _to_shares(value: str) -> float:
    """T86 的數量帶千分位逗號與全形空白，且**單位為股**（非仟股）。"""
    if value is None:
        return 0.0
    s = _NUM.sub("", str(value))
    if s in ("", "-"):
        return 0.0
    return float(s)


def parse_t86_file(path: Path) -> pd.DataFrame:
    doc = json.loads(path.read_text(encoding="utf-8"))
    fields, data = doc.get("fields"), doc.get("data")
    if not fields or not data:
        return pd.DataFrame()

    idx = {name: i for i, name in enumerate(fields)}
    buy_idx = [idx[f] for f in BUY_FIELDS if f in idx]
    sell_idx = [idx[f] for f in SELL_FIELDS if f in idx]
    if not buy_idx or not sell_idx:
        raise RuntimeError(f"T86 欄位無法辨識：{path.name} → {fields}")
    net_idx = idx.get(NET_FIELD)

    m = re.search(r"(\d{8})", path.name)
    date = pd.Timestamp(m.group(1)) if m else pd.Timestamp(doc.get("date"))

    rows = []
    for r in data:
        ticker = str(r[0]).strip()
        if not re.fullmatch(r"\d{4}", ticker):
            continue  # ETF、權證、特別股等非四碼者排除
        buy = sum(_to_shares(r[i]) for i in buy_idx)
        sell = sum(_to_shares(r[i]) for i in sell_idx)
        row = {"ticker": ticker, "date": date, "inst_buy": buy, "inst_sell": sell}
        if net_idx is not None:
            row["inst_net_reported"] = _to_shares(r[net_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def load_t86(t86_dir: Path, tickers: set[str] | None = None,
             audit_dir: Path | None = None) -> pd.DataFrame:
    frames, empty_days, schema_versions = [], [], {}
    files = sorted(t86_dir.glob("t86_*.json"))
    if not files:
        raise FileNotFoundError(f"找不到 T86 封存：{t86_dir}")

    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("fields"):
            schema_versions[len(doc["fields"])] = schema_versions.get(
                len(doc["fields"]), 0) + 1
        df = parse_t86_file(path)
        if df.empty:
            empty_days.append(path.name)
            continue
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # 守門：三大法人淨額必須等於買進減賣出（PRD R9 的量綱檢查）
    if "inst_net_reported" in out.columns:
        computed = out["inst_buy"] - out["inst_sell"]
        mismatch = (computed - out["inst_net_reported"]).abs() > 1.0
        n_bad = int(mismatch.sum())
        if n_bad > 0.001 * len(out):
            raise RuntimeError(
                f"T86 買賣加總與申報淨額不符 {n_bad}/{len(out)} 列——欄位對應可能錯誤")
    else:
        n_bad = 0

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "n_files": len(files),
            "n_empty_days": len(empty_days),
            "n_rows": len(out),
            "n_tickers": out["ticker"].nunique(),
            "first_date": str(out["date"].min().date()),
            "last_date": str(out["date"].max().date()),
            "schema_versions": ";".join(f"{k}欄={v}天" for k, v in
                                        sorted(schema_versions.items())),
            "n_net_mismatch": n_bad,
            "unit": "shares (股)",
        }]).to_csv(audit_dir / "t86_audit.csv", index=False)

    print(f"T86 {len(out):,} 列，{out['ticker'].nunique()} 檔，"
          f"{out['date'].min().date()} ~ {out['date'].max().date()}；"
          f"空白日 {len(empty_days)}；淨額不符 {n_bad} 列")
    return out[["ticker", "date", "inst_buy", "inst_sell"]]
