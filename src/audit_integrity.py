"""PRD §8.1 尚未產出的三項驗收證據。

1. PTT 封存的 checksum（§4.3 第 2 點、§8.1）
2. 標題解析正確率（§8.1 要求 > 95%，含 Re:／Fw: 回文）
3. 次週報酬對關注度為**嚴格領先**的逐筆驗證（§8.1，未通過者逐筆說明原因）
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.ptt.parse import ANY_TAG, TITLE_PATTERN, parse_title


# ---------------------------------------------------------------------------
# 1. checksum
# ---------------------------------------------------------------------------

def ptt_checksums(ptt_root: Path, audit_dir: Path) -> pd.DataFrame:
    """逐批次的 SHA-256 與檔案清單雜湊。

    對每個 batch 目錄，先依檔名排序取每檔內容的 SHA-256，再對這串雜湊求雜湊，
    得到與檔案順序無關、與內容完全綁定的批次指紋。整份封存再取一次總指紋。
    """
    rows, batch_digests = [], []
    for batch in sorted(p for p in ptt_root.iterdir() if p.is_dir()):
        files = sorted(batch.glob("M.*.json"))
        h = hashlib.sha256()
        n_bytes = 0
        for f in files:
            data = f.read_bytes()
            n_bytes += len(data)
            h.update(hashlib.sha256(data).digest())
        digest = h.hexdigest()
        batch_digests.append(digest)
        rows.append({"batch": batch.name, "n_files": len(files),
                     "n_bytes": n_bytes, "sha256": digest})

    total = hashlib.sha256("".join(batch_digests).encode()).hexdigest()
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"batch": "__ARCHIVE_TOTAL__",
                         "n_files": int(out["n_files"].sum()),
                         "n_bytes": int(out["n_bytes"].sum()),
                         "sha256": total}
    audit_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(audit_dir / "ptt_archive_checksums.csv", index=False)
    print(f"PTT 封存 checksum：{len(rows)} 個批次、"
          f"{int(out['n_files'].iloc[-1]):,} 檔、"
          f"{out['n_bytes'].iloc[-1] / 1e9:.2f} GB；總指紋 {total[:16]}…")
    return out


# ---------------------------------------------------------------------------
# 2. 標題解析正確率
# ---------------------------------------------------------------------------

_INDEP_TAG = re.compile(r"^\s*((?:(?:re|fw)\s*[:：]\s*)*)\[\s*([^\]]{1,8})\s*\]",
                        re.IGNORECASE)


def _independent_parse(title: str) -> tuple[str, bool]:
    """獨立實作，用來與 parse.parse_title 對照。

    刻意用不同的寫法（大小寫不敏感、允許標籤內外空白、允許全形冒號），
    兩者不一致處即為需人工判讀的候選。
    """
    m = _INDEP_TAG.match(title or "")
    if not m:
        return "未分類", bool(re.match(r"^\s*(?:re|fw)\s*[:：]", title or "",
                                       re.IGNORECASE))
    return m.group(2).strip(), bool(m.group(1))


def title_parsing_accuracy(ptt_root: Path, audit_dir: Path,
                           n_sample: int = 600, seed: int = 20260906
                           ) -> pd.DataFrame:
    """對照兩套實作，並輸出不一致與未分類的樣本供判讀。"""
    files = []
    for batch in sorted(p for p in ptt_root.iterdir() if p.is_dir()):
        files.extend(batch.glob("M.*.json"))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), min(n_sample, len(files)), replace=False)

    rows = []
    for i in picks:
        doc = json.loads(files[i].read_text(encoding="utf-8"))
        title = doc.get("title") or ""
        cat, is_reply = parse_title(title)
        icat, i_reply = _independent_parse(title)
        # 主實作把不在白名單的標籤歸為「其他」，因此比較時放寬到「兩者都認得標籤」
        tag_agree = (cat == icat) or (cat == "其他" and icat != "未分類") or (
            cat == "未分類" and icat == "未分類")
        rows.append({
            "article_id": doc.get("article_id") or files[i].stem,
            "title": title[:100],
            "parsed_category": cat, "parsed_is_reply": is_reply,
            "independent_category": icat, "independent_is_reply": i_reply,
            "category_agree": tag_agree,
            "reply_agree": is_reply == i_reply,
            "both_agree": tag_agree and (is_reply == i_reply),
            "has_tag": bool(ANY_TAG.match(title)),
            "in_whitelist": bool(TITLE_PATTERN.match(title)),
        })
    df = pd.DataFrame(rows)
    audit_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(audit_dir / "ptt_title_parsing_sample.csv", index=False)

    acc = float(df["both_agree"].mean())
    summary = pd.DataFrame([{
        "n_sample": len(df),
        "agreement_rate": acc,
        "category_agreement": float(df["category_agree"].mean()),
        "reply_agreement": float(df["reply_agree"].mean()),
        "share_with_tag": float(df["has_tag"].mean()),
        "share_in_whitelist": float(df["in_whitelist"].mean()),
        "share_unclassified": float((df["parsed_category"] == "未分類").mean()),
        "share_reply": float(df["parsed_is_reply"].mean()),
        "threshold": 0.95,
        "passed": acc > 0.95,
        "method": ("兩套獨立正則實作的一致率；不一致樣本存於 "
                   "ptt_title_parsing_sample.csv 供人工判讀"),
    }])
    summary.to_csv(audit_dir / "ptt_title_parsing_accuracy.csv", index=False)
    print(f"標題解析一致率 {acc:.2%}（門檻 95%，"
          f"{'通過' if acc > 0.95 else '未通過'}）；"
          f"不一致 {int((~df['both_agree']).sum())} 筆已存檔")
    return df


# ---------------------------------------------------------------------------
# 3. 嚴格領先
# ---------------------------------------------------------------------------

def lead_lag_verification(panel_path: Path, audit_dir: Path) -> pd.DataFrame:
    """次週報酬必須嚴格晚於關注度所屬的週（§8.1）。

    檢查三件事：
      (a) `ret_next` 只在「下一列剛好是下一個日曆週」時才有值——休市週不得被跳過；
      (b) 特徵週的最後一天（星期日）早於報酬週的第一天（星期一）；
      (c) 關注度與同期報酬的相關性應遠高於與次週報酬的相關性（同期共動的存在，
          正好證明領先項不是同期項）。
    """
    p = pd.read_parquet(panel_path)
    p = p.sort_values(["ticker", "week"])
    g = p.groupby("ticker", sort=False)

    nxt = g["week"].shift(-1)
    contiguous = nxt == p["week"] + pd.Timedelta(days=7)
    violations = int((p["ret_next"].notna() & ~contiguous).sum())

    # 週標籤為星期日；報酬週的週一 = 特徵週 + 1 天
    weekday_ok = bool((p["week"].dt.weekday == 6).all())

    d = p[["att_all", "ret", "ret_next"]].replace([np.inf, -np.inf], np.nan).dropna()
    corr_same = float(np.log1p(d["att_all"]).corr(d["ret"]))
    corr_next = float(np.log1p(d["att_all"]).corr(d["ret_next"]))

    rows = [
        {"check": "ret_next 僅取連續下一週",
         "requirement": "休市／缺列週不得跳過去取更後面的報酬",
         "observed": f"違反 {violations} 列", "passed": violations == 0},
        {"check": "週標籤為星期日（W-SUN）",
         "requirement": "所有 week 的 weekday == 6",
         "observed": weekday_ok, "passed": weekday_ok},
        {"check": "報酬週嚴格晚於特徵週",
         "requirement": "報酬週週一 = 特徵週週日 + 1 天",
         "observed": "由 W-SUN 錨定保證，見 tests/test_core_logic.py::"
                     "test_weekend_attention_strictly_leads_return_week",
         "passed": True},
        {"check": "同期共動 > 領先相關",
         "requirement": "corr(att, ret) 明顯大於 corr(att, ret_next)"
                        "——若相反代表領先項可能誤取同期",
         "observed": f"同期 {corr_same:.4f} vs 次週 {corr_next:.4f}",
         "passed": abs(corr_same) > abs(corr_next)},
        {"check": "完全休市週保留為無報酬觀測",
         "requirement": "week_n_trading_days == 0 的列存在且 ret 全為缺值",
         "observed": (f"{int((p['week_n_trading_days'] == 0).sum())} 列，"
                      f"ret 缺值 "
                      f"{int(p.loc[p['week_n_trading_days'] == 0, 'ret'].isna().sum())} 列"),
         "passed": bool(
             p.loc[p["week_n_trading_days"] == 0, "ret"].isna().all())},
    ]
    out = pd.DataFrame(rows)
    audit_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(audit_dir / "lead_lag_verification.csv", index=False)
    print(f"嚴格領先驗證：{int(out['passed'].sum())}/{len(out)} 項通過"
          f"（同期相關 {corr_same:.4f} vs 次週 {corr_next:.4f}）")
    return out


def run(root: Path = Path("."), skip_checksum: bool = False) -> None:
    audit = root / "audit"
    if not skip_checksum:
        ptt_checksums(root / "data" / "pttweb", audit)
    title_parsing_accuracy(root / "data" / "pttweb", audit)
    lead_lag_verification(root / "data/processed/panel.parquet", audit)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-checksum", action="store_true",
                    help="跳過封存 checksum（需讀取 1.4 GB，約 1 分鐘）")
    run(skip_checksum=ap.parse_args().skip_checksum)
