"""為人工／LLM 抽驗產生逐筆判讀證據（PRD §3.10 第 3 項、§8.1）。

抽驗要回答的是「這篇文章真的在講這一檔嗎」。判讀者需要看到的是**命中的片段與其
上下文**，而不是整篇文章——長尾個股的文章動輒數千字，全文丟給判讀者既昂貴又
容易誤判。本模組對每一筆抽樣輸出：標題、命中片段、命中前後各 120 字的視窗，
以及該檔的代號／簡稱是否出現。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.universe.name_matching import build_matcher, normalize

CONTEXT = 120


def _index_articles(ptt_root: Path) -> dict[str, Path]:
    idx = {}
    for batch in sorted(p for p in ptt_root.iterdir() if p.is_dir()):
        for path in batch.glob("M.*.json"):
            idx[path.stem] = path
    return idx


def build(sample_path: Path, ptt_root: Path, universe_cfg: Path,
          matches_path: Path, out_path: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_path, dtype={"ticker": str})
    cfg = yaml.safe_load(universe_cfg.read_text(encoding="utf-8"))
    matcher = build_matcher(cfg)
    names = {t: [v["text"] for v in spec.get("variants", [])]
             for t, spec in cfg["tickers"].items()}
    short = {t: spec["name_short"] for t, spec in cfg["tickers"].items()}

    matches = pd.read_parquet(matches_path)
    bulk = set(matches.loc[matches["is_bulk_listing"], "article_id"]) \
        if "is_bulk_listing" in matches.columns else set()
    n_tick = (matches.drop_duplicates("article_id")
              .set_index("article_id")["n_tickers_in_article"].to_dict()
              if "n_tickers_in_article" in matches.columns else {})

    index = _index_articles(ptt_root)
    rows = []
    for r in sample.itertuples():
        path = index.get(r.article_id)
        if path is None:
            rows.append({**r._asdict(), "evidence": "<找不到封存檔>"})
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        title, body = doc.get("title") or "", doc.get("body") or ""
        text = normalize(f"{title}\n{body}")

        # 重跑比對以定位命中片段
        spans = []
        for m in matcher.match_codes(text):
            if m.ticker == r.ticker:
                spans.append(("代號", m.matched_text, m.start, m.end))
        for m in matcher.match_names(text):
            if m.ticker == r.ticker:
                spans.append((f"簡稱({m.match_mode})", m.matched_text, m.start, m.end))

        windows = []
        for kind, txt, s, e in spans[:3]:          # 最多три個命中位置
            lo, hi = max(0, s - CONTEXT), min(len(text), e + CONTEXT)
            frag = text[lo:s] + "【" + text[s:e] + "】" + text[e:hi]
            windows.append(f"[{kind}] …{frag.strip()}…")

        rows.append({
            "stratum": r.stratum, "ticker": r.ticker,
            "name_short": short.get(r.ticker, "?"),
            "article_id": r.article_id, "timestamp": r.timestamp,
            "category": r.category, "match_mode": r.match_mode,
            "sparsity_tier": r.sparsity_tier,
            "is_bulk_listing": r.article_id in bulk,
            "n_tickers_in_article": n_tick.get(r.article_id, 0),
            "title": title[:120],
            "n_hits": len(spans),
            "code_present": f"\b{r.ticker}\b" in text or r.ticker in text,
            "name_present": any(n in text for n in names.get(r.ticker, [])),
            "body_len": len(body),
            "evidence": "\n".join(windows) if windows else "<無法定位命中片段>",
            "verdict": "", "note": "",
        })

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"判讀證據 {len(out)} 筆 → {out_path}")
    print(f"  其中大量清單型貼文 {int(out['is_bulk_listing'].sum())} 筆"
          f"（主規格已排除，仍列入抽驗以檢核該旗標）")
    print(f"  無法定位命中片段 {int((out['n_hits'] == 0).sum())} 筆")
    return out


if __name__ == "__main__":
    build(Path("audit/ptt_manual_review_sample.csv"), Path("data/pttweb"),
          Path("config/universe.yaml"),
          Path("data/interim/ptt_matches.parquet"),
          Path("audit/ptt_review_evidence.csv"))
