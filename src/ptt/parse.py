"""PTT 股板文章解析：標題分類、回文判定、認知投入分層（PRD §3.3）。

實測語料的分類分布與 PRD §3.3 的清單不同：另有「情報」（約 11%）、創作、爆卦、
投顧，而 PRD 列出的「問卷」在語料中不存在。分類清單依實得語料擴充。

**推文限制**：封存的 `pushes` 只有內容字串，沒有作者也沒有時戳。PRD §3.3 對此已
預先寫明備案——low_effort 定義為「不含推文」，且不得以文章時戳推估推文時間。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# 依實得語料擴充；PRD §3.3 原清單缺「情報」等
CATEGORIES = (
    "標的|新聞|請益|心得|閒聊|公告|問卷|其他|情報|創作|爆卦|投顧|討論|活動|贈送"
)
TITLE_PATTERN = re.compile(
    rf"^(?P<prefixes>(?:(?:Re|Fw|RE|FW):\s*)*)\[(?P<category>{CATEGORIES})\]"
)
ANY_TAG = re.compile(r"^(?:(?:Re|Fw|RE|FW):\s*)*\[(?P<category>[^\]]{1,6})\]")


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    body: str
    timestamp: int
    category: str
    is_reply: bool
    n_pushes: int
    path: str


def parse_title(title: str) -> tuple[str, bool]:
    """回傳 (category, is_reply)。容許 Re:／Fw: 前綴（回文佔股板貼文大宗）。"""
    title = (title or "").strip()
    m = TITLE_PATTERN.match(title)
    if m:
        return m.group("category"), bool(m.group("prefixes"))
    m2 = ANY_TAG.match(title)
    is_reply = bool(re.match(r"^(?:(?:Re|Fw|RE|FW):\s*)+", title))
    if m2:
        return "其他", is_reply
    return "未分類", is_reply


def effort_tier(category: str, is_reply: bool) -> str:
    """PRD §3.3 的認知投入分層。

    high_effort 嚴格限定 `[標的]` **原PO**（需論述與理由）；`[標的]` 回文歸 mid。
    推文無時戳，一律不進入任何窗口（見模組說明）。
    """
    if category == "標的":
        return "mid_effort" if is_reply else "high_effort"
    if category in ("請益", "心得"):
        return "mid_effort"
    if category in ("新聞", "閒聊", "情報", "公告", "爆卦", "投顧"):
        return "low_effort"
    return "unclassified"


def iter_articles(root: Path) -> Iterator[Article]:
    """走訪 data/pttweb/batch-*/*.json。缺檔或壞檔明確報錯（PRD §4.3 第 5 點）。"""
    batches = sorted(p for p in root.iterdir() if p.is_dir())
    if not batches:
        raise FileNotFoundError(f"找不到 PTT 封存：{root}")
    for batch in batches:
        for path in sorted(batch.glob("M.*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"PTT 封存解析失敗：{path}") from exc

            ts = doc.get("timestamp")
            if ts is None:
                # 少數檔案缺 timestamp 欄位，由 article_id 的 epoch 還原
                m = re.match(r"M\.(\d{10})\.", path.name)
                if not m:
                    raise RuntimeError(f"無法決定時戳：{path}")
                ts = int(m.group(1))

            title = doc.get("title") or ""
            category, is_reply = parse_title(title)
            pushes = doc.get("pushes") or []
            yield Article(
                article_id=doc.get("article_id") or path.stem,
                title=title,
                body=doc.get("body") or "",
                timestamp=int(ts),
                category=category,
                is_reply=is_reply,
                n_pushes=len(pushes),
                path=str(path),
            )
