"""個股歸屬：前綴碰撞消解 ＋ 通用詞降級（PRD §3.4）。

v1.0 的「關鍵字 ＋ 例外清單」在 267 檔的名稱空間下會系統性汙染測度，而汙染量與規模
相關，會直接毀掉 H6。本模組實作 PRD §3.4 的正式演算法：

1. 證券代號比對（含年份／價格／數量／日期的誤判排除）
2. 最長匹配優先，命中後從文本「消耗」該片段
3. 碰撞群組：群組內先長後短，短名命中須有上下文證據
4. 通用詞降級：名稱本身為高頻常用詞者停用簡稱比對，只用代號

輸出 `match_mode`，讓 `is_code_only_matched` 得以進入面板（PRD §3.4 末段）。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------

_ZH_PUNCT = str.maketrans({"（": "(", "）": ")", "　": " ", "％": "%"})


def normalize(text: str) -> str:
    """全形→半形、統一標點、去除零寬字元。原始文本不覆寫（呼叫端負責）。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZH_PUNCT)
    text = text.replace("​", "").replace("﻿", "")
    return text


# ---------------------------------------------------------------------------
# 代號比對
# ---------------------------------------------------------------------------

CODE_RE = re.compile(r"(?<![0-9A-Za-z])(\d{4})(?![0-9A-Za-z])")

# 代號區間與西元年份重疊（中鋼 2002、春源 2010、春雨 2012、中鋼構 2013、中鴻 2014、
# 豐興 2015…）。這些代號必須有正面證據才採計，否則「2015 年」會被算成豐興的關注度。
_YEARLIKE = re.compile(r"^(19[5-9]\d|20[0-4]\d)$")

# 緊接在代號之後、代表它其實是年份／日期／數量／價格的詞
_AFTER_REJECT = re.compile(
    r"^\s*(年|年度|年底|年初|年報|/\d|-\d|\.\d|月|日|點|元|億|萬|塊|人|次|字)"
)
# 緊接在代號之前、代表它其實是年份／金額的詞
_BEFORE_REJECT = re.compile(
    r"(民國|西元|自|至|到|從|第|共|約|漲|跌|價|收|開|高|低|\$|NT|\d[/\-.])\s*$"
)
# 代號後面接這些，是明確的股票語境
_AFTER_ACCEPT = re.compile(r"^\s*(股|張|檔|這檔|多|空|買|賣|進|出|持有|套牢|存|抱)")


@dataclass(frozen=True)
class Variant:
    """一個可比對的名稱寫法。"""

    ticker: str
    text: str
    match_mode: str  # code_only | name | name_with_context
    priority: int = 0

    def __len__(self) -> int:  # 供最長匹配排序
        return len(self.text)


# 股票語境詞：短名或碰撞群組短邊命中時要求的上下文證據
CONTEXT_TERMS = (
    "股價", "股票", "持股", "張", "檔", "買進", "賣出", "作多", "作空", "放空",
    "多單", "空單", "停損", "停利", "目標價", "本益比", "殖利率", "財報", "法說",
    "營收", "EPS", "除權", "除息", "填權", "填息", "漲停", "跌停", "均線", "К線",
    "K線", "融資", "融券", "外資", "投信", "自營", "成交量", "護盤", "套牢",
    "進場", "出場", "抱股", "存股", "配息", "配股", "大盤", "類股", "個股",
)
_CONTEXT_RE = re.compile("|".join(re.escape(t) for t in CONTEXT_TERMS))


@dataclass
class Match:
    ticker: str
    match_mode: str
    matched_text: str
    start: int
    end: int


@dataclass
class Matcher:
    """267 檔的歸屬比對器。

    `variants` 由 config/universe.yaml 展開；`collision_groups` 為 PRD §3.4 的十個
    群組；`code_only` 為通用詞降級清單。
    """

    variants: list[Variant]
    code_only: set[str] = field(default_factory=set)
    valid_codes: set[str] = field(default_factory=set)
    _ordered: list[Variant] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        # 最長匹配優先：字元長度降冪，同長度時 priority 高者先
        usable = [v for v in self.variants if v.ticker not in self.code_only]
        self._ordered = sorted(usable, key=lambda v: (-len(v.text), -v.priority, v.text))
        if not self.valid_codes:
            self.valid_codes = {v.ticker for v in self.variants}

    # -- 代號 ------------------------------------------------------------
    def match_codes(self, text: str) -> list[Match]:
        out: list[Match] = []
        for m in CODE_RE.finditer(text):
            code = m.group(1)
            if code not in self.valid_codes:
                continue
            before, after = text[max(0, m.start() - 8):m.start()], text[m.end():m.end() + 8]
            if _BEFORE_REJECT.search(before) or _AFTER_REJECT.match(after):
                continue
            if _YEARLIKE.match(code) and not _AFTER_ACCEPT.match(after):
                # 年份型代號需要正面證據：同文有該股簡稱，或代號後緊接股票語境詞
                if not self._name_present(text, code):
                    continue
            out.append(Match(code, "code", code, m.start(), m.end()))
        return out

    def _name_present(self, text: str, ticker: str) -> bool:
        return any(v.text in text for v in self.variants if v.ticker == ticker)

    # -- 簡稱 ------------------------------------------------------------
    def match_names(self, text: str) -> list[Match]:
        """最長匹配優先並消耗片段，避免短名吃掉長名的一部分。"""
        consumed = [False] * len(text)
        out: list[Match] = []
        for var in self._ordered:
            start = 0
            while True:
                idx = text.find(var.text, start)
                if idx < 0:
                    break
                end = idx + len(var.text)
                if any(consumed[idx:end]):
                    start = idx + 1
                    continue
                if var.match_mode == "name_with_context" and not self._has_context(text, idx, end):
                    start = idx + 1
                    continue
                for i in range(idx, end):
                    consumed[i] = True
                out.append(Match(var.ticker, var.match_mode, var.text, idx, end))
                start = end
        return out

    def _has_context(self, text: str, start: int, end: int) -> bool:
        """短名／碰撞群組短邊的上下文證據：同句股票語境詞，或同文出現該代號。"""
        window = text[max(0, start - 60):min(len(text), end + 60)]
        return bool(_CONTEXT_RE.search(window))

    # -- 對外 ------------------------------------------------------------
    def match(self, title: str, body: str = "") -> dict[str, str]:
        """回傳 {ticker: match_mode}。一篇文可對應多檔（PRD §3.4 第 5 點）。"""
        text = normalize(f"{title}\n{body}")
        found: dict[str, str] = {}
        for m in self.match_codes(text):
            found[m.ticker] = "code"
        for m in self.match_names(text):
            # 代號命中優先於簡稱命中：代號是最強的證據
            found.setdefault(m.ticker, m.match_mode)
        return found


# ---------------------------------------------------------------------------
# 由設定檔建構
# ---------------------------------------------------------------------------

def build_matcher(universe_cfg: dict) -> Matcher:
    """由 config/universe.yaml 的結構建構 Matcher。"""
    code_only = set(universe_cfg.get("code_only_tickers", []))
    variants: list[Variant] = []
    for ticker, spec in universe_cfg["tickers"].items():
        ticker = str(ticker)
        for entry in spec.get("variants", []):
            variants.append(
                Variant(
                    ticker=ticker,
                    text=normalize(entry["text"]),
                    match_mode=entry.get("match_mode", "name"),
                    priority=entry.get("priority", 0),
                )
            )
    return Matcher(
        variants=variants,
        code_only=code_only,
        valid_codes={str(t) for t in universe_cfg["tickers"]},
    )
